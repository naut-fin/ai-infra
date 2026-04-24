from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def build_bnb_config():
    """Mirror the QLoRA setup from the notebook for Linux + NVIDIA training."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_training_components(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    output_dir="artifacts/qlora-finance-summarizer",
):
    """
    Load the quantized base model and tokenizer used for QLoRA fine-tuning.

    This path intentionally assumes CUDA-compatible Linux hardware. The notebook
    already made that assumption, and keeping it explicit is better than implying
    the flow is usable on macOS/MPS.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    return model, tokenizer, training_args


def format_training_example(example, tokenizer):
    """Build prompt-only and prompt+answer chat transcripts for response-only loss."""
    system_prompt = """You are a financial news summarization assistant.
Your task is to read a financial news article and return a concise structured summary.

Rules:
1. Return valid JSON only.
2. Do not include markdown fences.
3. Do not include any text before or after the JSON.
4. Be factual and concise.
5. Do not invent facts not present in the article.
6. If sentiment is unclear, use "neutral".
7. Sentiment must be exactly one of: "positive", "negative", "neutral".
8. The "key_points" field must contain 3 to 5 short bullet-style strings.
9. The "entities" field should include important companies, people, sectors, products, or macro topics mentioned in the article.
10. The "why_it_matters" field should explain the practical investor or business relevance in 1 to 2 sentences."""

    user_prompt = f"""Read the following financial news article and return a JSON object with exactly these keys:

- summary
- key_points
- sentiment
- entities
- why_it_matters

JSON schema:
{{
  "summary": "string",
  "key_points": ["string", "string", "string"],
  "sentiment": "positive | negative | neutral",
  "entities": ["string"],
  "why_it_matters": "string"
}}

Article:
<article>
{example["article_text"]}
</article>
"""

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": example["target_json"]}
    ]

    return {
        "prompt_text": tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        ),
        "full_text": tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        ),
    }


def tokenize_and_mask(example, tokenizer, max_length=1024):
    """
    Train only on assistant tokens by masking prompt tokens with -100.

    This matches the intent in the notebook while keeping the implementation
    explicit and easy to debug outside Colab.
    """
    full = tokenizer(
        example["full_text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    prompt = tokenizer(
        example["prompt_text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    labels = full["input_ids"].copy()
    prompt_len = sum(1 for token in prompt["attention_mask"] if token == 1)
    labels[:prompt_len] = [-100] * prompt_len
    labels = [
        -100 if full["attention_mask"][index] == 0 else labels[index]
        for index in range(len(labels))
    ]

    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


def prepare_training_dataset(dataset, tokenizer, max_length=1024):
    formatted = dataset.map(
        lambda row: format_training_example(row, tokenizer),
    )
    tokenized = formatted.map(
        lambda row: tokenize_and_mask(row, tokenizer, max_length=max_length),
        remove_columns=formatted.column_names,
    )
    return tokenized


def build_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )


def build_trainer(model, training_args, train_dataset, eval_dataset):
    peft_model = get_peft_model(model, build_lora_config())
    return Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
