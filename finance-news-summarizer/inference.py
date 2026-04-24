import json
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_loader import load_prompt_text, render_prompt
from validation import validate_response


def load_model_and_tokenizer(
    model_id,
    adapter_path=None,
    torch_dtype=torch.bfloat16,
):
    """Load a base model and optionally attach a LoRA/QLoRA adapter."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer_source = adapter_path or model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_batch_responses(
    model,
    tokenizer,
    messages,
    max_new_tokens=300,
    do_sample=False,
    temperature=0.2,
):
    """
    Generate batched responses and return both decoded text and token metrics.

    The notebook relied on batched inference for throughput and measured first-pass
    vs repair-pass costs separately. Keeping that behavior in the repo makes the
    training-data generation and adapter evaluation flows consistent.
    """
    texts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    input_ids = inputs["input_ids"]
    input_token_counts = inputs["attention_mask"].sum(dim=1).tolist()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    responses = []
    output_token_counts = []

    for row_index in range(outputs.shape[0]):
        generated_ids = outputs[row_index][input_ids.shape[1] :]
        responses.append(
            tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        )
        output_token_counts.append(len(generated_ids))

    batch_time_sec = end_time - start_time
    total_input_tokens = int(sum(input_token_counts))
    total_output_tokens = int(sum(output_token_counts))
    total_tokens = total_input_tokens + total_output_tokens

    metrics = {
        "batch_size": len(messages),
        "batch_time_sec": batch_time_sec,
        "avg_sec_per_article": batch_time_sec / len(messages),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / batch_time_sec if batch_time_sec > 0 else None,
    }

    del inputs, outputs, input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return responses, metrics


def extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in text")

    return text[start : end + 1]


def build_user_prompt(article_text):
    return render_prompt("inference_user_prompt.txt", article=article_text)


def build_repair_messages(bad_outputs):
    return [
        [
            {
                "role": "system",
                "content": load_prompt_text("repair_system_prompt.txt"),
            },
            {
                "role": "user",
                "content": render_prompt(
                    "repair_user_prompt.txt",
                    bad_output=bad_output,
                ),
            },
        ]
        for bad_output in bad_outputs
    ]


def merge_generation_metrics(first_pass_metrics, repair_metrics=None):
    """Combine first-pass and repair-pass metrics into a single batch view."""
    repair_metrics = repair_metrics or {}

    total_time = first_pass_metrics.get("batch_time_sec", 0) + repair_metrics.get(
        "batch_time_sec", 0
    )
    total_input_tokens = first_pass_metrics.get(
        "input_tokens", 0
    ) + repair_metrics.get("input_tokens", 0)
    total_output_tokens = first_pass_metrics.get(
        "output_tokens", 0
    ) + repair_metrics.get("output_tokens", 0)
    total_tokens = total_input_tokens + total_output_tokens

    return {
        "first_pass_time_sec": first_pass_metrics.get("batch_time_sec", 0),
        "repair_time_sec": repair_metrics.get("batch_time_sec", 0),
        "total_generation_time_sec": total_time,
        "first_pass_input_tokens": first_pass_metrics.get("input_tokens", 0),
        "first_pass_output_tokens": first_pass_metrics.get("output_tokens", 0),
        "repair_input_tokens": repair_metrics.get("input_tokens", 0),
        "repair_output_tokens": repair_metrics.get("output_tokens", 0),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / total_time if total_time > 0 else None,
        "repair_used": bool(repair_metrics),
    }


def summarize_articles_with_validation_batch(
    model,
    tokenizer,
    article_texts,
    max_new_tokens=300,
):
    system_prompt = load_prompt_text("inference_system_prompt.txt")
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(article_text)},
        ]
        for article_text in article_texts
    ]

    raw_responses, first_pass_metrics = generate_batch_responses(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
    )

    results = [None] * len(article_texts)
    failed_indices = []
    repair_payloads = []

    for index, raw_response in enumerate(raw_responses):
        try:
            parsed_output = json.loads(extract_json_object(raw_response))
            is_valid, reason = validate_response(parsed_output)
            if is_valid:
                results[index] = {
                    "success": True,
                    "parsed": parsed_output,
                    "raw_response": raw_response,
                    "repair_used": False,
                    "repair_response": None,
                    "error": None,
                }
                continue
            error_text = reason
        except Exception as exc:
            error_text = str(exc)

        failed_indices.append(index)
        repair_payloads.append((raw_response, error_text))

    repair_metrics = None
    if repair_payloads:
        repair_messages = build_repair_messages(
            [payload[0] for payload in repair_payloads]
        )
        repaired_responses, repair_metrics = generate_batch_responses(
            model,
            tokenizer,
            repair_messages,
            max_new_tokens=max_new_tokens,
        )

        for failed_index, repaired_response, payload in zip(
            failed_indices, repaired_responses, repair_payloads
        ):
            raw_response, original_error = payload
            try:
                repaired_parsed = json.loads(extract_json_object(repaired_response))
                is_valid, reason = validate_response(repaired_parsed)
                if is_valid:
                    results[failed_index] = {
                        "success": True,
                        "parsed": repaired_parsed,
                        "raw_response": raw_response,
                        "repair_used": True,
                        "repair_response": repaired_response,
                        "error": None,
                    }
                else:
                    results[failed_index] = {
                        "success": False,
                        "parsed": None,
                        "raw_response": raw_response,
                        "repair_used": True,
                        "repair_response": repaired_response,
                        "error": reason,
                    }
            except Exception as exc:
                results[failed_index] = {
                    "success": False,
                    "parsed": None,
                    "raw_response": raw_response,
                    "repair_used": True,
                    "repair_response": repaired_response,
                    "error": f"{original_error}; repair failed: {exc}",
                }

    batch_metrics = merge_generation_metrics(first_pass_metrics, repair_metrics)
    batch_metrics["num_articles"] = len(article_texts)
    batch_metrics["num_first_pass_failures"] = len(failed_indices)
    batch_metrics["num_success"] = sum(1 for item in results if item and item["success"])
    batch_metrics["num_failed"] = len(article_texts) - batch_metrics["num_success"]
    batch_metrics["success_rate"] = batch_metrics["num_success"] / len(article_texts)

    return results, batch_metrics


def _coerce_articles(inference_dataset):
    if isinstance(inference_dataset, pd.DataFrame):
        return inference_dataset["article_text"].tolist()
    return [row["article_text"] for row in inference_dataset]


def run_inference(
    model_id,
    inference_dataset,
    batch_size=8,
    adapter_path=None,
    model_name=None,
    torch_dtype=torch.bfloat16,
    gpu_hourly_cost=None,
):
    """
    Run batched summarization, schema repair, and metric collection.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]
            Per-article results, per-batch metrics, and aggregate summary metrics.
    """
    load_start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        adapter_path=adapter_path,
        torch_dtype=torch_dtype,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    model_load_time_sec = time.perf_counter() - load_start
    article_texts_all = _coerce_articles(inference_dataset)
    all_results = []
    all_metrics = []

    inference_start = time.perf_counter()

    for start_index in range(0, len(article_texts_all), batch_size):
        batch_articles = article_texts_all[start_index : start_index + batch_size]
        batch_results, batch_metrics = summarize_articles_with_validation_batch(
            model,
            tokenizer,
            batch_articles,
        )

        for offset, result in enumerate(batch_results):
            parsed = result.get("parsed")
            all_results.append(
                {
                    "id": start_index + offset,
                    "article_text": batch_articles[offset],
                    "success": result["success"],
                    "repair_used": result.get("repair_used", False),
                    "summary": parsed.get("summary") if parsed else None,
                    "key_points": parsed.get("key_points") if parsed else None,
                    "sentiment": parsed.get("sentiment") if parsed else None,
                    "entities": parsed.get("entities") if parsed else None,
                    "why_it_matters": parsed.get("why_it_matters") if parsed else None,
                    "error": result.get("error"),
                }
            )

        all_metrics.append(
            {
                "model": model_name or adapter_path or model_id,
                "batch_id": start_index // batch_size,
                "start_index": start_index,
                "end_index": start_index + len(batch_articles) - 1,
                **batch_metrics,
            }
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_inference_time_sec = time.perf_counter() - inference_start

    results_df = pd.DataFrame(all_results)
    metrics_df = pd.DataFrame(all_metrics)

    num_articles = len(article_texts_all)
    total_input_tokens = metrics_df["total_input_tokens"].sum() if not metrics_df.empty else 0
    total_output_tokens = (
        metrics_df["total_output_tokens"].sum() if not metrics_df.empty else 0
    )
    total_tokens = metrics_df["total_tokens"].sum() if not metrics_df.empty else 0

    summary_metrics = {
        "model": model_name or adapter_path or model_id,
        "num_articles": num_articles,
        "batch_size": batch_size,
        "model_load_time_sec": model_load_time_sec,
        "total_inference_time_sec": total_inference_time_sec,
        "avg_sec_per_article": (
            total_inference_time_sec / num_articles if num_articles else None
        ),
        "articles_per_min": (
            (num_articles / total_inference_time_sec) * 60
            if total_inference_time_sec > 0 and num_articles
            else None
        ),
        "total_input_tokens": int(total_input_tokens),
        "total_output_tokens": int(total_output_tokens),
        "total_tokens": int(total_tokens),
        "tokens_per_sec": (
            total_tokens / total_inference_time_sec
            if total_inference_time_sec > 0
            else None
        ),
    }

    if gpu_hourly_cost is not None and total_inference_time_sec > 0:
        estimated_cost = (total_inference_time_sec / 3600) * gpu_hourly_cost
        summary_metrics["gpu_hourly_cost_assumption"] = gpu_hourly_cost
        summary_metrics["estimated_inference_cost"] = estimated_cost
        summary_metrics["estimated_cost_per_100_articles"] = (
            estimated_cost / num_articles * 100 if num_articles else None
        )

    if torch.cuda.is_available():
        summary_metrics["peak_gpu_memory_gb"] = (
            torch.cuda.max_memory_allocated() / 1024**3
        )

    return results_df, metrics_df, summary_metrics
