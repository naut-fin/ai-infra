# Finance News Summarizer

The intended pipeline is:

1. Run baseline inference with `Qwen/Qwen2.5-3B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`.
2. Keep only schema-valid 3B outputs and use them as teacher labels.
3. Fine-tune the 1.5B model with QLoRA on those labels.
4. Re-run inference with the adapter-attached 1.5B model.
5. Judge baseline vs fine-tuned outputs with a larger evaluation model.

## Notebook Experiment Flow

The notebook is not just “train a QLoRA adapter”. It is a staged experiment:

1. Run a baseline on 100 test samples with `Qwen/Qwen2.5-3B-Instruct`.
2. Run the same 100 test samples with `Qwen/Qwen2.5-1.5B-Instruct`.
3. For both baseline runs, record operational metrics such as:
   - latency
   - token counts
   - throughput
   - repair rate
   - estimated inference cost
4. Use `Qwen/Qwen2.5-7B-Instruct` as a judge model to compare the `3B` and `1.5B` outputs subjectively.
5. The notebook’s subjective evaluation shows `3B` is the overall winner, so `3B` is treated as the stronger teacher model.
6. Use the `3B` model to label 2000 samples from the training split.
7. Use the `3B` model to label 400 samples from the validation split.
8. Keep only schema-valid teacher outputs from those train and validation labeling runs.
9. Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with QLoRA on those `3B`-generated labels.
10. After training, run inference again on the same earlier 100 test samples, now using the `1.5B + QLoRA adapter` model.
11. Use the `7B` judge model again, this time comparing outputs from all three systems:
    - `3B`
    - `1.5B`
    - `1.5B QLoRA`
12. The final purpose of the experiment is to determine whether QLoRA fine-tuning helps the `1.5B` model close the gap with the stronger `3B` baseline.

That experiment sequence is the right mental model for this folder. The code here should support that end-to-end comparison, not just isolated training utilities.

## Hardware Assumptions

The training path is intentionally CUDA-first.

- QLoRA here assumes Linux + NVIDIA GPU + `bitsandbytes`.
- macOS/MPS is not the target path for this setup.
- The 7B judge model and 4-bit QLoRA training both require substantial VRAM.

That is consistent with the notebook you shared. I did not try to force this into an MPS-friendly implementation because that would be a different project.

## Expected Workflow

### 1. Baseline inference

Run baseline models over train/val/test splits and save the resulting CSVs. The key entry point is `run_inference(...)` in [inference.py](/Users/sidharthagarwal/ai-infra/finance-news-summarizer/inference.py).

Use:

- `Qwen/Qwen2.5-3B-Instruct` as the teacher/baseline
- `Qwen/Qwen2.5-1.5B-Instruct` as the smaller baseline

`run_inference(...)` returns:

- a per-article results dataframe
- a per-batch metrics dataframe
- a summary metrics dict

This corresponds to the notebook stages where:

- `Qwen/Qwen2.5-3B-Instruct` is evaluated on 100 test samples
- `Qwen/Qwen2.5-1.5B-Instruct` is evaluated on the same 100 test samples
- the same 3B model is then used again to label train and validation samples for supervised fine-tuning data

The notebook also tracks:

- schema-valid output rate
- repair-needed rate
- per-batch latency and token counts
- end-to-end inference cost estimates based on an assumed GPU hourly price

That behavior is now centralized in `run_inference(...)` and the batch repair path in [inference.py](/Users/sidharthagarwal/ai-infra/finance-news-summarizer/inference.py).

### 2. Build teacher labels

Once you have the 3B outputs:

- keep only rows where `success == True`
- convert stringified lists back into Python lists
- build `target_json`

Use `build_teacher_training_dataframe(...)` or `build_teacher_training_dataset(...)` from [data_processing.py](/Users/sidharthagarwal/ai-infra/finance-news-summarizer/data_processing.py).

This step is important because the notebook does not fine-tune on human-written summaries. It fine-tunes the 1.5B model on clean 3B-generated JSON outputs, so the 3B model acts as the teacher.

The notebook did this separately for:

- a 2000-sample training export from the train split
- a 400-sample validation export from the validation split

Only rows with valid schema should be retained for both.

### 3. Train the QLoRA adapter

Training is centered around:

- `load_training_components(...)`
- `prepare_training_dataset(...)`
- `build_trainer(...)`

A minimal flow is:

```python
import pandas as pd

from data_processing import build_teacher_training_dataset
from training import (
    build_trainer,
    load_training_components,
    prepare_training_dataset,
)

train_df = pd.read_csv("results_3B_train_filtered.csv")
val_df = pd.read_csv("results_3B_val_filtered.csv")

model, tokenizer, training_args = load_training_components()

train_dataset = prepare_training_dataset(
    build_teacher_training_dataset(train_df),
    tokenizer,
)
val_dataset = prepare_training_dataset(
    build_teacher_training_dataset(val_df),
    tokenizer,
)

trainer = build_trainer(
    model=model,
    training_args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.model.save_pretrained("artifacts/qlora-finance-summarizer")
tokenizer.save_pretrained("artifacts/qlora-finance-summarizer")
```

The notebook-specific training choices that should remain explicit are:

- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- quantization: 4-bit QLoRA with `nf4`
- double quantization: enabled
- compute dtype: `torch.bfloat16`
- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.05`
- target modules: `"all-linear"`
- batch size per device: `4`
- gradient accumulation steps: `8`
- effective batch size: `32`
- learning rate: `2e-4`
- epochs: `5`
- checkpoint/eval cadence: per epoch
- early stopping patience: `2`

The notebook also trained with response-only loss. That is why [training.py](/Users/sidharthagarwal/ai-infra/finance-news-summarizer/training.py) creates both `prompt_text` and `full_text`, then masks prompt tokens with `-100` so the model learns only from the assistant JSON.

### 4. Run inference with the fine-tuned adapter

Point `run_inference(...)` at:

- `model_id="Qwen/Qwen2.5-1.5B-Instruct"`
- `adapter_path="artifacts/qlora-finance-summarizer"`

That reproduces the notebook’s “base model + adapter” inference flow while keeping loading and metrics in one place.

In the notebook, this stage also records:

- model load time
- total inference time
- articles per minute
- total input and output tokens
- peak GPU memory
- estimated cost per 100 and per 1000 articles

Those fields are now part of the summary metrics returned by `run_inference(...)` when you pass a `gpu_hourly_cost`.

### 5. Judge output quality

Use [evaluate.py](/Users/sidharthagarwal/ai-infra/finance-news-summarizer/evaluate.py):

- `build_pairwise_comparison_frame(df_3b, df_1_5b)`
- `run_judge(comparison_df, judge_model_id="Qwen/Qwen2.5-7B-Instruct")`

This is the cleaned-up equivalent of the notebook’s judge-model evaluation stage.

The notebook actually has two judge stages:

1. Pairwise comparison of `3B` vs `1.5B`
2. Final comparison of `3B` vs `1.5B` vs `1.5B QLoRA`

The repo now includes both:

- pairwise judging for `3B` vs `1.5B`
- three-way judging for `3B` vs `1.5B` vs `1.5B QLoRA`

The notebook’s final evaluation question is not just “which model wins”. It is:

- whether `1.5B QLoRA` is better than untuned `1.5B`
- whether that improvement is enough to meaningfully reduce the gap to `3B`

## Validation And Repair

One important notebook behavior that should stay documented is the two-pass inference design:

1. Generate the structured JSON summary.
2. Validate it against the required schema.
3. If it is malformed or fails validation, run a repair prompt to coerce it back into valid JSON.

That flow is implemented in [inference.py](/Users/sidharthagarwal/ai-infra/finance-news-summarizer/inference.py) and validated by [validation.py](/Users/sidharthagarwal/ai-infra/finance-news-summarizer/validation.py).

The required response schema is:

```json
{
  "summary": "string",
  "key_points": ["string", "string", "string"],
  "sentiment": "positive | negative | neutral",
  "entities": ["string"],
  "why_it_matters": "string"
}
```

Validation currently enforces:

- exact key match
- non-empty `summary`
- `key_points` length between 3 and 5
- `sentiment` in `positive|negative|neutral`
- `entities` as a list of non-empty strings
- non-empty `why_it_matters`

## Install

From this folder:

```bash
cd finance-news-summarizer
pip install -r requirements.txt
```

The current modules use local imports such as `from inference import ...`, so run scripts and notebooks from inside `finance-news-summarizer` unless you later convert the folder into a proper installable package with package-relative imports.

## Quick Start

### Full notebook-style run

From inside `finance-news-summarizer`, you can now run the full experiment end to end with one command:

```bash
python full_pipeline.py
```

This executes the notebook flow verbatim:

1. run 100-sample baseline inference for `3B`
2. run 100-sample baseline inference for `1.5B`
3. judge `3B` vs `1.5B` with `7B`
4. label 2000 train samples with `3B`
5. label 400 validation samples with `3B`
6. train the `1.5B` QLoRA adapter on those teacher labels
7. rerun the same 100 test samples with the fine-tuned adapter
8. judge `3B` vs `1.5B` vs `1.5B QLoRA` with `7B`

Useful flags:

- `--output-dir artifacts/full-pipeline`
- `--test-samples 100`
- `--train-teacher-samples 2000`
- `--val-teacher-samples 400`
- `--inference-batch-size 8`
- `--judge-batch-size 4`
- `--gpu-hourly-cost 0.171`

### Baseline 3B / 1.5B inference

```python
from data_processing import get_datasets
from inference import run_inference

train_dataset, val_dataset, test_dataset = get_datasets()

results_3b, metrics_3b, summary_3b = run_inference(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    inference_dataset=test_dataset,
    batch_size=8,
    model_name="Qwen2.5-3B-Instruct",
)

results_1_5b, metrics_1_5b, summary_1_5b = run_inference(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    inference_dataset=test_dataset,
    batch_size=8,
    model_name="Qwen2.5-1.5B-Instruct",
)

results_3b.to_csv("results_3B.csv", index=False)
metrics_3b.to_csv("results_3B_metrics.csv", index=False)

results_1_5b.to_csv("results_1_5B.csv", index=False)
metrics_1_5b.to_csv("results_1_5B_metrics.csv", index=False)
```

### Fine-tuned adapter inference

```python
import pandas as pd

from inference import run_inference

test_df = pd.read_csv("results_1_5B.csv")

results_qlora, metrics_qlora, summary_qlora = run_inference(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path="artifacts/qlora-finance-summarizer",
    inference_dataset=test_df,
    batch_size=8,
    model_name="Qwen2.5-1.5B-Instruct-QLoRA",
)

results_qlora.to_csv("results_1_5B_QLoRA.csv", index=False)
metrics_qlora.to_csv("results_1_5B_QLoRA_metrics.csv", index=False)
pd.DataFrame([summary_qlora]).to_csv("results_1_5B_QLoRA_summary_metrics.csv", index=False)
```

### Judge baseline outputs

```python
import pandas as pd

from evaluate import build_pairwise_comparison_frame, run_judge

df_3b = pd.read_csv("results_3B.csv")
df_1_5b = pd.read_csv("results_1_5B.csv")

comparison_df = build_pairwise_comparison_frame(df_3b, df_1_5b)
judge_df = run_judge(comparison_df)
judge_df.to_csv("judge_comparison_results.csv", index=False)
```

## Expected Artifacts

If you follow the notebook-style workflow, these are the main files you should persist between stages:

- `results_3B.csv`: teacher-model outputs used for comparison and training-label generation.
- `results_1_5B.csv`: untuned small-model baseline outputs.
- `results_3B_metrics.csv` and `results_1_5B_metrics.csv`: batch-level inference metrics.
- `results_3B_train_filtered.csv` and `results_3B_val_filtered.csv`: clean teacher-labeled subsets used for QLoRA.
- `artifacts/qlora-finance-summarizer/`: saved adapter weights and tokenizer.
- `results_1_5B_QLoRA.csv`: inference outputs from the fine-tuned adapter.
- `results_1_5B_QLoRA_metrics.csv`: batch metrics for adapter inference.
- `results_1_5B_QLoRA_summary_metrics.csv`: aggregate throughput, token, and cost metrics for adapter inference.
- `judge_comparison_results.csv`: evaluation output from the 7B judge model.

## Current Gaps

This refactor now includes a full pipeline runner and three-way judging, but there are still some real gaps.

- There are no smoke tests yet for prompt rendering, schema validation, or JSON repair.
- The pipeline is runnable end to end, but it is still a script-driven workflow rather than a packaged CLI tool with resumable stages.
- Failed long-running stages do not automatically resume from saved artifacts; reruns are manual.
- The code still uses local imports, so execution should happen from inside `finance-news-summarizer` unless the project is converted to a proper package.

## Notes

- The code assumes the Hugging Face models are accessible in the execution environment.
- Dataset download still comes from `ashraq/financial-news-articles`.
- For long-term maintainability, the next step should be adding small smoke tests around validation, JSON extraction, and prompt rendering.
