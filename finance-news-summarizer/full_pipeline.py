import argparse
import json
from pathlib import Path

import pandas as pd

from data_processing import build_teacher_training_dataset, get_datasets
from evaluate import (
    build_pairwise_comparison_frame,
    build_three_way_comparison_frame,
    run_judge,
    run_three_way_judge,
)
from inference import run_inference
from training import build_trainer, load_training_components, prepare_training_dataset


DEFAULT_3B_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_1_5B_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_7B_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _select_n(dataset, size):
    return dataset.select(range(min(size, len(dataset))))


def _write_inference_artifacts(results_df, metrics_df, summary_metrics, output_prefix):
    results_path = Path(f"{output_prefix}.csv")
    metrics_path = Path(f"{output_prefix}_metrics.csv")
    summary_path = Path(f"{output_prefix}_summary_metrics.csv")

    results_df.to_csv(results_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame([summary_metrics]).to_csv(summary_path, index=False)


def _winner_counts(df, columns):
    summary = {}
    for column in columns:
        if column in df.columns:
            summary[column] = df[column].value_counts(dropna=False).to_dict()
    return summary


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the finance-news-summarizer notebook workflow end to end.",
    )
    parser.add_argument("--output-dir", default="artifacts/full-pipeline")
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--train-teacher-samples", type=int, default=2000)
    parser.add_argument("--val-teacher-samples", type=int, default=400)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--gpu-hourly-cost", type=float, default=0.171)
    parser.add_argument("--baseline-3b-model", default=DEFAULT_3B_MODEL)
    parser.add_argument("--baseline-1-5b-model", default=DEFAULT_1_5B_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_7B_JUDGE_MODEL)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset = get_datasets()
    test_subset = _select_n(test_dataset, args.test_samples)
    train_teacher_subset = _select_n(train_dataset, args.train_teacher_samples)
    val_teacher_subset = _select_n(val_dataset, args.val_teacher_samples)

    run_manifest = {
        "baseline_3b_model": args.baseline_3b_model,
        "baseline_1_5b_model": args.baseline_1_5b_model,
        "judge_model": args.judge_model,
        "test_samples": len(test_subset),
        "train_teacher_samples": len(train_teacher_subset),
        "val_teacher_samples": len(val_teacher_subset),
        "inference_batch_size": args.inference_batch_size,
        "judge_batch_size": args.judge_batch_size,
        "gpu_hourly_cost": args.gpu_hourly_cost,
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)

    results_3b, metrics_3b, summary_3b = run_inference(
        model_id=args.baseline_3b_model,
        inference_dataset=test_subset,
        batch_size=args.inference_batch_size,
        model_name="Qwen2.5-3B-Instruct",
        gpu_hourly_cost=args.gpu_hourly_cost,
    )
    _write_inference_artifacts(results_3b, metrics_3b, summary_3b, output_dir / "results_3B")

    results_1_5b, metrics_1_5b, summary_1_5b = run_inference(
        model_id=args.baseline_1_5b_model,
        inference_dataset=test_subset,
        batch_size=args.inference_batch_size,
        model_name="Qwen2.5-1.5B-Instruct",
        gpu_hourly_cost=args.gpu_hourly_cost,
    )
    _write_inference_artifacts(
        results_1_5b,
        metrics_1_5b,
        summary_1_5b,
        output_dir / "results_1_5B",
    )

    pairwise_comparison_df = build_pairwise_comparison_frame(results_3b, results_1_5b)
    pairwise_judge_df = run_judge(
        pairwise_comparison_df,
        judge_model_id=args.judge_model,
        batch_size=args.judge_batch_size,
    )
    pairwise_judge_df.to_csv(output_dir / "judge_comparison_results.csv", index=False)
    _write_json(
        output_dir / "judge_comparison_summary.json",
        _winner_counts(
            pairwise_judge_df,
            [
                "overall_winner",
                "summary_winner",
                "sentiment_winner",
                "why_it_matters_winner",
            ],
        ),
    )

    results_3b_train, metrics_3b_train, summary_3b_train = run_inference(
        model_id=args.baseline_3b_model,
        inference_dataset=train_teacher_subset,
        batch_size=args.inference_batch_size,
        model_name="Qwen2.5-3B-Instruct-Teacher-Train",
        gpu_hourly_cost=args.gpu_hourly_cost,
    )
    _write_inference_artifacts(
        results_3b_train,
        metrics_3b_train,
        summary_3b_train,
        output_dir / "results_3B_train",
    )

    results_3b_val, metrics_3b_val, summary_3b_val = run_inference(
        model_id=args.baseline_3b_model,
        inference_dataset=val_teacher_subset,
        batch_size=args.inference_batch_size,
        model_name="Qwen2.5-3B-Instruct-Teacher-Val",
        gpu_hourly_cost=args.gpu_hourly_cost,
    )
    _write_inference_artifacts(
        results_3b_val,
        metrics_3b_val,
        summary_3b_val,
        output_dir / "results_3B_val",
    )

    train_teacher_df = results_3b_train[results_3b_train["success"] == True].copy()
    val_teacher_df = results_3b_val[results_3b_val["success"] == True].copy()
    train_teacher_df.to_csv(output_dir / "results_3B_train_filtered.csv", index=False)
    val_teacher_df.to_csv(output_dir / "results_3B_val_filtered.csv", index=False)

    if train_teacher_df.empty or val_teacher_df.empty:
        raise RuntimeError(
            "Teacher labeling produced no valid training rows. "
            "Inspect the saved 3B train/val inference artifacts before retrying."
        )

    adapter_dir = output_dir / "qlora_finance_summarizer"
    checkpoints_dir = output_dir / "training_checkpoints"
    model, tokenizer, training_args = load_training_components(
        model_id=args.baseline_1_5b_model,
        output_dir=str(checkpoints_dir),
    )

    train_dataset_for_training = prepare_training_dataset(
        build_teacher_training_dataset(train_teacher_df),
        tokenizer,
    )
    val_dataset_for_training = prepare_training_dataset(
        build_teacher_training_dataset(val_teacher_df),
        tokenizer,
    )

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset_for_training,
        eval_dataset=val_dataset_for_training,
    )
    train_result = trainer.train()

    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    _write_json(
        output_dir / "training_metrics.json",
        train_result.metrics,
    )

    results_qlora, metrics_qlora, summary_qlora = run_inference(
        model_id=args.baseline_1_5b_model,
        adapter_path=str(adapter_dir),
        inference_dataset=test_subset,
        batch_size=args.inference_batch_size,
        model_name="Qwen2.5-1.5B-Instruct-QLoRA",
        gpu_hourly_cost=args.gpu_hourly_cost,
    )
    _write_inference_artifacts(
        results_qlora,
        metrics_qlora,
        summary_qlora,
        output_dir / "results_1_5B_QLoRA",
    )

    three_way_comparison_df = build_three_way_comparison_frame(
        results_3b,
        results_1_5b,
        results_qlora,
    )
    three_way_judge_df = run_three_way_judge(
        three_way_comparison_df,
        judge_model_id=args.judge_model,
        batch_size=args.judge_batch_size,
    )
    three_way_judge_df.to_csv(output_dir / "judge_three_way_results.csv", index=False)
    _write_json(
        output_dir / "judge_three_way_summary.json",
        _winner_counts(
            three_way_judge_df,
            [
                "overall_winner",
                "summary_winner",
                "sentiment_winner",
                "why_it_matters_winner",
            ],
        ),
    )


if __name__ == "__main__":
    main()
