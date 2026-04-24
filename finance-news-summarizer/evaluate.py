import json

import pandas as pd
import torch

from inference import extract_json_object, generate_batch_responses, load_model_and_tokenizer
from prompt_loader import load_prompt_text, render_prompt


def build_pairwise_comparison_frame(df_3b, df_1_5b):
    merged_df = df_3b.merge(
        df_1_5b,
        on=["id", "article_text"],
        suffixes=("_3b", "_1_5b"),
    )
    return merged_df[
        (merged_df["success_3b"] == True) & (merged_df["success_1_5b"] == True)
    ].copy()


def build_three_way_comparison_frame(df_3b, df_1_5b, df_1_5b_qlora):
    merged_df = build_pairwise_comparison_frame(df_3b, df_1_5b)
    merged_df = merged_df.merge(
        df_1_5b_qlora,
        on=["id", "article_text"],
        suffixes=("", "_1_5b_qlora"),
    )
    return merged_df[merged_df["success"] == True].copy()


def build_judge_prompt(article_text, row):
    return render_prompt(
        "evaluation_user_prompt.txt",
        article_text=article_text,
        summary_3b=row["summary_3b"],
        key_points_3b=row["key_points_3b"],
        sentiment_3b=row["sentiment_3b"],
        why_it_matters_3b=row["why_it_matters_3b"],
        summary_1_5b=row["summary_1_5b"],
        key_points_1_5b=row["key_points_1_5b"],
        sentiment_1_5b=row["sentiment_1_5b"],
        why_it_matters_1_5b=row["why_it_matters_1_5b"],
    )


def build_three_way_judge_prompt(article_text, row):
    return render_prompt(
        "evaluation_user_prompt_three_way.txt",
        article_text=article_text,
        summary_3b=row["summary_3b"],
        key_points_3b=row["key_points_3b"],
        sentiment_3b=row["sentiment_3b"],
        why_it_matters_3b=row["why_it_matters_3b"],
        summary_1_5b=row["summary_1_5b"],
        key_points_1_5b=row["key_points_1_5b"],
        sentiment_1_5b=row["sentiment_1_5b"],
        why_it_matters_1_5b=row["why_it_matters_1_5b"],
        summary_1_5b_qlora=row["summary"],
        key_points_1_5b_qlora=row["key_points"],
        sentiment_1_5b_qlora=row["sentiment"],
        why_it_matters_1_5b_qlora=row["why_it_matters"],
    )


def parse_judge_response(text):
    return json.loads(extract_json_object(text))


def _run_judge_batches(
    comparison_df,
    prompt_builder,
    judge_model_id,
    batch_size,
    torch_dtype,
):
    """Load the judge model once and score the dataframe in batches."""
    judge_model, judge_tokenizer = load_model_and_tokenizer(
        model_id=judge_model_id,
        torch_dtype=torch_dtype,
    )

    judge_results = []
    system_prompt = load_prompt_text("evaluation_system_prompt.txt")

    for start_index in range(0, len(comparison_df), batch_size):
        batch_df = comparison_df.iloc[start_index : start_index + batch_size]
        prompts = [
            prompt_builder(row["article_text"], row)
            for _, row in batch_df.iterrows()
        ]
        messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]

        raw_responses, _ = generate_batch_responses(
            judge_model,
            judge_tokenizer,
            messages,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.1,
        )

        for (_, row), raw_response in zip(batch_df.iterrows(), raw_responses):
            try:
                parsed = parse_judge_response(raw_response)
                judge_results.append(
                    {
                        "id": row["id"],
                        "judge_success": True,
                        "judge_raw_response": raw_response,
                        **parsed,
                    }
                )
            except Exception as exc:
                judge_results.append(
                    {
                        "id": row["id"],
                        "judge_success": False,
                        "judge_raw_response": raw_response,
                        "judge_error": str(exc),
                    }
                )

    judge_df = pd.DataFrame(judge_results)
    return comparison_df.merge(judge_df, on="id", how="left")


def run_judge(
    comparison_df,
    judge_model_id="Qwen/Qwen2.5-7B-Instruct",
    batch_size=4,
    torch_dtype=torch.bfloat16,
):
    """Run pairwise judging for the 3B and 1.5B baselines."""
    return _run_judge_batches(
        comparison_df=comparison_df,
        prompt_builder=build_judge_prompt,
        judge_model_id=judge_model_id,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
    )


def run_three_way_judge(
    comparison_df,
    judge_model_id="Qwen/Qwen2.5-7B-Instruct",
    batch_size=4,
    torch_dtype=torch.bfloat16,
):
    """Run the notebook's final three-model evaluation stage."""
    return _run_judge_batches(
        comparison_df=comparison_df,
        prompt_builder=build_three_way_judge_prompt,
        judge_model_id=judge_model_id,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
    )
