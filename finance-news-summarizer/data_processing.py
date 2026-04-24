import ast
import json

import pandas as pd
from datasets import Dataset, load_dataset


DATASET_NAME = "ashraq/financial-news-articles"


def normalize_example(example):
    return {"article_text": example["text"]}


def get_datasets(dataset_name=DATASET_NAME, seed=42):
    """Return train/validation/test splits with a normalized article_text field."""
    dataset = load_dataset(dataset_name)["train"]
    dataset = dataset.map(normalize_example, remove_columns=dataset.column_names)

    first_split = dataset.train_test_split(test_size=0.2, seed=seed)
    train_split = first_split["train"]
    test_dataset = first_split["test"]

    second_split = train_split.train_test_split(test_size=0.125, seed=seed)
    train_dataset = second_split["train"]
    val_dataset = second_split["test"]

    return train_dataset, val_dataset, test_dataset


def parse_list_cell(value):
    """Convert CSV stringified Python lists back into plain Python lists."""
    if isinstance(value, list):
        return value

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            return ast.literal_eval(stripped)

    return value


def build_target_json(row):
    return json.dumps(
        {
            "summary": row["summary"],
            "key_points": row["key_points"],
            "sentiment": row["sentiment"],
            "entities": row["entities"],
            "why_it_matters": row["why_it_matters"],
        },
        ensure_ascii=False,
    )


def build_teacher_training_dataframe(results_df):
    """
    Keep only schema-valid teacher outputs and prepare prompt/target columns.

    The notebook used the 3B model as a teacher to label train/validation splits.
    This helper converts those CSV exports into a compact dataframe that can be
    fed into QLoRA fine-tuning.
    """
    clean_df = results_df[results_df["success"] == True].copy()
    clean_df["key_points"] = clean_df["key_points"].apply(parse_list_cell)
    clean_df["entities"] = clean_df["entities"].apply(parse_list_cell)
    clean_df["target_json"] = clean_df.apply(build_target_json, axis=1)
    return clean_df[["article_text", "target_json"]]


def build_teacher_training_dataset(results_df):
    return Dataset.from_pandas(build_teacher_training_dataframe(results_df))
