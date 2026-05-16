import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_CONTEXT_PATH = (
    "benchmark_results/"
    "context_quality_judge_context_aware_chunk_v1_hnsw_reranked_top10.json"
)
DEFAULT_ANSWER_PATH = (
    "benchmark_results/"
    "rag_answer_judge_context_aware_ef_search_120_hnsw.json"
)
DEFAULT_OUTPUT_PATH = (
    "benchmark_results/"
    "context_vs_answer_judge_comparison_context_aware_hnsw.json"
)

ANSWERABILITY_VALUES = ("yes", "partial", "no", "error", "missing")


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    script_relative = Path(__file__).resolve().parent / path
    if script_relative.exists():
        return script_relative

    return script_relative if not candidate.is_absolute() else candidate


def load_json(path: str) -> Dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_answerability(value: Any) -> str:
    if not isinstance(value, str):
        return "missing"

    normalized = value.strip().lower()
    if normalized in ANSWERABILITY_VALUES:
        return normalized
    return "error"


def result_by_query(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_query: Dict[str, Dict[str, Any]] = {}

    for item in results:
        query = item.get("query")
        if not query:
            continue
        by_query[query] = item

    return by_query


def get_answerability(item: Optional[Dict[str, Any]]) -> str:
    if item is None:
        return "missing"
    judgment = item.get("judgment") or {}
    return normalize_answerability(judgment.get("answerability"))


def get_score(item: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if item is None:
        return None
    value = (item.get("judgment") or {}).get(key)
    return value if isinstance(value, (int, float)) else None


def get_reason(item: Optional[Dict[str, Any]]) -> str:
    if item is None:
        return ""
    return str((item.get("judgment") or {}).get("reason") or "")


def get_failure_type(item: Optional[Dict[str, Any]]) -> str:
    if item is None:
        return ""
    return str((item.get("judgment") or {}).get("failure_type") or "")


def pct(count: int, total: int) -> float:
    return count / total if total else 0.0


def init_transition_matrix() -> Dict[str, Dict[str, int]]:
    return {
        context_value: {
            answer_value: 0
            for answer_value in ANSWERABILITY_VALUES
        }
        for context_value in ANSWERABILITY_VALUES
    }


def compact_case(
    query: str,
    context_item: Optional[Dict[str, Any]],
    answer_item: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "query": query,
        "context_answerability": get_answerability(context_item),
        "answer_answerability": get_answerability(answer_item),
        "context_failure_type": get_failure_type(context_item),
        "answer_failure_type": get_failure_type(answer_item),
        "context_relevance_score": get_score(context_item, "context_relevance_score"),
        "context_completeness_score": get_score(context_item, "context_completeness_score"),
        "answer_answerability_score": get_score(answer_item, "answerability_score"),
        "answer_groundedness_score": get_score(answer_item, "groundedness_score"),
        "answer_completeness_score": get_score(answer_item, "completeness_score"),
        "source_titles": (answer_item or {}).get("source_titles", []),
        "source_chunk_ids": (answer_item or {}).get("source_chunk_ids", []),
        "answer": (answer_item or {}).get("answer", ""),
        "context_reason": get_reason(context_item),
        "answer_reason": get_reason(answer_item),
    }


def compare_judges(
    context_data: Dict[str, Any],
    answer_data: Dict[str, Any],
) -> Dict[str, Any]:
    context_by_query = result_by_query(context_data.get("results", []))
    answer_by_query = result_by_query(answer_data.get("results", []))
    all_queries = sorted(set(context_by_query) | set(answer_by_query))

    transition_matrix = init_transition_matrix()
    transitions: Dict[str, int] = {}
    matched_count = 0
    missing_context_count = 0
    missing_answer_count = 0

    requested_cases = {
        "A_context_partial_to_answer_yes": [],
        "B_context_no_to_answer_yes": [],
        "C_context_yes_to_answer_no": [],
    }
    questionable_context_not_yes_answer_yes = []
    all_compared_rows = []

    for query in all_queries:
        context_item = context_by_query.get(query)
        answer_item = answer_by_query.get(query)
        context_answerability = get_answerability(context_item)
        answer_answerability = get_answerability(answer_item)

        if context_item is None:
            missing_context_count += 1
        if answer_item is None:
            missing_answer_count += 1
        if context_item is not None and answer_item is not None:
            matched_count += 1

        transition_matrix[context_answerability][answer_answerability] += 1
        transition_key = f"{context_answerability}_to_{answer_answerability}"
        transitions[transition_key] = transitions.get(transition_key, 0) + 1

        row = compact_case(query, context_item, answer_item)
        all_compared_rows.append(row)

        if context_answerability == "partial" and answer_answerability == "yes":
            requested_cases["A_context_partial_to_answer_yes"].append(row)
        if context_answerability == "no" and answer_answerability == "yes":
            requested_cases["B_context_no_to_answer_yes"].append(row)
        if context_answerability == "yes" and answer_answerability == "no":
            requested_cases["C_context_yes_to_answer_no"].append(row)
        if context_answerability != "yes" and answer_answerability == "yes":
            questionable_context_not_yes_answer_yes.append(row)

    total = len(all_queries)
    requested_case_counts = {
        key: len(value)
        for key, value in requested_cases.items()
    }
    questionable_count = len(questionable_context_not_yes_answer_yes)

    return {
        "metadata": {
            "context_metadata": context_data.get("metadata", {}),
            "answer_metadata": answer_data.get("metadata", {}),
        },
        "summary": {
            "total_queries_union": total,
            "matched_queries": matched_count,
            "missing_context_queries": missing_context_count,
            "missing_answer_queries": missing_answer_count,
            "requested_case_counts": requested_case_counts,
            "requested_case_pct_of_matched": {
                key: pct(value, matched_count)
                for key, value in requested_case_counts.items()
            },
            "questionable_context_not_yes_answer_yes_count": questionable_count,
            "questionable_context_not_yes_answer_yes_pct_of_matched": pct(
                questionable_count,
                matched_count,
            ),
        },
        "transition_matrix": transition_matrix,
        "transition_counts": dict(sorted(transitions.items())),
        "requested_cases": requested_cases,
        "questionable_context_not_yes_answer_yes": questionable_context_not_yes_answer_yes,
        "rows": all_compared_rows,
    }


def print_summary(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    requested = summary["requested_case_counts"]

    print("Context judge vs answer judge comparison")
    print(f"Matched queries: {summary['matched_queries']}")
    print(f"Total query union: {summary['total_queries_union']}")
    print()
    print("Requested metrics:")
    print(f"A. context partial -> answer yes: {requested['A_context_partial_to_answer_yes']}")
    print(f"B. context no -> answer yes: {requested['B_context_no_to_answer_yes']}")
    print(f"C. context yes -> answer no: {requested['C_context_yes_to_answer_no']}")
    print()
    print(
        "Questionable context_answerability != yes and "
        "answer_answerability == yes: "
        f"{summary['questionable_context_not_yes_answer_yes_count']}"
    )
    print()
    print("Transition matrix:")
    print(json.dumps(report["transition_matrix"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-query context judge answerability against RAG answer "
            "judge answerability and surface suspicious answerable-from-weak-context cases."
        )
    )
    parser.add_argument("--context", default=DEFAULT_CONTEXT_PATH)
    parser.add_argument("--answer", default=DEFAULT_ANSWER_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print metrics without writing the detailed comparison JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_data = load_json(args.context)
    answer_data = load_json(args.answer)

    report = compare_judges(
        context_data=context_data,
        answer_data=answer_data,
    )

    print_summary(report)

    if not args.no_save:
        save_json(args.output, report)
        print()
        print(f"Saved detailed comparison to: {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
