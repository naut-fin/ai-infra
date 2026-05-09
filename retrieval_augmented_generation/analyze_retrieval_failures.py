import os
import json
import argparse
from typing import Any, Dict, List


RESULTS_DIR = "benchmark_results"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def index_by_query(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["query"]: item for item in results["queries"]}


def get_recall_value(item: Dict[str, Any], k: int) -> float:
    return item.get("recall", {}).get(f"recall@{k}", 0.0)


def preview(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def print_result_block(label: str, results: List[Dict[str, Any]], top_n: int) -> None:
    print(f"\n--- {label} TOP {top_n} ---")

    if not results:
        print("No results.")
        return

    for idx, row in enumerate(results[:top_n], start=1):
        print(f"\n#{idx}")
        print("chunk_id:", row.get("chunk_id"))
        print("title:", row.get("title"))
        print("similarity:", round(row.get("similarity", 0.0), 4))
        print("content:", preview(row.get("content", "")))


def analyze_failures(
        flat_results_path: str,
        hnsw_results_path: str,
        comparison_path: str,
        recall_k: int,
        max_failures: int,
        top_n_to_print: int,
) -> Dict[str, Any]:
    flat_results = load_json(flat_results_path)
    hnsw_results = load_json(hnsw_results_path)
    comparison = load_json(comparison_path)

    flat_by_query = index_by_query(flat_results)
    hnsw_by_query = index_by_query(hnsw_results)

    per_query = comparison["per_query"]

    # Sort by worst recall first.
    sorted_failures = sorted(
        per_query,
        key=lambda item: get_recall_value(item, recall_k),
    )

    worst = sorted_failures[:max_failures]

    failure_report = {
        "metadata": {
            "recall_k": recall_k,
            "max_failures": max_failures,
            "top_n_to_print": top_n_to_print,
            "flat_results_path": flat_results_path,
            "hnsw_results_path": hnsw_results_path,
            "comparison_path": comparison_path,
        },
        "failures": [],
    }

    print("\n==============================")
    print(f"WORST RETRIEVAL FAILURES BY recall@{recall_k}")
    print("==============================")

    for item in worst:
        query = item["query"]
        recall_value = get_recall_value(item, recall_k)

        flat_item = flat_by_query.get(query, {})
        hnsw_item = hnsw_by_query.get(query, {})

        flat_rows = flat_item.get("results", [])
        hnsw_rows = hnsw_item.get("results", [])

        flat_ids = [row.get("chunk_id") for row in flat_rows[:recall_k]]
        hnsw_ids = [row.get("chunk_id") for row in hnsw_rows[:recall_k]]
        overlap = sorted(set(flat_ids).intersection(set(hnsw_ids)))

        print("\n\n============================================================")
        print("QUERY:", query)
        print(f"recall@{recall_k}:", recall_value)
        print("overlap_count:", len(overlap))
        print("flat_top_title:", flat_rows[0].get("title") if flat_rows else None)
        print("hnsw_top_title:", hnsw_rows[0].get("title") if hnsw_rows else None)

        print_result_block("FLAT / GROUND TRUTH", flat_rows, top_n_to_print)
        print_result_block("HNSW / ANN", hnsw_rows, top_n_to_print)

        failure_report["failures"].append(
            {
                "query": query,
                f"recall@{recall_k}": recall_value,
                "overlap_count": len(overlap),
                "overlap_chunk_ids": overlap,
                "flat_top_title": flat_rows[0].get("title") if flat_rows else None,
                "hnsw_top_title": hnsw_rows[0].get("title") if hnsw_rows else None,
                "flat_top_results": flat_rows[:top_n_to_print],
                "hnsw_top_results": hnsw_rows[:top_n_to_print],
            }
        )

    output_path = os.path.join(
        RESULTS_DIR,
        f"failure_analysis_recall_at_{recall_k}.json",
    )

    with open(output_path, "w") as f:
        json.dump(failure_report, f, indent=2)

    print("\n==============================")
    print("SAVED FAILURE REPORT")
    print("==============================")
    print(output_path)

    return failure_report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze retrieval failures between Flat and HNSW benchmark results."
    )

    parser.add_argument(
        "--flat",
        default=os.path.join(RESULTS_DIR, "benchmark_results_flat.json"),
        help="Path to Flat benchmark results JSON.",
    )

    parser.add_argument(
        "--hnsw",
        default=os.path.join(RESULTS_DIR, "benchmark_results_hnsw.json"),
        help="Path to HNSW benchmark results JSON.",
    )

    parser.add_argument(
        "--comparison",
        default=os.path.join(RESULTS_DIR, "compare_flat_vs_hnsw.json"),
        help="Path to comparison JSON.",
    )

    parser.add_argument(
        "--recall-k",
        type=int,
        default=10,
        help="Analyze failures based on recall@k.",
    )

    parser.add_argument(
        "--max-failures",
        type=int,
        default=10,
        help="Number of worst queries to print.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of Flat/HNSW results to print per failed query.",
    )

    args = parser.parse_args()

    analyze_failures(
        flat_results_path=args.flat,
        hnsw_results_path=args.hnsw,
        comparison_path=args.comparison,
        recall_k=args.recall_k,
        max_failures=args.max_failures,
        top_n_to_print=args.top_n,
    )


if __name__ == "__main__":
    main()