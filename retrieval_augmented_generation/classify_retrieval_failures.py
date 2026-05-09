import os
import json
import argparse
from collections import Counter
from typing import Any, Dict, List, Tuple


RESULTS_DIR = "benchmark_results"


# Heuristic thresholds.
# These are not absolute truths; they are practical debugging signals.
LOW_SIMILARITY_THRESHOLD = 0.30
GOOD_SIMILARITY_THRESHOLD = 0.40

EXPECTED_TERMS = {
    "hadoop": ["hadoop", "mapreduce", "hdfs"],
    "spark": ["apache spark", "spark", "cluster", "distributed"],
    "docker": ["docker", "container", "containerization"],
    "kubernetes": ["kubernetes", "container orchestration", "cluster"],
    "api": ["application programming interface", "api"],
    "cloud computing": ["cloud computing", "distributed computing", "virtual machine"],
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def index_by_query(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["query"]: item for item in results["queries"]}


def get_recall(item: Dict[str, Any], k: int) -> float:
    return item.get("recall", {}).get(f"recall@{k}", 0.0)


def top_similarity(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(rows[0].get("similarity", 0.0))


def top_title(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    return str(rows[0].get("title", ""))


def query_terms(query: str) -> List[str]:
    """
    Very simple keyword extraction.
    Good enough for failure debugging.
    """
    stopwords = {
        "what", "is", "the", "a", "an", "of", "in", "on", "to", "and", "or",
        "how", "why", "does", "do", "explain", "define", "uses", "use",
        "advantages", "benefits", "basics"
    }

    clean = (
        query.lower()
        .replace("?", "")
        .replace(",", "")
        .replace(".", "")
        .replace("/", " ")
    )

    return [t for t in clean.split() if t and t not in stopwords]


def title_matches_query(query: str, title: str) -> bool:
    terms = query_terms(query)
    title_lower = title.lower()
    return any(term in title_lower for term in terms)


def content_matches_query(query: str, rows: List[Dict[str, Any]], top_n: int = 3) -> bool:
    terms = query_terms(query)

    combined = " ".join(
        str(row.get("content", "")).lower()
        for row in rows[:top_n]
    )

    return any(term in combined for term in terms)


def expected_term_present(query, rows, top_n=3):
    q = query.lower()
    expected = []

    for key, terms in EXPECTED_TERMS.items():
        if key in q:
            expected.extend(terms)

    if not expected:
        return True

    combined = " ".join(
        (row.get("title", "") + " " + row.get("content", "")).lower()
        for row in rows[:top_n]
    )

    return any(term in combined for term in expected)

def classify_failure(
        query: str,
        recall_at_k: float,
        flat_rows: List[Dict[str, Any]],
        hnsw_rows: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Classify failure mode using practical heuristics.

    Labels:
    - corpus_gap
    - semantic_ambiguity
    - ann_miss
    - minor_ranking_difference
    - healthy
    """

    flat_sim = top_similarity(flat_rows)
    hnsw_sim = top_similarity(hnsw_rows)

    flat_title = top_title(flat_rows)
    hnsw_title = top_title(hnsw_rows)

    flat_title_match = title_matches_query(query, flat_title)
    hnsw_title_match = title_matches_query(query, hnsw_title)

    flat_content_match = content_matches_query(query, flat_rows)
    hnsw_content_match = content_matches_query(query, hnsw_rows)

    # If recall is perfect, not a failure.
    if recall_at_k >= 1.0:
        return "healthy", "HNSW matched Flat top-k exactly."

    # If Flat itself has weak similarity and weak query-term match,
    # then Flat ground truth is not very meaningful.
    flat_has_expected_terms = expected_term_present(query, flat_rows)

    if not flat_has_expected_terms:
        return (
            "corpus_gap",
            "Flat results do not contain expected domain terms; likely corpus does not cover the query."
        )

    # If query term appears to map to wrong sense in both systems,
    # e.g. Spark -> electrical spark, Cloud -> weather cloud.
    if (
            flat_sim >= LOW_SIMILARITY_THRESHOLD
            and hnsw_sim >= LOW_SIMILARITY_THRESHOLD
            and not flat_title_match
            and not hnsw_title_match
    ):
        return (
            "semantic_ambiguity",
            "Both systems retrieved semantically nearby but likely wrong-sense results."
        )

    # If Flat looks meaningfully better than HNSW,
    # this is likely a true ANN approximation miss.
    if (
            flat_sim >= GOOD_SIMILARITY_THRESHOLD
            or flat_title_match
            or flat_content_match
    ) and not hnsw_title_match and hnsw_sim < flat_sim:
        return (
            "ann_miss",
            "Flat found stronger/relevant candidates but HNSW missed or ranked weaker candidates."
        )

    # Partial overlap but different ordering is less severe.
    if recall_at_k >= 0.5:
        return (
            "minor_ranking_difference",
            "HNSW overlapped with Flat but changed ranking/order."
        )

    return (
        "uncertain",
        "Failure needs manual inspection."
    )


def classify_failures(
        flat_path: str,
        hnsw_path: str,
        comparison_path: str,
        recall_k: int,
) -> Dict[str, Any]:
    flat = load_json(flat_path)
    hnsw = load_json(hnsw_path)
    comparison = load_json(comparison_path)

    flat_by_query = index_by_query(flat)
    hnsw_by_query = index_by_query(hnsw)

    classified = []
    counts = Counter()

    for item in comparison["per_query"]:
        query = item["query"]
        recall_value = get_recall(item, recall_k)

        flat_item = flat_by_query.get(query, {})
        hnsw_item = hnsw_by_query.get(query, {})

        flat_rows = flat_item.get("results", [])
        hnsw_rows = hnsw_item.get("results", [])

        label, reason = classify_failure(
            query=query,
            recall_at_k=recall_value,
            flat_rows=flat_rows,
            hnsw_rows=hnsw_rows,
        )

        counts[label] += 1

        classified.append(
            {
                "query": query,
                f"recall@{recall_k}": recall_value,
                "classification": label,
                "reason": reason,
                "flat_top_title": top_title(flat_rows),
                "hnsw_top_title": top_title(hnsw_rows),
                "flat_top_similarity": top_similarity(flat_rows),
                "hnsw_top_similarity": top_similarity(hnsw_rows),
            }
        )

    total = len(classified)

    metrics = {
        label: {
            "count": count,
            "percentage": round((count / total) * 100, 2) if total else 0.0,
        }
        for label, count in counts.items()
    }

    output = {
        "metadata": {
            "recall_k": recall_k,
            "total_queries": total,
            "flat_path": flat_path,
            "hnsw_path": hnsw_path,
            "comparison_path": comparison_path,
        },
        "metrics": metrics,
        "classified_queries": classified,
    }

    output_path = os.path.join(
        RESULTS_DIR,
        f"classified_failures_recall_at_{recall_k}.json",
    )

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n==============================")
    print("FAILURE CLASSIFICATION METRICS")
    print("==============================")
    print(json.dumps(metrics, indent=2))

    print("\nSaved report:")
    print(output_path)

    print("\nWorst non-healthy examples:")
    for row in classified:
        if row["classification"] != "healthy":
            print(
                f"- {row['query']} | {row['classification']} | "
                f"recall@{recall_k}={row[f'recall@{recall_k}']} | "
                f"Flat={row['flat_top_title']} | HNSW={row['hnsw_top_title']}"
            )

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Classify Flat-vs-HNSW retrieval failures into practical failure categories."
    )

    parser.add_argument(
        "--flat",
        default=os.path.join(RESULTS_DIR, "benchmark_results_flat.json"),
    )

    parser.add_argument(
        "--hnsw",
        default=os.path.join(RESULTS_DIR, "benchmark_results_hnsw.json"),
    )

    parser.add_argument(
        "--comparison",
        default=os.path.join(RESULTS_DIR, "compare_flat_vs_hnsw.json"),
    )

    parser.add_argument(
        "--recall-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    classify_failures(
        flat_path=args.flat,
        hnsw_path=args.hnsw,
        comparison_path=args.comparison,
        recall_k=args.recall_k,
    )


if __name__ == "__main__":
    main()