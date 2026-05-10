import os
import json
import time
import argparse
import statistics
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import cohere


load_dotenv()

RESULTS_DIR = "benchmark_results"

COHERE_API_KEY = os.environ["COHERE_API_KEY"]
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

DEFAULT_INPUT = (
    "benchmark_results/"
    "benchmark_results_section_aware_v2_hybrid_semantic_top_50_lexical_top_50_ef_search_120.json"
)

DEFAULT_OUTPUT = (
    "benchmark_results/"
    "reranked_results_section_aware_v2_hybrid_semantic_top_50_lexical_top_50_to_top_10.json"
)

co = cohere.ClientV2(api_key=COHERE_API_KEY)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * (p / 100)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)

    if lower == upper:
        return sorted_values[lower]

    weight = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    return {
        "count": len(values),
        "avg_ms": statistics.mean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def get_chunk_id(row: Dict[str, Any]) -> str:
    if row.get("chunk_id") is not None:
        return str(row["chunk_id"])
    if row.get("id") is not None:
        return str(row["id"])
    return ""


def build_rerank_document(row: Dict[str, Any]) -> str:
    """
    Build reranker input for one hybrid candidate.

    For hybrid retrieval, include:
    - title
    - section_path
    - retrieval sources
    - content

    Do not include too much scoring metadata in the reranker text.
    Scores/ranks are useful for debugging, but the reranker should judge
    query-document relevance from semantic content.
    """
    title = row.get("title") or ""
    section_path = row.get("section_path") or ""

    metadata = row.get("metadata") or {}
    if not title:
        title = metadata.get("article_title") or metadata.get("title") or ""

    if not section_path:
        section_path = metadata.get("section_path") or metadata.get("section_title") or ""

    content = row.get("content") or ""

    parts = []

    if title:
        parts.append(f"Title: {title}")

    if section_path:
        parts.append(f"Section: {section_path}")

    parts.append(f"Content:\n{content}")

    return "\n\n".join(parts).strip()


def rerank_query_results(
        query: str,
        rows: List[Dict[str, Any]],
        rerank_top_k: int,
) -> Dict[str, Any]:
    if not rows:
        return {
            "reranked_results": [],
            "rerank_latency_ms": 0.0,
        }

    documents = [build_rerank_document(row) for row in rows]

    start = time.perf_counter()

    response = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(rerank_top_k, len(documents)),
    )

    rerank_latency_ms = (time.perf_counter() - start) * 1000

    reranked_results = []

    for result in response.results:
        original_idx = result.index
        original_row = rows[original_idx]

        reranked_results.append(
            {
                **original_row,
                "chunk_id": get_chunk_id(original_row),
                "original_rank": original_idx + 1,
                "rerank_score": result.relevance_score,
            }
        )

    return {
        "reranked_results": reranked_results,
        "rerank_latency_ms": rerank_latency_ms,
    }


def candidate_source_counts(candidates: List[Dict[str, Any]]) -> Dict[str, int]:
    semantic_only = 0
    lexical_only = 0
    both = 0
    unknown = 0

    for row in candidates:
        sources = row.get("retrieval_sources") or []

        if sources == ["semantic"] or set(sources) == {"semantic"}:
            semantic_only += 1
        elif sources == ["lexical"] or set(sources) == {"lexical"}:
            lexical_only += 1
        elif set(sources) == {"semantic", "lexical"}:
            both += 1
        else:
            unknown += 1

    return {
        "semantic_only": semantic_only,
        "lexical_only": lexical_only,
        "both_semantic_and_lexical": both,
        "unknown": unknown,
    }


def reranked_source_counts(reranked_results: List[Dict[str, Any]]) -> Dict[str, int]:
    return candidate_source_counts(reranked_results)


def rerank_hybrid_benchmark_file(
        input_path: str,
        output_path: str,
        candidate_limit: int,
        rerank_top_k: int,
) -> Dict[str, Any]:
    benchmark = load_json(input_path)

    rerank_latencies = []
    reranked_queries = []

    for idx, item in enumerate(benchmark.get("queries", []), start=1):
        query = item["query"]

        candidates = item.get("results") or []

        if not candidates:
            # Defensive fallback, but hybrid benchmark should write candidates to "results".
            candidates = (
                    item.get("hybrid_results")
                    or item.get("expanded_results")
                    or item.get("original_top_results")
                    or []
            )

        candidates = candidates[:candidate_limit]

        print(
            f"[{idx}/{len(benchmark.get('queries', []))}] "
            f"Reranking hybrid candidates: {query}"
        )

        rerank_output = rerank_query_results(
            query=query,
            rows=candidates,
            rerank_top_k=rerank_top_k,
        )

        rerank_latencies.append(rerank_output["rerank_latency_ms"])

        reranked_results = rerank_output["reranked_results"]

        candidate_counts = candidate_source_counts(candidates)
        top_counts = reranked_source_counts(reranked_results)

        reranked_queries.append(
            {
                "query": query,

                "candidate_count": len(candidates),
                "rerank_top_k": rerank_top_k,
                "rerank_latency_ms": rerank_output["rerank_latency_ms"],

                "semantic_candidate_count": item.get("semantic_candidate_count"),
                "lexical_candidate_count": item.get("lexical_candidate_count"),
                "hybrid_candidate_count": item.get("hybrid_candidate_count"),
                "semantic_lexical_overlap_count": item.get("semantic_lexical_overlap_count"),

                "candidate_source_counts": candidate_counts,
                "reranked_source_counts": top_counts,

                "original_top_results": candidates[:rerank_top_k],
                "reranked_results": reranked_results,
            }
        )

        if reranked_results:
            top = reranked_results[0]
            print(
                f"  top_after_rerank={top.get('title')} "
                f"score={top.get('rerank_score'):.4f} "
                f"original_rank={top.get('original_rank')} "
                f"sources={top.get('retrieval_sources')}"
            )

    output = {
        "metadata": {
            "input_path": input_path,
            "candidate_limit": candidate_limit,
            "rerank_top_k": rerank_top_k,
            "rerank_model": COHERE_RERANK_MODEL,
            "rerank_input_field": "queries[].results",
            "retrieval_mode": "hybrid_semantic_plus_lexical",
        },
        "summary": {
            "rerank_latency": summarize(rerank_latencies),
            "avg_candidate_count": (
                statistics.mean([q["candidate_count"] for q in reranked_queries])
                if reranked_queries
                else None
            ),
            "avg_semantic_only_candidates": (
                statistics.mean(
                    [
                        q["candidate_source_counts"]["semantic_only"]
                        for q in reranked_queries
                    ]
                )
                if reranked_queries
                else None
            ),
            "avg_lexical_only_candidates": (
                statistics.mean(
                    [
                        q["candidate_source_counts"]["lexical_only"]
                        for q in reranked_queries
                    ]
                )
                if reranked_queries
                else None
            ),
            "avg_both_source_candidates": (
                statistics.mean(
                    [
                        q["candidate_source_counts"]["both_semantic_and_lexical"]
                        for q in reranked_queries
                    ]
                )
                if reranked_queries
                else None
            ),
            "avg_semantic_only_in_reranked_topk": (
                statistics.mean(
                    [
                        q["reranked_source_counts"]["semantic_only"]
                        for q in reranked_queries
                    ]
                )
                if reranked_queries
                else None
            ),
            "avg_lexical_only_in_reranked_topk": (
                statistics.mean(
                    [
                        q["reranked_source_counts"]["lexical_only"]
                        for q in reranked_queries
                    ]
                )
                if reranked_queries
                else None
            ),
            "avg_both_source_in_reranked_topk": (
                statistics.mean(
                    [
                        q["reranked_source_counts"]["both_semantic_and_lexical"]
                        for q in reranked_queries
                    ]
                )
                if reranked_queries
                else None
            ),
        },
        "queries": reranked_queries,
    }

    save_json(output_path, output)

    print("\nSaved hybrid reranked results:")
    print(output_path)
    print("\nSummary:")
    print(json.dumps(output["summary"], indent=2))

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Rerank hybrid semantic + lexical retrieval results using Cohere Rerank."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Hybrid benchmark JSON path.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSON path for reranked hybrid results.",
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=100,
        help="Max hybrid candidates to send to reranker after semantic+lexical dedupe.",
    )

    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=10,
        help="How many results to keep after reranking.",
    )

    args = parser.parse_args()

    rerank_hybrid_benchmark_file(
        input_path=args.input,
        output_path=args.output,
        candidate_limit=args.candidate_limit,
        rerank_top_k=args.rerank_top_k,
    )


if __name__ == "__main__":
    main()