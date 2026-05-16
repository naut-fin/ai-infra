import os
import re
import json
import time
import argparse
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from routing_eval_queries import ROUTING_EVAL_QUERIES

# Import from your existing files
from benchmark_hybrid_retrieval import benchmark_hybrid_query
from benchmark_retrieval import benchmark_query


RESULTS_DIR = "benchmark_results"

DEFAULT_EF_SEARCH = 120
DEFAULT_SEMANTIC_TOP_K = 50
DEFAULT_LEXICAL_TOP_K = 50
DEFAULT_REPEAT_COUNT = 1
DEFAULT_WARMUP_COUNT = 0
DEFAULT_CHUNKING_STRATEGY = "section_aware_v2"


ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
NUMBER_PATTERN = re.compile(r"\b\d+(\.\d+)?\b")
DATE_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
QUOTED_PATTERN = re.compile(r"[\"“”']")
CODE_LIKE_PATTERN = re.compile(
    r"\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z0-9_.]+\b|"
    r"\b[a-zA-Z_][a-zA-Z0-9_]*Exception\b|"
    r"\b[A-Z]+_\w+\b|"
    r"\b\w+@\d+\b|"
    r"\b[A-Za-z]+-\d+\b"
)

RARE_EXACT_TERMS = {
    "hnsw",
    "bm25",
    "pgvector",
    "ivfflat",
    "ndcg",
    "recall@5",
    "recall@10",
    "ndcg@10",
    "ef_search",
    "oauth",
    "cuda",
    "kubernetes",
    "sqs",
    "vacuum",
    "qwen2.5",
    "text-embedding-3-small",
    "gpt-4",
    "10-k",
    "iso 8601",
    "http 429",
    "401",
    "acid",
    "raid",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def route_output_path(
        ef_search: int,
        semantic_top_k: int,
        lexical_top_k: int,
        chunking_strategy: str,
) -> str:
    safe_strategy = chunking_strategy.replace("/", "_").replace(" ", "_")
    return os.path.join(
        RESULTS_DIR,
        (
            f"routed_retrieval_results_{safe_strategy}_"
            f"semantic_top_{semantic_top_k}_lexical_top_{lexical_top_k}_"
            f"ef_search_{ef_search}.json"
        ),
    )


def has_rare_exact_term(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in RARE_EXACT_TERMS)


def route_query(query: str) -> Dict[str, Any]:
    """
    Cheap production-friendly routing rule.

    Hybrid is selected when the query has strong lexical/exact-match signals.
    Otherwise semantic-only is used.
    """

    matched_signals = []

    if ACRONYM_PATTERN.search(query):
        matched_signals.append("acronym")

    if NUMBER_PATTERN.search(query):
        matched_signals.append("number")

    if DATE_PATTERN.search(query):
        matched_signals.append("date")

    if QUOTED_PATTERN.search(query):
        matched_signals.append("quoted_phrase")

    if CODE_LIKE_PATTERN.search(query):
        matched_signals.append("code_like_token")

    if has_rare_exact_term(query):
        matched_signals.append("rare_exact_term")

    if matched_signals:
        return {
            "route": "hybrid",
            "routing_reason": "lexical_signal_detected",
            "matched_signals": matched_signals,
        }

    return {
        "route": "semantic",
        "routing_reason": "no_strong_lexical_signal",
        "matched_signals": [],
    }


def normalize_routed_result(
        *,
        query: str,
        expected_route: Optional[str],
        routing_decision: Dict[str, Any],
        retrieval_output: Dict[str, Any],
        latency_ms: float,
) -> Dict[str, Any]:
    route = routing_decision["route"]
    semantic_lexical_overlap_count = retrieval_output.get(
        "semantic_lexical_overlap_count"
    )
    lexical_candidate_count = retrieval_output.get("lexical_candidate_count")
    lexical_only_candidate_count = (
        lexical_candidate_count - semantic_lexical_overlap_count
        if lexical_candidate_count is not None
        and semantic_lexical_overlap_count is not None
        else None
    )

    return {
        "query": query,
        "expected_route": expected_route,
        "selected_route": route,
        "route_correct": expected_route == route if expected_route else None,
        "routing_reason": routing_decision["routing_reason"],
        "matched_signals": routing_decision["matched_signals"],

        "routing_plus_retrieval_latency_ms": latency_ms,
        "embedding_latency_ms": retrieval_output.get("embedding_latency_ms"),
        "retrieval_latency_summary": retrieval_output.get("retrieval_latency_summary"),

        "candidate_count": len(retrieval_output.get("results", [])),
        "semantic_candidate_count": retrieval_output.get("semantic_candidate_count"),
        "lexical_candidate_count": lexical_candidate_count,
        "hybrid_candidate_count": retrieval_output.get("hybrid_candidate_count"),
        "semantic_lexical_overlap_count": semantic_lexical_overlap_count,
        "lexical_only_candidate_count": lexical_only_candidate_count,
        "result_chunk_ids": retrieval_output.get("result_chunk_ids", []),

        # Common field for reranker input.
        "results": retrieval_output.get("results", []),

        # Keep route-specific debug fields.
        "semantic_results": retrieval_output.get("semantic_results"),
        "lexical_results": retrieval_output.get("lexical_results"),

        "error": retrieval_output.get("error"),
    }


def run_routed_retrieval(
        semantic_top_k: int,
        lexical_top_k: int,
        ef_search: int,
        repeat_count: int,
        warmup_count: int,
        chunking_strategy: str,
) -> Dict[str, Any]:

    query_results = []

    print("\nRunning routed retrieval benchmark")
    print(
        f"queries={len(ROUTING_EVAL_QUERIES)} "
        f"semantic_top_k={semantic_top_k} "
        f"lexical_top_k={lexical_top_k} "
        f"ef_search={ef_search}"
    )

    semantic_candidate_counts = []
    lexical_candidate_counts = []
    semantic_lexical_overlap_counts = []
    lexical_only_candidate_counts = []
    for idx, item in enumerate(ROUTING_EVAL_QUERIES, start=1):
        query = item["query"]
        expected_route = item.get("expected_route")

        routing_decision = route_query(query)
        selected_route = routing_decision["route"]

        print(
            f"[{idx}/{len(ROUTING_EVAL_QUERIES)}] "
            f"route={selected_route} query={query}"
        )

        start = time.perf_counter()

        try:
            if selected_route == "hybrid":
                retrieval_output = benchmark_hybrid_query(
                    query=query,
                    semantic_top_k=semantic_top_k,
                    lexical_top_k=lexical_top_k,
                    ef_search=ef_search,
                    repeat_count=repeat_count,
                    warmup_count=warmup_count,
                    chunking_strategy=chunking_strategy,
                )
                semantic_candidate_count = retrieval_output.get(
                    "semantic_candidate_count"
                )
                lexical_candidate_count = retrieval_output.get(
                    "lexical_candidate_count"
                )
                semantic_lexical_overlap_count = retrieval_output.get(
                    "semantic_lexical_overlap_count"
                )

                if semantic_candidate_count is not None:
                    semantic_candidate_counts.append(semantic_candidate_count)
                if lexical_candidate_count is not None:
                    lexical_candidate_counts.append(lexical_candidate_count)
                if semantic_lexical_overlap_count is not None:
                    semantic_lexical_overlap_counts.append(
                        semantic_lexical_overlap_count
                    )
                if (
                    lexical_candidate_count is not None
                    and semantic_lexical_overlap_count is not None
                ):
                    lexical_only_candidate_counts.append(
                        lexical_candidate_count - semantic_lexical_overlap_count
                    )

            else:
                retrieval_output = benchmark_query(
                    query=query,
                    mode="hnsw",
                    top_k=semantic_top_k,
                    repeat_count=repeat_count,
                    warmup_count=warmup_count,
                    ef_search=ef_search,
                )

            latency_ms = (time.perf_counter() - start) * 1000

            routed_result = normalize_routed_result(
                query=query,
                expected_route=expected_route,
                routing_decision=routing_decision,
                retrieval_output=retrieval_output,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            routed_result = {
                "query": query,
                "expected_route": expected_route,
                "selected_route": selected_route,
                "route_correct": expected_route == selected_route if expected_route else None,
                "routing_reason": routing_decision["routing_reason"],
                "matched_signals": routing_decision["matched_signals"],
                "routing_plus_retrieval_latency_ms": latency_ms,
                "candidate_count": 0,
                "semantic_candidate_count": None,
                "lexical_candidate_count": None,
                "hybrid_candidate_count": None,
                "semantic_lexical_overlap_count": None,
                "lexical_only_candidate_count": None,
                "results": [],
                "result_chunk_ids": [],
                "error": repr(e),
            }

            print(f"  ERROR: {repr(e)}")

        query_results.append(routed_result)

        print(
            f"  route_correct={routed_result.get('route_correct')} "
            f"candidates={routed_result.get('candidate_count')} "
            f"latency={routed_result.get('routing_plus_retrieval_latency_ms'):.2f}ms"
        )

    route_counts = {}
    route_correct_values = []
    latency_values = []
    candidate_counts = []

    for item in query_results:
        selected_route = item.get("selected_route")
        route_counts[selected_route] = route_counts.get(selected_route, 0) + 1

        if item.get("route_correct") is not None:
            route_correct_values.append(1 if item["route_correct"] else 0)

        if item.get("routing_plus_retrieval_latency_ms") is not None:
            latency_values.append(item["routing_plus_retrieval_latency_ms"])

        candidate_counts.append(item.get("candidate_count", 0))

    output = {
        "metadata": {
            "mode": "routed_retrieval",
            "created_at": now_iso(),
            "query_count": len(ROUTING_EVAL_QUERIES),
            "semantic_top_k": semantic_top_k,
            "lexical_top_k": lexical_top_k,
            "ef_search": ef_search,
            "repeat_count": repeat_count,
            "warmup_count": warmup_count,
            "chunking_strategy": chunking_strategy,
            "routing_strategy": "cheap_lexical_signal_rule",
            "next_stage": "reranker_can_consume_queries[*].results",
        },
        "summary": {
            "route_counts": route_counts,
            "routing_accuracy_against_expected_labels": (
                statistics.mean(route_correct_values) if route_correct_values else None
            ),
            "avg_latency_ms": statistics.mean(latency_values) if latency_values else None,
            "p50_latency_ms": (
                sorted(latency_values)[len(latency_values) // 2]
                if latency_values
                else None
            ),
            "avg_candidate_count": (
                statistics.mean(candidate_counts) if candidate_counts else None
            ),
            "avg_semantic_candidate_count": (
                statistics.mean(semantic_candidate_counts)
                if semantic_candidate_counts
                else None
            ),
            "avg_lexical_candidate_count": (
                statistics.mean(lexical_candidate_counts)
                if lexical_candidate_counts
                else None
            ),
            "avg_semantic_lexical_overlap_count": (
                statistics.mean(semantic_lexical_overlap_counts)
                if semantic_lexical_overlap_counts
                else None
            ),
            "avg_lexical_only_candidate_count": (
                statistics.mean(lexical_only_candidate_counts)
                if lexical_only_candidate_counts
                else None
            ),
        },
        "queries": query_results,
    }

    ensure_results_dir()

    output_path = route_output_path(
        ef_search=ef_search,
        semantic_top_k=semantic_top_k,
        lexical_top_k=lexical_top_k,
        chunking_strategy=chunking_strategy,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved routed retrieval output to {output_path}")
    print(json.dumps(output["summary"], indent=2))

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Route queries to semantic-only or hybrid retrieval and save common candidate output."
    )

    parser.add_argument("--semantic-top-k", type=int, default=DEFAULT_SEMANTIC_TOP_K)
    parser.add_argument("--lexical-top-k", type=int, default=DEFAULT_LEXICAL_TOP_K)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument("--warmup-count", type=int, default=DEFAULT_WARMUP_COUNT)
    parser.add_argument("--chunking-strategy", default=DEFAULT_CHUNKING_STRATEGY)

    args = parser.parse_args()

    run_routed_retrieval(
        semantic_top_k=args.semantic_top_k,
        lexical_top_k=args.lexical_top_k,
        ef_search=args.ef_search,
        repeat_count=args.repeat_count,
        warmup_count=args.warmup_count,
        chunking_strategy=args.chunking_strategy,
    )


if __name__ == "__main__":
    main()
