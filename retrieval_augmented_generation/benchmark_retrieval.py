import os
import json
import time
import argparse
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
from questions import EVAL_QUERIES

import math


load_dotenv()

# -----------------------------
# Configuration
# -----------------------------

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

DEFAULT_TOP_K = 10
DEFAULT_REPEAT_COUNT = 5
DEFAULT_WARMUP_COUNT = 1

RESULTS_DIR = "benchmark_results"

# add more ef_search values in here to differentiate between different ef_search values
ef_search_list = [120]

# rpc added inside sql directory, you can use that to create your own

SEARCH_RPC_BY_MODE = {
    "flat": "match_chunks_hnsw",
    "hnsw": "match_chunks_hnsw",
}


# Fixed query set.
# Keep this stable across runs so Flat/HNSW comparisons are meaningful.



openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# -----------------------------
# Utility functions
# -----------------------------

def now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def ensure_results_dir() -> None:
    """Create benchmark output directory if missing."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

def percentile_nearest_rank(scores, percentile):
    """
    percentile: 0.10 for p10, 0.50 for p50, etc.
    """
    if not scores:
        return None

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    rank = math.ceil(percentile * n)
    rank = max(1, min(rank, n))

    return sorted_scores[rank - 1]


def percentile(values: List[float], p: float) -> Optional[float]:
    """
    Compute percentile using nearest-rank style interpolation.

    values: list of latency values in ms
    p: percentile, e.g. 50, 95, 99
    """
    if not values:
        return None

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * (p / 100)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)

    if lower == upper:
        return sorted_values[lower]

    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def embed_query(query: str) -> Tuple[List[float], float]:
    """
    Generate embedding for a query and return:
    - embedding vector
    - embedding latency in milliseconds

    We measure this separately because embedding latency and DB retrieval latency
    are different system bottlenecks.
    """
    start = time.perf_counter()

    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=query,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    return response.data[0].embedding, latency_ms


def benchmark_results_path(mode: str, ef_search: Optional[int] = None) -> str:
    """Return the benchmark result path for a mode/configuration."""
    if mode == "flat":
        return os.path.join(RESULTS_DIR, "benchmark_results_context_aware_flat.json")

    if ef_search is None:
        raise ValueError(f"ef_search is required for {mode} benchmark results")

    return os.path.join(RESULTS_DIR, f"benchmark_results_context_aware_ef_search_{ef_search}_{mode}.json")


def comparison_results_path(flat_mode: str, ann_mode: str, ef_search: int) -> str:
    """Return the comparison result path for an ANN ef_search configuration."""
    return os.path.join(RESULTS_DIR, f"compare_context_aware_{flat_mode}_vs_{ann_mode}.json")


def search_rpc(
        mode: str,
        query_embedding: List[float],
        top_k: int,
        ef_search: Optional[int],
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Call Supabase RPC that performs vector search.

    This measures DB/RPC retrieval latency only.
    It does NOT include embedding generation time.
    """
    if mode not in SEARCH_RPC_BY_MODE:
        raise ValueError(f"Unsupported benchmark mode for RPC search: {mode}")

    rpc_args = {
        "query_embedding": query_embedding,
        "match_count": top_k,
    }

    if mode == "hnsw":
        if ef_search is None:
            raise ValueError("ef_search is required for hnsw mode")
        rpc_args["ef_search"] = ef_search

    start = time.perf_counter()

    response = supabase.rpc(
        SEARCH_RPC_BY_MODE[mode],
        rpc_args,
    ).execute()

    latency_ms = (time.perf_counter() - start) * 1000
    return response.data or [], latency_ms


def result_ids(results: List[Dict[str, Any]], k: int) -> List[str]:
    """Extract top-k chunk IDs from result rows."""
    return [str(row["chunk_id"]) for row in results[:k]]


def recall_at_k(flat_ids: List[str], ann_ids: List[str], k: int) -> Optional[float]:
    """
    Compute recall@k using Flat exact search as ground truth.

    recall@k = overlap(flat_top_k, ann_top_k) / k
    """
    if len(flat_ids) < k:
        return None

    flat_top_k = set(flat_ids[:k])
    ann_top_k = set(ann_ids[:k])

    if not flat_top_k:
        return 0.0

    return len(flat_top_k.intersection(ann_top_k)) / len(flat_top_k)


def summarize_latencies(latencies: List[float]) -> Dict[str, Optional[float]]:
    """Return standard latency summary."""
    if not latencies:
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
        "count": len(latencies),
        "avg_ms": statistics.mean(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


# -----------------------------
# Benchmark execution
# -----------------------------

def benchmark_query(
        query: str,
        mode: str,
        top_k: int,
        repeat_count: int,
        warmup_count: int,
        ef_search: Optional[int],
) -> Dict[str, Any]:
    """
    Benchmark one query.

    Steps:
    1. Embed query once.
    2. Run warmup retrievals, not recorded.
    3. Run repeated measured retrievals.
    4. Store final result set and latency stats.

    Why embed once?
    - We want stable retrieval comparisons.
    - Re-embedding same query multiple times adds external API noise.
    - DB benchmark should isolate vector search latency.
    """
    query_embedding, embedding_latency_ms = embed_query(query)

    # Warmup calls reduce first-request noise.
    for _ in range(warmup_count):
        search_rpc(mode, query_embedding, top_k, ef_search)

    retrieval_latencies = []
    last_results: List[Dict[str, Any]] = []

    for _ in range(repeat_count):
        results, retrieval_latency_ms = search_rpc(mode, query_embedding, top_k, ef_search)
        retrieval_latencies.append(retrieval_latency_ms)
        last_results = results

    return {
        "query": query,
        "embedding_latency_ms": embedding_latency_ms,
        "retrieval_latency_summary": summarize_latencies(retrieval_latencies),
        "results": last_results,
        "result_chunk_ids": result_ids(last_results, top_k),
    }


def run_benchmark(
        mode: str,
        top_k: int,
        repeat_count: int,
        warmup_count: int,
        ef_search: Optional[int],
) -> Dict[str, Any]:
    """
    Run full benchmark for all fixed evaluation queries.

    mode can be:
    - flat
    - hnsw

    Important:
    This script calls a separate RPC per supported benchmark mode. Flat should
    use an exact ordered scan RPC, while HNSW can set hnsw.ef_search inside
    its RPC.
    """
    print(f"\nRunning benchmark mode={mode}")
    print(f"queries={len(EVAL_QUERIES)} top_k={top_k} repeats={repeat_count}")

    query_results = []

    for idx, query in enumerate(EVAL_QUERIES, start=1):
        print(f"[{idx}/{len(EVAL_QUERIES)}] {query}")

        try:
            result = benchmark_query(
                query=query,
                mode=mode,
                top_k=top_k,
                repeat_count=repeat_count,
                warmup_count=warmup_count,
                ef_search=ef_search,
            )
            query_results.append(result)

            p50 = result["retrieval_latency_summary"]["p50_ms"]
            p95 = result["retrieval_latency_summary"]["p95_ms"]

            print(f"  p50={p50:.2f}ms p95={p95:.2f}ms top_title={result['results'][0]['title'] if result['results'] else 'NONE'}")

        except Exception as e:
            print(f"  ERROR: {repr(e)}")
            query_results.append(
                {
                    "query": query,
                    "error": repr(e),
                    "results": [],
                    "result_chunk_ids": [],
                }
            )

    all_retrieval_latencies = []

    for item in query_results:
        summary = item.get("retrieval_latency_summary")
        if not summary:
            continue

        # We only have summarized latency per query here.
        # For global summary, use p50 per query as a simple aggregate signal.
        if summary.get("p50_ms") is not None:
            all_retrieval_latencies.append(summary["p50_ms"])

    benchmark_output = {
        "metadata": {
            "mode": mode,
            "created_at": now_iso(),
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "query_count": len(EVAL_QUERIES),
            "top_k": top_k,
            "repeat_count": repeat_count,
            "warmup_count": warmup_count,
            "ef_search": ef_search,
            "search_rpc": SEARCH_RPC_BY_MODE.get(mode),
        },
        "summary": {
            "retrieval_p50_across_queries_ms": percentile(all_retrieval_latencies, 50),
            "retrieval_p95_across_queries_ms": percentile(all_retrieval_latencies, 95),
            "retrieval_avg_across_queries_ms": statistics.mean(all_retrieval_latencies)
            if all_retrieval_latencies else None,
        },
        "queries": query_results,
    }

    ensure_results_dir()

    output_path = benchmark_results_path(mode, ef_search)

    with open(output_path, "w") as f:
        json.dump(benchmark_output, f, indent=2)

    print(f"\nSaved benchmark results to {output_path}")
    print("Summary:", json.dumps(benchmark_output["summary"], indent=2))

    return benchmark_output


# -----------------------------
# Comparison logic
# -----------------------------


def dcg_at_k(relevances, k):
    """
    relevances: list of relevance scores in retrieved order.
    Example: [10, 0, 8, 3]
    """
    relevances = relevances[:k]
    return sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(relevances, start=1)
    )


def ndcg_at_k(flat_ids, retrieved_ids,  k):
    """
    retrieved_ids: HNSW result chunk ids in ranked order
    flat_ids: Flat result chunk ids in ranked order
    k: cutoff, e.g. 10 or 50
    """

    # Assign higher relevance to higher-ranked Flat results.
    # Flat rank 1 gets k, Flat rank 2 gets k-1, ...
    flat_top_k = flat_ids[:k]

    relevance_by_id = {
        chunk_id: k - rank + 1
        for rank, chunk_id in enumerate(flat_top_k, start=1)
    }

    retrieved_relevances = [
        relevance_by_id.get(chunk_id, 0)
        for chunk_id in retrieved_ids[:k]
    ]

    ideal_relevances = sorted(
        relevance_by_id.values(),
        reverse=True
    )

    dcg = dcg_at_k(retrieved_relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg

def load_results(mode: str, ef_search: Optional[int] = None) -> Dict[str, Any]:
    """Load previously saved benchmark result JSON."""
    path = benchmark_results_path(mode, ef_search)

    with open(path, "r") as f:
        return json.load(f)


def compare_results(flat_mode: str, ann_mode: str, top_k_values: List[int], ef_search: int) -> Dict[str, Any]:
    """
    Compare ANN mode against Flat mode.

    Flat is treated as exact ground truth.
    ANN is currently HNSW.
    """
    flat = load_results(flat_mode, ef_search)
    ann = load_results(ann_mode, ef_search)

    flat_by_query = {item["query"]: item for item in flat["queries"]}
    ann_by_query = {item["query"]: item for item in ann["queries"]}

    per_query = []
    recall_sums = {k: [] for k in top_k_values}
    ndcg_k = {k: [] for k in top_k_values}

    for query, flat_item in flat_by_query.items():
        ann_item = ann_by_query.get(query)

        if not ann_item:
            continue

        flat_ids = flat_item.get("result_chunk_ids", [])
        ann_ids = ann_item.get("result_chunk_ids", [])

        query_comparison = {
            "query": query,
            "recall": {},
            "ndcg":{},
            "flat_top_title": flat_item["results"][0]["title"] if flat_item.get("results") else None,
            "ann_top_title": ann_item["results"][0]["title"] if ann_item.get("results") else None,
        }

        for k in top_k_values:
            r = recall_at_k(flat_ids, ann_ids, k)
            n = ndcg_at_k(flat_ids, ann_ids, k)

            # Queries with fewer than k ground-truth neighbors were excluded from recall@k computation to maintain metric validity.

            if r is None:
                continue
            query_comparison["recall"][f"recall@{k}"] = r
            query_comparison["ndcg"][f"ndcg@{k}"] = n
            ndcg_k[k].append(n)
            recall_sums[k].append(r)

        per_query.append(query_comparison)

    summary = {}

    for k, values in recall_sums.items():
        summary[f"avg_recall@{k}"] = statistics.mean(values) if values else None
        summary[f"min_recall@{k}"] = min(values) if values else None

    for k, values in ndcg_k.items():
        summary[f"avg_ndcg@{k}"] = statistics.mean(values) if values else None
        summary[f"min_ndcg@{k}"] = min(values) if values else None
        summary[f"median_ndcg@{k}"] = statistics.median(values) if values else None
        summary[f"p10_ndcg@{k}"] = percentile_nearest_rank(values, 0.10) if values else None


    comparison_output = {
        "metadata": {
            "created_at": now_iso(),
            "flat_mode": flat_mode,
            "ann_mode": ann_mode,
            "ef_search": ef_search,
            "top_k_values": top_k_values,
        },
        "summary": summary,
        "per_query": per_query,
    }

    ensure_results_dir()

    output_path = comparison_results_path(flat_mode, ann_mode, ef_search)

    with open(output_path, "w") as f:
        json.dump(comparison_output, f, indent=2)

    print(f"\nSaved comparison results to {output_path}")
    print("Summary:", json.dumps(summary, indent=2))

    return comparison_output


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Supabase pgvector retrieval performance."
    )

    parser.add_argument(
        "--mode",
        choices=["flat", "hnsw"],
        help="Benchmark mode. Flat and HNSW use separate Supabase RPC paths.",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare flat benchmark results against another mode.",
    )

    parser.add_argument(
        "--ann-mode",
        choices=["hnsw"],
        default="hnsw",
        help="ANN mode to compare against flat.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to retrieve per query.",
    )

    parser.add_argument(
        "--repeat-count",
        type=int,
        default=DEFAULT_REPEAT_COUNT,
        help="Number of measured retrieval calls per query.",
    )

    parser.add_argument(
        "--warmup-count",
        type=int,
        default=DEFAULT_WARMUP_COUNT,
        help="Number of unmeasured warmup calls per query.",
    )

    args = parser.parse_args()

    if args.compare:
        for ef_search in ef_search_list:
            print(f"Comparing benchmark data for ef_search:{ef_search} HNSW vs Flat")
            compare_results(
                flat_mode="flat",
                ann_mode=args.ann_mode,
                top_k_values=[5, 10, 50],
                ef_search=ef_search,
            )
        return


    if not args.mode:
        raise ValueError("Provide --mode flat|hnsw or use --compare")

    if args.mode == "flat":
        run_benchmark(
            mode=args.mode,
            top_k=args.top_k,
            repeat_count=args.repeat_count,
            warmup_count=args.warmup_count,
            ef_search=None,
        )
        return

    for ef_search in ef_search_list:
        print(f"fetching benchmark data for mode:{args.mode} for ef_search:{ef_search}")
        run_benchmark(
            mode=args.mode,
            top_k=args.top_k,
            repeat_count=args.repeat_count,
            warmup_count=args.warmup_count,
            ef_search=ef_search,
        )


if __name__ == "__main__":
    main()
