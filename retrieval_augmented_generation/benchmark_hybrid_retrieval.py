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


load_dotenv()

# -----------------------------
# Configuration
# -----------------------------

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

RESULTS_DIR = "benchmark_results"

DEFAULT_SEMANTIC_TOP_K = 50
DEFAULT_LEXICAL_TOP_K = 50
DEFAULT_REPEAT_COUNT = 1
DEFAULT_WARMUP_COUNT = 0
DEFAULT_EF_SEARCH = 120
DEFAULT_CHUNKING_STRATEGY = "section_aware_v2"

SEMANTIC_RPC = "match_chunks_hnsw"
LEXICAL_RPC = "match_chunks_lexical"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# -----------------------------
# Utility functions
# -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


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
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_latencies(latencies: List[float]) -> Dict[str, Optional[float]]:
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


def embed_query(query: str) -> Tuple[List[float], float]:
    start = time.perf_counter()

    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=query,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    return response.data[0].embedding, latency_ms


def normalize_chunk_id(row: Dict[str, Any]) -> str:
    if row.get("chunk_id") is not None:
        return str(row["chunk_id"])

    if row.get("id") is not None:
        return str(row["id"])

    raise ValueError(f"Row does not contain chunk_id or id: {row.keys()}")


def result_ids(results: List[Dict[str, Any]], k: int) -> List[str]:
    return [normalize_chunk_id(row) for row in results[:k]]


def hybrid_results_path(
        ef_search: int,
        semantic_top_k: int,
        lexical_top_k: int,
        chunking_strategy: str,
) -> str:
    safe_strategy = chunking_strategy.replace("/", "_").replace(" ", "_")

    return os.path.join(
        RESULTS_DIR,
        (
            f"benchmark_results_{safe_strategy}_hybrid_"
            f"semantic_top_{semantic_top_k}_lexical_top_{lexical_top_k}_"
            f"ef_search_{ef_search}.json"
        ),
    )


# -----------------------------
# Retrieval calls
# -----------------------------

def semantic_search_hnsw(
        query_embedding: List[float],
        top_k: int,
        ef_search: int,
) -> Tuple[List[Dict[str, Any]], float]:
    rpc_args = {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "ef_search": ef_search,
    }

    start = time.perf_counter()

    response = supabase.rpc(
        SEMANTIC_RPC,
        rpc_args,
    ).execute()

    latency_ms = (time.perf_counter() - start) * 1000

    rows = response.data or []

    for idx, row in enumerate(rows, start=1):
        row["retrieval_source"] = "semantic"
        row["semantic_rank"] = idx
        row["semantic_score"] = row.get("similarity")
        row["lexical_rank"] = None
        row["lexical_score"] = None
        row["retrieval_sources"] = ["semantic"]

    return rows, latency_ms


def lexical_search(
        query_text: str,
        top_k: int
) -> Tuple[List[Dict[str, Any]], float]:
    rpc_args = {
        "query_text": query_text,
        "match_count": top_k
    }

    start = time.perf_counter()

    response = supabase.rpc(
        LEXICAL_RPC,
        rpc_args,
    ).execute()

    latency_ms = (time.perf_counter() - start) * 1000

    rows = response.data or []

    for idx, row in enumerate(rows, start=1):
        row["retrieval_source"] = "lexical"
        row["semantic_rank"] = None
        row["semantic_score"] = None
        row["lexical_rank"] = idx
        row["lexical_score"] = row.get("lexical_rank")
        row["retrieval_sources"] = ["lexical"]

        # Keep shape compatible with semantic rows.
        if "similarity" not in row:
            row["similarity"] = None

    return rows, latency_ms


# -----------------------------
# Hybrid merge / dedupe
# -----------------------------

def merge_and_dedupe_candidates(
        semantic_results: List[Dict[str, Any]],
        lexical_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge semantic and lexical candidates by chunk_id.

    If a chunk appears in both:
    - keep one row
    - preserve semantic score/rank
    - preserve lexical score/rank
    - retrieval_sources = ["semantic", "lexical"]

    Ordering:
    - semantic-only and shared candidates preserve semantic order first
    - lexical-only candidates are appended in lexical order

    The reranker will decide the final order later.
    """

    merged_by_id: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []

    # Add semantic candidates first.
    for row in semantic_results:
        chunk_id = normalize_chunk_id(row)

        item = dict(row)
        item["chunk_id"] = chunk_id
        item["retrieval_sources"] = ["semantic"]
        item["semantic_rank"] = row.get("semantic_rank")
        item["semantic_score"] = row.get("semantic_score", row.get("similarity"))
        item["lexical_rank"] = None
        item["lexical_score"] = None

        merged_by_id[chunk_id] = item
        ordered_ids.append(chunk_id)

    # Add lexical candidates, deduping against semantic.
    for row in lexical_results:
        chunk_id = normalize_chunk_id(row)

        lexical_rank = row.get("lexical_rank")
        lexical_score = row.get("lexical_score", row.get("lexical_rank"))

        if chunk_id in merged_by_id:
            item = merged_by_id[chunk_id]

            sources = item.get("retrieval_sources", [])
            if "lexical" not in sources:
                sources.append("lexical")

            item["retrieval_sources"] = sources
            item["lexical_rank"] = lexical_rank
            item["lexical_score"] = lexical_score
            item["lexical_rank_score"] = row.get("lexical_rank")

            # Fill missing metadata/content from lexical row if needed.
            for field in [
                "title",
                "section_path",
                "content",
                "metadata",
                "document_id",
                "chunk_index",
            ]:
                if item.get(field) is None and row.get(field) is not None:
                    item[field] = row.get(field)

            continue

        item = dict(row)
        item["chunk_id"] = chunk_id
        item["retrieval_sources"] = ["lexical"]
        item["semantic_rank"] = None
        item["semantic_score"] = None
        item["lexical_rank"] = lexical_rank
        item["lexical_score"] = lexical_score
        item["lexical_rank_score"] = row.get("lexical_rank")

        merged_by_id[chunk_id] = item
        ordered_ids.append(chunk_id)

    merged = [merged_by_id[chunk_id] for chunk_id in ordered_ids]

    for idx, row in enumerate(merged, start=1):
        row["hybrid_candidate_rank"] = idx

    return merged


# -----------------------------
# Benchmark execution
# -----------------------------

def benchmark_hybrid_query(
        query: str,
        semantic_top_k: int,
        lexical_top_k: int,
        ef_search: int,
        repeat_count: int,
        warmup_count: int,
        chunking_strategy: Optional[str],
) -> Dict[str, Any]:
    query_embedding, embedding_latency_ms = embed_query(query)

    # Warmup calls, not recorded.
    for _ in range(warmup_count):
        semantic_search_hnsw(
            query_embedding=query_embedding,
            top_k=semantic_top_k,
            ef_search=ef_search,
        )
        lexical_search(
            query_text=query,
            top_k=lexical_top_k,
        )

    semantic_latencies = []
    lexical_latencies = []
    merge_latencies = []

    last_semantic_results: List[Dict[str, Any]] = []
    last_lexical_results: List[Dict[str, Any]] = []
    last_hybrid_results: List[Dict[str, Any]] = []

    for _ in range(repeat_count):
        semantic_results, semantic_latency_ms = semantic_search_hnsw(
            query_embedding=query_embedding,
            top_k=semantic_top_k,
            ef_search=ef_search,
        )

        lexical_results, lexical_latency_ms = lexical_search(
            query_text=query,
            top_k=lexical_top_k,
        )

        merge_start = time.perf_counter()

        hybrid_results = merge_and_dedupe_candidates(
            semantic_results=semantic_results,
            lexical_results=lexical_results,
        )

        merge_latency_ms = (time.perf_counter() - merge_start) * 1000

        semantic_latencies.append(semantic_latency_ms)
        lexical_latencies.append(lexical_latency_ms)
        merge_latencies.append(merge_latency_ms)

        last_semantic_results = semantic_results
        last_lexical_results = lexical_results
        last_hybrid_results = hybrid_results

    semantic_ids = set(result_ids(last_semantic_results, len(last_semantic_results)))
    lexical_ids = set(result_ids(last_lexical_results, len(last_lexical_results)))
    overlap_ids = semantic_ids.intersection(lexical_ids)

    return {
        "query": query,
        "embedding_latency_ms": embedding_latency_ms,

        "semantic_top_k": semantic_top_k,
        "lexical_top_k": lexical_top_k,
        "hybrid_candidate_count": len(last_hybrid_results),
        "semantic_candidate_count": len(last_semantic_results),
        "lexical_candidate_count": len(last_lexical_results),
        "semantic_lexical_overlap_count": len(overlap_ids),

        "retrieval_latency_summary": {
            "semantic": summarize_latencies(semantic_latencies),
            "lexical": summarize_latencies(lexical_latencies),
            "merge": summarize_latencies(merge_latencies),
            "total_retrieval_p50_ms": (
                    (summarize_latencies(semantic_latencies).get("p50_ms") or 0)
                    + (summarize_latencies(lexical_latencies).get("p50_ms") or 0)
                    + (summarize_latencies(merge_latencies).get("p50_ms") or 0)
            ),
        },

        # Keep all three to make debugging easy.
        "semantic_results": last_semantic_results,
        "lexical_results": last_lexical_results,

        # IMPORTANT:
        # Existing reranker can use this field as candidates.
        "results": last_hybrid_results,
        "result_chunk_ids": result_ids(last_hybrid_results, len(last_hybrid_results)),
    }


def run_hybrid_benchmark(
        semantic_top_k: int,
        lexical_top_k: int,
        ef_search: int,
        repeat_count: int,
        warmup_count: int,
        chunking_strategy: Optional[str],
) -> Dict[str, Any]:
    print("\nRunning HYBRID retrieval benchmark")
    print(
        f"queries={len(EVAL_QUERIES)} "
        f"semantic_top_k={semantic_top_k} "
        f"lexical_top_k={lexical_top_k} "
        f"ef_search={ef_search} "
        f"strategy={chunking_strategy} "
        f"repeats={repeat_count}"
    )

    query_results = []

    for idx, query in enumerate(EVAL_QUERIES, start=1):
        print(f"[{idx}/{len(EVAL_QUERIES)}] {query}")

        try:
            result = benchmark_hybrid_query(
                query=query,
                semantic_top_k=semantic_top_k,
                lexical_top_k=lexical_top_k,
                ef_search=ef_search,
                repeat_count=repeat_count,
                warmup_count=warmup_count,
                chunking_strategy=chunking_strategy,
            )

            query_results.append(result)

            total_p50 = result["retrieval_latency_summary"]["total_retrieval_p50_ms"]
            hybrid_count = result["hybrid_candidate_count"]
            overlap = result["semantic_lexical_overlap_count"]
            top_title = result["results"][0]["title"] if result["results"] else "NONE"

            print(
                f"  total_p50={total_p50:.2f}ms "
                f"hybrid_candidates={hybrid_count} "
                f"overlap={overlap} "
                f"top_title={top_title}"
            )

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

    total_p50_values = []
    semantic_p50_values = []
    lexical_p50_values = []
    hybrid_candidate_counts = []
    overlap_counts = []

    for item in query_results:
        summary = item.get("retrieval_latency_summary") or {}
        semantic_summary = summary.get("semantic") or {}
        lexical_summary = summary.get("lexical") or {}

        if summary.get("total_retrieval_p50_ms") is not None:
            total_p50_values.append(summary["total_retrieval_p50_ms"])

        if semantic_summary.get("p50_ms") is not None:
            semantic_p50_values.append(semantic_summary["p50_ms"])

        if lexical_summary.get("p50_ms") is not None:
            lexical_p50_values.append(lexical_summary["p50_ms"])

        if item.get("hybrid_candidate_count") is not None:
            hybrid_candidate_counts.append(item["hybrid_candidate_count"])

        if item.get("semantic_lexical_overlap_count") is not None:
            overlap_counts.append(item["semantic_lexical_overlap_count"])

    benchmark_output = {
        "metadata": {
            "mode": "hybrid",
            "created_at": now_iso(),
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "query_count": len(EVAL_QUERIES),
            "semantic_top_k": semantic_top_k,
            "lexical_top_k": lexical_top_k,
            "ef_search": ef_search,
            "repeat_count": repeat_count,
            "warmup_count": warmup_count,
            "chunking_strategy": chunking_strategy,
            "semantic_rpc": SEMANTIC_RPC,
            "lexical_rpc": LEXICAL_RPC,
            "dedupe_key": "chunk_id",
            "candidate_strategy": "semantic_top_k + lexical_top_k -> dedupe by chunk_id -> rerank later",
        },
        "summary": {
            "retrieval_total_p50_across_queries_ms": percentile(total_p50_values, 50),
            "retrieval_total_p95_across_queries_ms": percentile(total_p50_values, 95),
            "retrieval_total_avg_across_queries_ms": statistics.mean(total_p50_values)
            if total_p50_values else None,

            "semantic_p50_across_queries_ms": percentile(semantic_p50_values, 50),
            "lexical_p50_across_queries_ms": percentile(lexical_p50_values, 50),

            "avg_hybrid_candidate_count": statistics.mean(hybrid_candidate_counts)
            if hybrid_candidate_counts else None,
            "min_hybrid_candidate_count": min(hybrid_candidate_counts)
            if hybrid_candidate_counts else None,
            "max_hybrid_candidate_count": max(hybrid_candidate_counts)
            if hybrid_candidate_counts else None,

            "avg_semantic_lexical_overlap_count": statistics.mean(overlap_counts)
            if overlap_counts else None,
        },
        "queries": query_results,
    }

    ensure_results_dir()

    output_path = hybrid_results_path(
        ef_search=ef_search,
        semantic_top_k=semantic_top_k,
        lexical_top_k=lexical_top_k,
        chunking_strategy=chunking_strategy or "all",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved hybrid benchmark results to {output_path}")
    print("Summary:", json.dumps(benchmark_output["summary"], indent=2))

    return benchmark_output


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hybrid retrieval: semantic HNSW + lexical BM25-style search."
    )

    parser.add_argument(
        "--semantic-top-k",
        type=int,
        default=DEFAULT_SEMANTIC_TOP_K,
        help="Number of semantic HNSW candidates per query.",
    )

    parser.add_argument(
        "--lexical-top-k",
        type=int,
        default=DEFAULT_LEXICAL_TOP_K,
        help="Number of lexical/BM25-style candidates per query.",
    )

    parser.add_argument(
        "--ef-search",
        type=int,
        default=DEFAULT_EF_SEARCH,
        help="HNSW ef_search value.",
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

    parser.add_argument(
        "--chunking-strategy",
        default=DEFAULT_CHUNKING_STRATEGY,
        help="Chunking strategy filter for lexical retrieval, e.g. section_aware_v2.",
    )

    args = parser.parse_args()

    run_hybrid_benchmark(
        semantic_top_k=args.semantic_top_k,
        lexical_top_k=args.lexical_top_k,
        ef_search=args.ef_search,
        repeat_count=args.repeat_count,
        warmup_count=args.warmup_count,
        chunking_strategy=args.chunking_strategy,
    )


if __name__ == "__main__":
    main()