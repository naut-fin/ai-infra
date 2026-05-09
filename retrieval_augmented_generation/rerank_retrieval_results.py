import os
import json
import time
import argparse
import statistics
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import cohere
import math


load_dotenv()

RESULTS_DIR = "benchmark_results"

COHERE_API_KEY = os.environ["COHERE_API_KEY"]
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

co = cohere.ClientV2(api_key=COHERE_API_KEY)

ef_search_list = [120]

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


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


def build_rerank_document(row: Dict[str, Any]) -> str:
    """
    Cohere accepts strings or structured documents.
    For now, use a compact text format.

    Including title helps reranker judge relevance better.
    """
    title = row.get("title", "")
    content = row.get("content", "")

    return f"Title: {title}\nContent: {content}"


def rerank_query_results(
        query: str,
        rows: List[Dict[str, Any]],
        rerank_top_k: int,
) -> Dict[str, Any]:
    """
    Rerank candidate rows for one query.

    Input:
      rows = original Flat/HNSW retrieval results

    Output:
      reranked rows ordered by Cohere relevance score
    """
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
                "original_rank": original_idx + 1,
                "rerank_score": result.relevance_score,
            }
        )

    return {
        "reranked_results": reranked_results,
        "rerank_latency_ms": rerank_latency_ms,
    }


def rerank_benchmark_file(
        input_path: str,
        output_path: str,
        candidate_limit: int,
        rerank_top_k: int,
) -> Dict[str, Any]:
    """
    Rerank every query in a benchmark result file.

    Example:
      benchmark_results_hnsw.json
        -> reranked_results_hnsw.json
    """
    benchmark = load_json(input_path)

    rerank_latencies = []
    reranked_queries = []

    for idx, item in enumerate(benchmark["queries"], start=1):
        query = item["query"]

        # Candidate limit controls how many vector results we give to reranker.
        # Example: HNSW top 20 -> rerank top 5.
        candidates = (
            item.get("expanded_results")
            or item.get("original_top_results")
            or item.get("results")
            or []
        )[:candidate_limit]

        print(f"[{idx}/{len(benchmark['queries'])}] Reranking: {query}")

        rerank_output = rerank_query_results(
            query=query,
            rows=candidates,
            rerank_top_k=rerank_top_k,
        )

        rerank_latencies.append(rerank_output["rerank_latency_ms"])

        reranked_queries.append(
            {
                "query": query,
                "candidate_count": len(candidates),
                "rerank_top_k": rerank_top_k,
                "rerank_latency_ms": rerank_output["rerank_latency_ms"],
                "original_top_results": candidates[:rerank_top_k],
                "reranked_results": rerank_output["reranked_results"],
            }
        )

        if rerank_output["reranked_results"]:
            top = rerank_output["reranked_results"][0]
            print(
                f"  top_after_rerank={top.get('title')} "
                f"score={top.get('rerank_score'):.4f} "
                f"original_rank={top.get('original_rank')}"
            )

    output = {
        "metadata": {
            "input_path": input_path,
            "candidate_limit": candidate_limit,
            "rerank_top_k": rerank_top_k,
            "rerank_model": COHERE_RERANK_MODEL,
        },
        "summary": {
            "rerank_latency": summarize(rerank_latencies),
        },
        "queries": reranked_queries,
    }

    save_json(output_path, output)

    print("\nSaved reranked results:")
    print(output_path)
    print("\nRerank latency summary:")
    print(json.dumps(output["summary"], indent=2))

    return output


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


def ndcg_at_k(flat_ids, retrieved_ids,  top_k):
    """
    retrieved_ids: HNSW result chunk ids in ranked order
    flat_ids: Flat result chunk ids in ranked order
    k: cutoff, e.g. 10 or 50
    """

    # Assign higher relevance to higher-ranked Flat results.
    # Flat rank 1 gets k, Flat rank 2 gets k-1, ...

    k = min(top_k, len(flat_ids))

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

def compare_reranked_flat_vs_hnsw(
        flat_reranked_path: str,
        hnsw_reranked_path: str,
        top_k: int,
        ef_search:int,
) -> Dict[str, Any]:
    """
    Compare reranked HNSW against reranked Flat.

    Important:
    This tells us whether HNSW+rerank produces the same final top-k
    as Flat+rerank.
    """
    flat = load_json(flat_reranked_path)
    hnsw = load_json(hnsw_reranked_path)

    flat_by_query = {item["query"]: item for item in flat["queries"]}
    hnsw_by_query = {item["query"]: item for item in hnsw["queries"]}

    per_query = []
    recalls = []
    ndcg_list = []

    for query, flat_item in flat_by_query.items():
        hnsw_item = hnsw_by_query.get(query)
        if not hnsw_item:
            continue

        flat_ids = [
            row["chunk_id"]
            for row in flat_item.get("reranked_results", [])[:top_k]
        ]

        hnsw_ids = [
            row["chunk_id"]
            for row in hnsw_item.get("reranked_results", [])[:top_k]
        ]

        overlap = set(flat_ids).intersection(set(hnsw_ids))
        recall = len(overlap) / min(top_k, len(flat_ids)) if flat_ids else 0.0
        ndcg = ndcg_at_k(flat_ids, hnsw_ids, top_k)

        recalls.append(recall)

        ndcg_list.append(ndcg)

        per_query.append(
            {
                "query": query,
                f"reranked_recall@{top_k}": recall,
                f"ndcg@{top_k}":ndcg,
                "flat_top_title_after_rerank": (
                    flat_item["reranked_results"][0]["title"]
                    if flat_item.get("reranked_results")
                    else None
                ),
                "hnsw_top_title_after_rerank": (
                    hnsw_item["reranked_results"][0]["title"]
                    if hnsw_item.get("reranked_results")
                    else None
                ),
                "flat_top_score": (
                    flat_item["reranked_results"][0]["rerank_score"]
                    if flat_item.get("reranked_results")
                    else None
                ),
                "hnsw_top_score": (
                    hnsw_item["reranked_results"][0]["rerank_score"]
                    if hnsw_item.get("reranked_results")
                    else None
                ),
            }
        )

    output = {
        "metadata": {
            "flat_reranked_path": flat_reranked_path,
            "hnsw_reranked_path": hnsw_reranked_path,
            "top_k": top_k,
        },
        "summary": {
            f"avg_reranked_recall@{top_k}": statistics.mean(recalls) if recalls else None,
            f"min_reranked_recall@{top_k}": min(recalls) if recalls else None,
            f"avg_ndcg@{top_k}":statistics.mean(ndcg_list) if ndcg_list else None,
            f"min_ndcg@{top_k}": min(ndcg_list) if ndcg_list else None,
        },
        "per_query": per_query,
    }

    output_path = os.path.join(
        RESULTS_DIR,
        f"compare_reranked_flat_vs_hnsw_top_{top_k}.json",
    )

    save_json(output_path, output)

    print("\nSaved reranked comparison:")
    print(output_path)
    print(json.dumps(output["summary"], indent=2))

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Rerank Flat/HNSW retrieval results using Cohere Rerank."
    )

    parser.add_argument(
        "--mode",
        choices=["flat", "hnsw"],
        help="Which benchmark result file to rerank.",
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=20,
        help="How many retrieved candidates to send to reranker.",
    )

    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=5,
        help="How many results to keep after reranking.",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare reranked Flat vs reranked HNSW.",
    )

    parser.add_argument(
        "--compare-top-k",
        type=int,
        default=5,
        help="Top-k used for reranked Flat-vs-HNSW comparison.",
    )

    parser.add_argument(
        "--input",
        help="Optional benchmark JSON path to rerank. Overrides the default path inferred from --mode.",
    )

    parser.add_argument(
        "--output",
        help="Optional output JSON path for reranked results. Overrides the default path inferred from --mode.",
    )

    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.compare:
        for ef_search in ef_search_list:
            compare_reranked_flat_vs_hnsw(
                flat_reranked_path=os.path.join(
                    RESULTS_DIR,
                    "reranked_results_context_aware_flat.json",
                ),
                hnsw_reranked_path=os.path.join(
                    RESULTS_DIR,
                    f"reranked_results_context_aware_ef_search_{ef_search}_hnsw.json",
                ),
                top_k=args.compare_top_k,
                ef_search=ef_search,
            )
        return

    if not args.mode:
        raise ValueError("Provide --mode flat|hnsw or use --compare")


    for ef_search in ef_search_list:
        input_path = args.input or os.path.join(
            RESULTS_DIR,
            f"benchmark_results_context_aware_{args.mode}.json",
        )

        output_path = args.output or os.path.join(
            RESULTS_DIR,
            f"reranked_results_context_aware_{args.mode}.json",
        )

        if args.mode == "hnsw" and not args.input:
            input_path = os.path.join(
                RESULTS_DIR,
                f"benchmark_results_context_aware_neighbor_expanded_ef_search_{ef_search}_{args.mode}.json",
            )

        if args.mode == "hnsw" and not args.output:
            output_path = os.path.join(
                RESULTS_DIR,
                f"reranked_results_context_aware_neighbor_expanded_ef_search_{ef_search}_{args.mode}.json",
            )


        rerank_benchmark_file(
            input_path=input_path,
            output_path=output_path,
            candidate_limit=args.candidate_limit,
            rerank_top_k=args.rerank_top_k,
        )


if __name__ == "__main__":
    main()
