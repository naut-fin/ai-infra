import argparse
import json
import os
import time
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar
from collections import defaultdict

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
from questions import EVAL_QUERIES

# The idea is in context aware chunking the content does not contain overlap, but rather are confined, this is done as most of the chunks do not start or end mid sentence, so doing that seems wasteful,
# however the issue around breaking knowledge or continuity of meaning would still exist so in order to address that we are performing, expand neighboring to pull N-1, N, N+1 chunks for a given N chunk so the LLM
# get('s complete meaning and not broken context or meaning, now why we do not keep overlap between chunks to reduce noise and we do not store overlap inside the chunk content because we are trying to
# avoid polluting the LLM context with duplicated/fragmented text.)


load_dotenv()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

openai_client: Optional[OpenAI] = None
supabase: Optional[Client] = None
T = TypeVar("T")

SUPABASE_REQUEST_MAX_ATTEMPTS = 4
SUPABASE_RETRY_BASE_DELAY_SECONDS = 1.0


def get_openai_client() -> OpenAI:
    global openai_client

    if openai_client is None:
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    return openai_client


def get_supabase_client() -> Client:
    global supabase

    if supabase is None:
        supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )

    return supabase


def reset_supabase_client() -> None:
    global supabase
    supabase = None


def execute_supabase_request(operation_name: str, request_factory: Callable[[], T]) -> T:
    """
    Retries transient transport errors from long Supabase/PostgREST runs.

    Supabase can terminate an HTTP/2 connection after many streams; rebuilding the
    client before retrying avoids reusing a stale connection from the pool.
    """

    for attempt in range(1, SUPABASE_REQUEST_MAX_ATTEMPTS + 1):
        try:
            return request_factory()
        except httpx.TransportError as exc:
            if attempt == SUPABASE_REQUEST_MAX_ATTEMPTS:
                raise

            delay = SUPABASE_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logging.warning(
                "%s failed with transient HTTP transport error on attempt %s/%s: %s. "
                "Retrying in %.1fs with a fresh Supabase client.",
                operation_name,
                attempt,
                SUPABASE_REQUEST_MAX_ATTEMPTS,
                exc,
                delay,
            )
            reset_supabase_client()
            time.sleep(delay)

    raise RuntimeError(f"{operation_name} failed unexpectedly")


def load_queries(path: str) -> List[str]:
    """
    Supports:
    1. JSON list: ["query1", "query2"]
    2. JSON object with "queries": [{"query": "..."}] or ["..."]
    3. JSONL with {"query": "..."}
    """

    if path.endswith(".jsonl"):
        queries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    queries.append(item["query"])
        return queries

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            return data
        return [x["query"] for x in data]

    if isinstance(data, dict):
        raw_queries = data.get("queries", [])
        if all(isinstance(x, str) for x in raw_queries):
            return raw_queries
        return [x["query"] for x in raw_queries]

    raise ValueError(f"Unsupported query file format: {path}")


def embed_query(query: str) -> List[float]:
    response = get_openai_client().embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


def hnsw_search(
        query_embedding: List[float],
        match_count: int,
        ef_search: int,
        strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Assumes you have an RPC like:

    match_document_chunks(
      query_embedding vector(1536),
      match_count int,
      ef_search int
    )

    OR a strategy-aware RPC if you add one.

    Returned rows should include:
      chunk_id
      document_id
      title
      content
      similarity
      chunk_index OR metadata.chunk_index
      metadata
    """

    params = {
        "query_embedding": query_embedding,
        "match_count": match_count,
        "ef_search": ef_search,
    }

    # If your SQL RPC supports target strategy, uncomment:
    # if strategy:
    #     params["target_chunking_strategy"] = strategy

    response = execute_supabase_request(
        "match_document_chunks RPC",
        lambda: get_supabase_client().rpc("match_document_chunks", params).execute(),
    )
    rows = response.data or []

    # If your RPC does not filter strategy, filter here using metadata.
    # This only works if metadata is returned by the RPC.
    if strategy:
        filtered = []
        for row in rows:
            metadata = row.get("metadata") or {}
            row_strategy = metadata.get("chunking_strategy")
            if row_strategy == strategy:
                filtered.append(row)
        # WARNING: client-side filtering can reduce candidate count.
        # Better to filter in SQL RPC.
        if filtered:
            rows = filtered

    return rows


def normalize_chunk_index(row: Dict[str, Any]) -> Optional[int]:
    if row.get("chunk_index") is not None:
        return int(row["chunk_index"])

    metadata = row.get("metadata") or {}
    if metadata.get("chunk_index") is not None:
        return int(metadata["chunk_index"])

    return None


def fetch_neighbor_chunks(
        document_id: str,
        center_chunk_index: int,
        neighbor_window: int,
        same_section_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetches chunks with chunk_index in [N-window, N+window]
    from the same document.

    If same_section_path is provided and metadata.section_path exists,
    filters neighbors to the same section.
    """

    start_index = max(0, center_chunk_index - neighbor_window)
    end_index = center_chunk_index + neighbor_window

    response = execute_supabase_request(
        "fetch neighbor chunks",
        lambda: (
            get_supabase_client()
            .table("document_chunks")
            .select("id,document_id,chunk_index,content,token_count,metadata")
            .eq("document_id", document_id)
            .gte("chunk_index", start_index)
            .lte("chunk_index", end_index)
            .order("chunk_index")
            .execute()
        ),
    )

    rows = response.data or []

    if same_section_path:
        filtered_rows = []
        for row in rows:
            metadata = row.get("metadata") or {}
            if metadata.get("section_path") == same_section_path:
                filtered_rows.append(row)
        rows = filtered_rows

    return rows


def build_expanded_passage(
        seed: Dict[str, Any],
        neighbors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Builds one rerankable passage for a seed chunk.

    The passage includes N-1/N/N+1 when available.
    """

    seed_metadata = seed.get("metadata") or {}
    title = seed.get("title") or seed_metadata.get("article_title", "")
    seed_chunk_index = normalize_chunk_index(seed)
    section_path = seed_metadata.get("section_path", "")

    sorted_neighbors = sorted(
        neighbors,
        key=lambda x: x.get("chunk_index", 0),
    )

    passage_parts = []

    for n in sorted_neighbors:
        n_metadata = n.get("metadata") or {}
        n_index = n.get("chunk_index")
        n_section = n_metadata.get("section_title") or n_metadata.get("section_path") or ""

        passage_parts.append(
            f"[chunk_index={n_index} section={n_section}]\n{n.get('content', '')}"
        )

    passage_text = "\n\n".join(passage_parts).strip()

    neighbor_chunk_ids = [row.get("id") for row in sorted_neighbors]
    neighbor_chunk_indexes = [row.get("chunk_index") for row in sorted_neighbors]

    return {
        "chunk_id": seed.get("chunk_id") or seed.get("id"),
        "document_id": seed.get("document_id"),
        "title": title,
        "content": passage_text,
        "similarity": seed.get("similarity"),
        "seed_chunk_id": seed.get("chunk_id") or seed.get("id"),
        "seed_chunk_index": seed_chunk_index,
        "neighbor_chunk_ids": neighbor_chunk_ids,
        "neighbor_chunk_indexes": neighbor_chunk_indexes,
        "section_path": section_path,
        "original_rank": seed.get("original_rank"),
        "is_neighbor_expanded": True,
    }


def expand_results_with_neighbors(
        seed_results: List[Dict[str, Any]],
        neighbor_window: int,
        restrict_to_same_section: bool,
) -> List[Dict[str, Any]]:
    """
    Converts top-k seed chunks into top-k expanded passages.
    One expanded passage per seed chunk.
    """

    expanded_passages = []

    seen_seed_keys = set()

    for rank, seed in enumerate(seed_results, start=1):
        document_id = seed.get("document_id")
        chunk_index = normalize_chunk_index(seed)

        if not document_id or chunk_index is None:
            continue

        seed_metadata = seed.get("metadata") or {}
        same_section_path = (
            seed_metadata.get("section_path")
            if restrict_to_same_section
            else None
        )

        seed_key = (document_id, chunk_index)
        if seed_key in seen_seed_keys:
            continue
        seen_seed_keys.add(seed_key)

        seed = dict(seed)
        seed["original_rank"] = rank

        neighbors = fetch_neighbor_chunks(
            document_id=document_id,
            center_chunk_index=chunk_index,
            neighbor_window=neighbor_window,
            same_section_path=same_section_path,
        )

        if not neighbors:
            # Fallback to seed-only passage.
            neighbors = [
                {
                    "id": seed.get("chunk_id") or seed.get("id"),
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "content": seed.get("content", ""),
                    "metadata": seed_metadata,
                }
            ]

        expanded_passages.append(
            build_expanded_passage(seed=seed, neighbors=neighbors)
        )

    return expanded_passages


def build_metadata(
        queries: List[str],
        top_k: int,
        ef_search: int,
        neighbor_window: int,
        strategy: Optional[str],
        restrict_to_same_section: bool,
) -> Dict[str, Any]:
    return {
        "retrieval_mode": "hnsw_neighbor_expanded",
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "seed_top_k": top_k,
        "ef_search": ef_search,
        "neighbor_window": neighbor_window,
        "strategy": strategy,
        "restrict_to_same_section": restrict_to_same_section,
        "num_queries": len(queries),
    }


def build_results_payload(metadata: Dict[str, Any], query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies_ms = [
        result["retrieval_latency_ms"]
        for result in query_results
        if result.get("retrieval_latency_ms") is not None
    ]

    return {
        "metadata": metadata,
        "summary": summarize_latencies(latencies_ms),
        "queries": query_results,
    }


def load_existing_results(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_resume_results(
        existing_results: Dict[str, Any],
        expected_metadata: Dict[str, Any],
        queries: List[str],
) -> List[Dict[str, Any]]:
    existing_metadata = existing_results.get("metadata") or {}

    metadata_keys = [
        "retrieval_mode",
        "embedding_model",
        "seed_top_k",
        "ef_search",
        "neighbor_window",
        "strategy",
        "restrict_to_same_section",
        "num_queries",
    ]
    for key in metadata_keys:
        if existing_metadata.get(key) != expected_metadata.get(key):
            raise ValueError(
                f"Cannot resume from existing output: metadata.{key} is "
                f"{existing_metadata.get(key)!r}, expected {expected_metadata.get(key)!r}."
            )

    existing_query_results = existing_results.get("queries") or []
    if len(existing_query_results) > len(queries):
        raise ValueError(
            f"Cannot resume from existing output: it has {len(existing_query_results)} "
            f"queries, but this run only has {len(queries)}."
        )

    for idx, result in enumerate(existing_query_results):
        if result.get("query") != queries[idx]:
            raise ValueError(
                f"Cannot resume from existing output: query {idx + 1} is "
                f"{result.get('query')!r}, expected {queries[idx]!r}."
            )

    return existing_query_results


def run_benchmark(
        queries: List[str],
        top_k: int,
        ef_search: int,
        neighbor_window: int,
        strategy: Optional[str],
        restrict_to_same_section: bool,
        output_path: Optional[str] = None,
        resume: bool = False,
) -> Dict[str, Any]:
    metadata = build_metadata(
        queries=queries,
        top_k=top_k,
        ef_search=ef_search,
        neighbor_window=neighbor_window,
        strategy=strategy,
        restrict_to_same_section=restrict_to_same_section,
    )
    all_results: List[Dict[str, Any]] = []

    if resume:
        if not output_path:
            raise ValueError("--resume requires an output path.")

        existing_results = load_existing_results(output_path)
        if existing_results:
            all_results = validate_resume_results(
                existing_results=existing_results,
                expected_metadata=metadata,
                queries=queries,
            )
            logging.info(
                "Resuming from %s with %s/%s queries already complete.",
                output_path,
                len(all_results),
                len(queries),
            )
        else:
            logging.info("Resume requested, but %s does not exist. Starting a new run.", output_path)

    start_index = len(all_results)

    for idx, query in enumerate(queries[start_index:], start=start_index + 1):
        logging.info("[%s/%s] query=%s", idx, len(queries), query)

        start = time.perf_counter()

        query_embedding = embed_query(query)

        seed_results = hnsw_search(
            query_embedding=query_embedding,
            match_count=top_k,
            ef_search=ef_search,
            strategy=strategy,
        )

        expanded_results = expand_results_with_neighbors(
            seed_results=seed_results,
            neighbor_window=neighbor_window,
            restrict_to_same_section=restrict_to_same_section,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        all_results.append(
            {
                "query": query,
                "retrieval_mode": "hnsw_neighbor_expanded",
                "seed_top_k": top_k,
                "expanded_candidate_count": len(expanded_results),
                "neighbor_window": neighbor_window,
                "restrict_to_same_section": restrict_to_same_section,
                "retrieval_latency_ms": elapsed_ms,
                "original_top_results": seed_results,
                "expanded_results": expanded_results,
            }
        )

        if output_path:
            save_json(output_path, build_results_payload(metadata, all_results))

    return build_results_payload(metadata, all_results)


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None

    values = sorted(values)
    if len(values) == 1:
        return values[0]

    rank = (len(values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower

    return values[lower] * (1 - weight) + values[upper] * weight


def summarize_latencies(latencies_ms: List[float]) -> Dict[str, Any]:
    if not latencies_ms:
        return {}

    return {
        "retrieval_p50_across_queries_ms": percentile(latencies_ms, 0.50),
        "retrieval_p95_across_queries_ms": percentile(latencies_ms, 0.95),
        "retrieval_avg_across_queries_ms": sum(latencies_ms) / len(latencies_ms),
        "retrieval_min_ms": min(latencies_ms),
        "retrieval_max_ms": max(latencies_ms),
    }


def save_json(path: str, data: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    os.replace(temp_path, path)

    logging.info("Saved results to %s", path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of HNSW seed chunks to retrieve.",
    )

    parser.add_argument(
        "--ef-search",
        type=int,
        default=120,
        help="HNSW ef_search value.",
    )

    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=1,
        help="1 means fetch N-1, N, N+1.",
    )

    parser.add_argument(
        "--strategy",
        default=None,
        help="Optional chunking_strategy filter, e.g. section_aware_v2.",
    )

    parser.add_argument(
        "--restrict-to-same-section",
        action="store_true",
        help="Only include neighbors with same metadata.section_path.",
    )

    parser.add_argument(
        "--output",
        default="benchmark_results/benchmark_results_context_aware_neighbor_expanded_ef_search_120_hnsw.json",
        help="Output JSON path.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the existing output file and skip queries already saved there.",
    )

    args = parser.parse_args()


    results = run_benchmark(
        queries=EVAL_QUERIES,
        top_k=args.top_k,
        ef_search=args.ef_search,
        neighbor_window=args.neighbor_window,
        strategy=args.strategy,
        restrict_to_same_section=args.restrict_to_same_section,
        output_path=args.output,
        resume=args.resume,
    )

    save_json(args.output, results)

    print("Summary:")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
