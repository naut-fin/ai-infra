import os
import re
import json
import math
import argparse
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import tiktoken
from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

OUTPUT_DIR = "benchmark_results"

encoder = tiktoken.get_encoding("cl100k_base")


ENDING_PUNCTUATION = (
    ".",
    "?",
    "!",
    '"',
    "'",
    ")",
    "]",
)


def token_count(text: str) -> int:
    return len(encoder.encode(text or ""))


def percentile_nearest_rank(values: List[float], percentile: float) -> Optional[float]:
    """
    Nearest-rank percentile.
    percentile = 0.10 means p10.
    """
    if not values:
        return None

    sorted_values = sorted(values)
    n = len(sorted_values)

    rank = math.ceil(percentile * n)
    rank = max(1, min(rank, n))

    return sorted_values[rank - 1]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def starts_with_lowercase(text: str) -> bool:
    text = (text or "").strip()
    return bool(text) and text[0].islower()


def starts_with_punctuation(text: str) -> bool:
    text = (text or "").strip()
    return bool(text) and bool(re.match(r"^[^\w\s]", text))


def ends_mid_sentence(text: str) -> bool:
    """
    Heuristic:
    A chunk that does not end with sentence-like punctuation
    may have been cut mid-sentence.
    """
    text = (text or "").strip()

    if not text:
        return False

    return not text.endswith(ENDING_PUNCTUATION)


def starts_like_fragment(text: str) -> bool:
    """
    Heuristic for orphan/fragment chunks.
    Examples:
      "of the resulting logical disk..."
      "partitions. Usually..."
      ", a South African..."
    """
    text = (text or "").strip()

    if not text:
        return False

    if starts_with_lowercase(text):
        return True

    if starts_with_punctuation(text):
        return True

    # Starts with a continuation-style word.
    first_word = text.split()[0].lower().strip(",.;:")
    continuation_words = {
        "and",
        "or",
        "but",
        "because",
        "therefore",
        "however",
        "which",
        "that",
        "this",
        "these",
        "those",
        "it",
        "they",
        "he",
        "she",
    }

    return first_word in continuation_words


def looks_like_short_label(text: str) -> bool:
    """
    Detects tiny label/list chunks like:
      "(e) Eris"
      "short-period comets"
      "RAID-DP is another way..."
    """
    text = (text or "").strip()

    if not text:
        return False

    tokens = token_count(text)

    if tokens <= 8:
        return True

    # Very short text with no sentence ending.
    if tokens <= 15 and not text.endswith((".", "?", "!")):
        return True

    return False


def exact_duplicate_ratio(chunks: List[Dict[str, Any]]) -> float:
    """
    Counts exact duplicate normalized chunk content.
    This does not catch partial overlap, but catches full duplicates.
    """
    normalized_contents = [
        normalize_text(chunk.get("content", ""))
        for chunk in chunks
        if chunk.get("content")
    ]

    if not normalized_contents:
        return 0.0

    counts = Counter(normalized_contents)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)

    return duplicate_count / len(normalized_contents)


def token_jaccard(text_a: str, text_b: str) -> float:
    """
    Measures token-set overlap between two chunks.
    Useful as a proxy for overlap pollution.
    """
    tokens_a = set(encoder.encode(text_a or ""))
    tokens_b = set(encoder.encode(text_b or ""))

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def adjacent_overlap_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates adjacent chunk overlap within each document.
    Requires document_id and chunk_index.
    """
    chunks_by_document = defaultdict(list)

    for chunk in chunks:
        chunks_by_document[chunk.get("document_id")].append(chunk)

    overlaps = []

    for document_id, document_chunks in chunks_by_document.items():
        sorted_chunks = sorted(
            document_chunks,
            key=lambda c: c.get("chunk_index") if c.get("chunk_index") is not None else 0,
        )

        for prev_chunk, curr_chunk in zip(sorted_chunks, sorted_chunks[1:]):
            prev_content = prev_chunk.get("content", "")
            curr_content = curr_chunk.get("content", "")

            overlaps.append(token_jaccard(prev_content, curr_content))

    if not overlaps:
        return {
            "avg_adjacent_token_jaccard": 0.0,
            "median_adjacent_token_jaccard": 0.0,
            "p90_adjacent_token_jaccard": 0.0,
            "max_adjacent_token_jaccard": 0.0,
        }

    return {
        "avg_adjacent_token_jaccard": sum(overlaps) / len(overlaps),
        "median_adjacent_token_jaccard": statistics.median(overlaps),
        "p90_adjacent_token_jaccard": percentile_nearest_rank(overlaps, 0.90),
        "max_adjacent_token_jaccard": max(overlaps),
    }


def calculate_chunk_quality(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_chunks = len(chunks)

    if total_chunks == 0:
        return {
            "total_chunks": 0,
            "error": "No chunks found for the given filter.",
        }

    token_counts = []

    for chunk in chunks:
        stored_token_count = chunk.get("token_count")
        content = chunk.get("content", "")

        if stored_token_count is not None:
            token_counts.append(int(stored_token_count))
        else:
            token_counts.append(token_count(content))

    chunks_lt_40 = sum(1 for t in token_counts if t < 40)
    chunks_lt_80 = sum(1 for t in token_counts if t < 80)

    lowercase_starts = sum(
        1 for chunk in chunks if starts_with_lowercase(chunk.get("content", ""))
    )

    punctuation_starts = sum(
        1 for chunk in chunks if starts_with_punctuation(chunk.get("content", ""))
    )

    mid_sentence_ends = sum(
        1 for chunk in chunks if ends_mid_sentence(chunk.get("content", ""))
    )

    fragment_starts = sum(
        1 for chunk in chunks if starts_like_fragment(chunk.get("content", ""))
    )

    short_label_chunks = sum(
        1 for chunk in chunks if looks_like_short_label(chunk.get("content", ""))
    )

    overlap_stats = adjacent_overlap_stats(chunks)

    return {
        "total_chunks": total_chunks,

        "avg_tokens_per_chunk": sum(token_counts) / total_chunks,
        "median_tokens_per_chunk": statistics.median(token_counts),
        "min_tokens_per_chunk": min(token_counts),
        "max_tokens_per_chunk": max(token_counts),
        "p10_tokens_per_chunk": percentile_nearest_rank(token_counts, 0.10),
        "p90_tokens_per_chunk": percentile_nearest_rank(token_counts, 0.90),

        "chunks_lt_40_tokens": chunks_lt_40,
        "chunks_lt_80_tokens": chunks_lt_80,
        "pct_chunks_lt_40_tokens": safe_divide(chunks_lt_40, total_chunks),
        "pct_chunks_lt_80_tokens": safe_divide(chunks_lt_80, total_chunks),

        "chunks_starting_with_lowercase": lowercase_starts,
        "pct_chunks_starting_with_lowercase": safe_divide(lowercase_starts, total_chunks),

        "chunks_starting_with_punctuation": punctuation_starts,
        "pct_chunks_starting_with_punctuation": safe_divide(punctuation_starts, total_chunks),

        "chunks_ending_mid_sentence": mid_sentence_ends,
        "pct_chunks_ending_mid_sentence": safe_divide(mid_sentence_ends, total_chunks),

        "fragment_like_starts": fragment_starts,
        "pct_fragment_like_starts": safe_divide(fragment_starts, total_chunks),

        "short_label_like_chunks": short_label_chunks,
        "pct_short_label_like_chunks": safe_divide(short_label_chunks, total_chunks),

        "exact_duplicate_ratio": exact_duplicate_ratio(chunks),

        **overlap_stats,
    }


def fetch_chunks_for_strategy(
        supabase: Client,
        strategy: str,
        page_size: int = 1000,
        max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetches chunks joined with document metadata.

    Assumes:
      documents.metadata->>'chunking_strategy' = strategy

    Supabase Python client does not support arbitrary SQL joins directly
    unless foreign relationships are configured, so we use a two-step fetch:

    1. Fetch matching document IDs from documents.
    2. Fetch chunks for those document IDs in batches.
    """
    print(f"Fetching documents for strategy={strategy}")

    documents = []
    offset = 0

    while True:
        response = (
            supabase.table("documents")
            .select("id,title,metadata")
            .range(offset, offset + page_size - 1)
            .execute()
        )

        rows = response.data or []
        documents.extend(rows)

        if len(rows) < page_size:
            break

        offset += page_size

    document_ids = [doc["id"] for doc in documents]

    print(f"Found documents={len(document_ids)}")

    if not document_ids:
        return []

    title_by_document_id = {
        doc["id"]: doc.get("title")
        for doc in documents
    }

    all_chunks = []

    # Supabase .in_ can get large, so batch document IDs.
    document_id_batch_size = 100

    for batch_start in range(0, len(document_ids), document_id_batch_size):
        document_id_batch = document_ids[batch_start : batch_start + document_id_batch_size]

        offset = 0

        while True:
            query = (
                supabase.table("document_chunks")
                .select("id,document_id,content,token_count,metadata,chunk_index")
                .in_("document_id", document_id_batch)
                .range(offset, offset + page_size - 1)
            )

            response = query.execute()
            rows = response.data or []

            for row in rows:
                row["document_title"] = title_by_document_id.get(row.get("document_id"))

            all_chunks.extend(rows)

            if max_rows and len(all_chunks) >= max_rows:
                return all_chunks[:max_rows]

            if len(rows) < page_size:
                break

            offset += page_size

    print(f"Fetched chunks={len(all_chunks)}")

    return all_chunks


def get_shortest_chunks(chunks: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    enriched = []

    for chunk in chunks:
        content = chunk.get("content", "")
        t_count = chunk.get("token_count")

        if t_count is None:
            t_count = token_count(content)

        enriched.append(
            {
                "id": chunk.get("id"),
                "document_id": chunk.get("document_id"),
                "document_title": chunk.get("document_title"),
                "chunk_index": chunk.get("chunk_index"),
                "token_count": int(t_count),
                "content_preview": content[:500],
            }
        )

    return sorted(enriched, key=lambda x: x["token_count"])[:limit]


def get_fragment_examples(chunks: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    examples = []

    for chunk in chunks:
        content = chunk.get("content", "")

        if starts_like_fragment(content) or looks_like_short_label(content):
            t_count = chunk.get("token_count")
            if t_count is None:
                t_count = token_count(content)

            examples.append(
                {
                    "id": chunk.get("id"),
                    "document_id": chunk.get("document_id"),
                    "document_title": chunk.get("document_title"),
                    "chunk_index": chunk.get("chunk_index"),
                    "token_count": int(t_count),
                    "content_preview": content[:500],
                }
            )

    return examples[:limit]


def save_json(data: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        required=True,
        help="Chunking strategy value stored in documents.metadata->>'chunking_strategy'",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max chunks to analyze for quick testing",
    )

    args = parser.parse_args()

    strategy = args.strategy

    output_path = args.output or os.path.join(
        OUTPUT_DIR,
        f"chunk_quality_{strategy}.json",
    )

    supabase: Client = create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
    )

    chunks = fetch_chunks_for_strategy(
        supabase=supabase,
        strategy=strategy,
        max_rows=args.max_rows,
    )

    metrics = calculate_chunk_quality(chunks)

    report = {
        "strategy": strategy,
        "metrics": metrics,
        "examples": {
            "shortest_chunks": get_shortest_chunks(chunks, limit=20),
            "fragment_or_short_label_examples": get_fragment_examples(chunks, limit=20),
        },
    }

    save_json(report, output_path)

    print("Summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()