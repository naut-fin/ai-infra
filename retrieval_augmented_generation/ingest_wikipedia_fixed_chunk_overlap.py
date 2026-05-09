import os
import time
import logging
import hashlib
from typing import Any, Dict, List, Iterable

import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
from supabase import create_client, Client

load_dotenv()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

MAX_TOKENS_PER_CHUNK = 256
CHUNK_OVERLAP = 24

EMBEDDING_BATCH_SIZE = 64
SUPABASE_INSERT_BATCH_SIZE = 100
MAX_RETRIES = 5

MIN_TEXT_CHARS = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)

encoder = tiktoken.get_encoding("cl100k_base")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def with_retry(fn, description: str, max_retries: int = MAX_RETRIES):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            sleep_seconds = min(2 ** attempt, 30)

            logging.warning(
                "%s failed. attempt=%s/%s error=%s sleeping=%ss",
                description,
                attempt,
                max_retries,
                repr(e),
                sleep_seconds,
            )

            time.sleep(sleep_seconds)

    raise RuntimeError(f"{description} failed after {max_retries} retries") from last_error


def batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def chunk_text(text: str) -> List[str]:
    tokens = encoder.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + MAX_TOKENS_PER_CHUNK
        chunk_tokens = tokens[start:end]
        chunk = encoder.decode(chunk_tokens).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(tokens):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def embed_batch(texts: List[str]) -> List[List[float]]:
    def call_openai():
        response = openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    return with_retry(call_openai, f"OpenAI embedding batch size={len(texts)}")


def insert_document(title: str, source_url: str | None, metadata: Dict[str, Any]) -> str:
    def call_supabase():
        response = (
            supabase.table("documents")
            .insert(
                {
                    "title": title,
                    "source_url": source_url,
                    "metadata": metadata,
                }
            )
            .execute()
        )
        return response.data[0]["id"]

    return with_retry(call_supabase, f"insert document title={title}")


def insert_chunks(rows: List[Dict[str, Any]]) -> None:
    def call_supabase():
        return supabase.table("document_chunks").insert(rows).execute()

    with_retry(call_supabase, f"insert chunks batch size={len(rows)}")


def ingest_wikipedia(max_articles: int = 10):
    logging.info("Loading Simple Wikipedia dataset")

    dataset = load_dataset(
        "wikipedia",
        "20220301.simple",
        split=f"train[:{max_articles}]",
        trust_remote_code=True
    )

    logging.info("Loaded articles=%s", len(dataset))

    total_documents = 0
    total_chunks = 0

    for item in tqdm(dataset, desc="Ingesting Wikipedia articles"):
        title = item.get("title", "").strip()
        text = item.get("text", "").strip()
        url = item.get("url")

        if not title or not text:
            continue

        if len(text) < MIN_TEXT_CHARS:
            continue

        article_hash = stable_hash(text)
        chunks = chunk_text(text)

        if not chunks:
            continue

        document_id = insert_document(
            title=title,
            source_url=url,
            metadata={
                "source": "wikipedia",
                "dataset": "20220301.simple",
                "article_id": item.get("id"),
                "article_hash": article_hash,
                "text_chars": len(text),
                "chunk_count": len(chunks),
            },
        )

        all_rows = []

        for chunk_batch_start, chunk_batch in enumerate(
                batched(chunks, EMBEDDING_BATCH_SIZE)
        ):
            embeddings = embed_batch(chunk_batch)

            for local_idx, (chunk, embedding) in enumerate(zip(chunk_batch, embeddings)):
                global_chunk_index = chunk_batch_start * EMBEDDING_BATCH_SIZE + local_idx

                all_rows.append(
                    {
                        "document_id": document_id,
                        "chunk_index": global_chunk_index,
                        "content": chunk,
                        "token_count": len(encoder.encode(chunk)),
                        "metadata": {
                            "source": "wikipedia",
                            "dataset": "20220301.simple",
                            "article_title": title,
                            "article_url": url,
                            "article_hash": article_hash,
                            "chunk_index": global_chunk_index,
                        },
                        "embedding": embedding,
                    }
                )

        for rows_batch in batched(all_rows, SUPABASE_INSERT_BATCH_SIZE):
            insert_chunks(rows_batch)

        total_documents += 1
        total_chunks += len(all_rows)

        logging.info(
            "Inserted title=%s chunks=%s total_documents=%s total_chunks=%s",
            title,
            len(all_rows),
            total_documents,
            total_chunks,
        )

    logging.info(
        "DONE. total_documents=%s total_chunks=%s",
        total_documents,
        total_chunks,
    )


if __name__ == "__main__":
    ingest_wikipedia(max_articles=20000)