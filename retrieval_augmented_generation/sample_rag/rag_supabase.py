import os
import uuid
from typing import List, Dict, Any

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

load_dotenv()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS_PER_CHUNK = 500
CHUNK_OVERLAP = 80

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)


def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks


def embed_text(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def insert_document(title: str, text: str, source_url: str | None = None, metadata: Dict[str, Any] | None = None):
    metadata = metadata or {}

    doc_response = supabase.table("documents").insert({
        "title": title,
        "source_url": source_url,
        "metadata": metadata,
    }).execute()

    document_id = doc_response.data[0]["id"]

    chunks = chunk_text(text)

    rows = []
    for idx, chunk in enumerate(chunks):
        embedding = embed_text(chunk)

        rows.append({
            "document_id": document_id,
            "chunk_index": idx,
            "content": chunk,
            "token_count": len(tiktoken.get_encoding("cl100k_base").encode(chunk)),
            "metadata": {
                "chunk_index": idx,
                **metadata,
            },
            "embedding": embedding,
        })

    supabase.table("document_chunks").insert(rows).execute()

    return {
        "document_id": document_id,
        "chunks_inserted": len(rows),
    }


def search_documents(query: str, match_count: int = 5, similarity_threshold: float = 0.0):
    query_embedding = embed_text(query)

    response = supabase.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "similarity_threshold": similarity_threshold,
        },
    ).execute()

    return response.data


if __name__ == "__main__":

    sample_text = """
    Retrieval Augmented Generation improves language model answers by retrieving relevant context
    from a knowledge base before generating a response. A vector database stores embeddings for
    document chunks and enables semantic search over those chunks.
    """

    # inserted = insert_document(
    #     title="RAG Intro",
    #     text=sample_text,
    #     source_url=None,
    #     metadata={"category": "rag-learning"},
    # )
    #
    # print("Inserted:", inserted)

    results = search_documents("What is RAG and why do we use vector databases?")

    for r in results:
        print("\n---")
        print("Title:", r["title"])
        print("Similarity:", r["similarity"])
        print("Content:", r["content"])