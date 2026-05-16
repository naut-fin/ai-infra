import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from questions import EVAL_QUERIES

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")

DEFAULT_INPUT_PATH = (
    "benchmark_results/reranked_results_context_aware_ef_search_120_hnsw.json"
)
DEFAULT_OUTPUT_PATH = (
    "benchmark_results/answers_context_aware_ef_search_120_hnsw.json"
)
DEFAULT_CONTEXT_FIELD = "reranked_results"
DEFAULT_MAX_CHUNKS = 10


def format_context(chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for idx, chunk in enumerate(chunks, start=1):
        similarity = chunk.get("similarity")
        similarity_line = (
            f"Similarity: {similarity:.4f}\n"
            if isinstance(similarity, (int, float))
            else ""
        )
        rerank_score = chunk.get("rerank_score")
        rerank_line = (
            f"Rerank score: {rerank_score:.4f}\n"
            if isinstance(rerank_score, (int, float))
            else ""
        )
        context_blocks.append(
            f"""
[Source {idx}]
Title: {chunk.get("title", "Untitled")}
{similarity_line}{rerank_line}Content:
{chunk.get("content", "")}
""".strip()
        )

    return "\n\n".join(context_blocks)


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    script_relative = Path(__file__).resolve().parent / path
    if script_relative.exists():
        return script_relative

    return candidate


def load_reranked_results(path: str) -> Dict[str, Any]:
    input_path = resolve_path(path)
    with input_path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def find_query_item(data: Dict[str, Any], question: str) -> Optional[Dict[str, Any]]:
    for item in data.get("queries", []):
        if item.get("query") == question:
            return item
    return None


def answer_with_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not chunks:
        return {
            "question": question,
            "answer": (
                "I don't know. I could not find relevant context in the "
                "knowledge base."
            ),
            "sources": [],
        }

    context = format_context(chunks)

    prompt = f"""
You are a grounded RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
- If the context contains the answer, answer clearly and concisely.
- If the context does not contain the answer, say: "I don't know based on the provided context."
- Do not use outside knowledge.
- Cite sources inline using [Source 1], [Source 2], etc.
- Do not mention similarity scores or rerank scores in the answer.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    response = client.responses.create(
        model=ANSWER_MODEL,
        input=prompt,
        temperature=0.2,
    )

    return {
        "question": question,
        "answer": response.output_text,
        "sources": chunks,
    }


def answer_question(
    question: str,
    results_data: Dict[str, Any],
    context_field: str = DEFAULT_CONTEXT_FIELD,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> Dict[str, Any]:
    query_item = find_query_item(results_data, question)

    if query_item is None:
        return {
            "question": question,
            "answer": (
                "I don't know. I could not find this query in the reranked "
                "results file."
            ),
            "sources": [],
        }

    chunks = query_item.get(context_field, [])[:max_chunks]
    return answer_with_chunks(question, chunks)


def answer_queries(
    input_path: str = DEFAULT_INPUT_PATH,
    context_field: str = DEFAULT_CONTEXT_FIELD,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    only_query: Optional[str] = None,
) -> Dict[str, Any]:
    data = load_reranked_results(input_path)
    query_set = {only_query} if only_query else set(EVAL_QUERIES)

    skipped_count = 0
    answered_queries = []

    for item in data.get("queries", []):
        question = item.get("query")
        if question not in query_set:
            skipped_count += 1
            continue

        chunks = item.get(context_field, [])[:max_chunks]
        result = answer_with_chunks(question, chunks)
        answered_queries.append(
            {
                "query": question,
                "answer_response": {
                    "model": ANSWER_MODEL,
                    "answer": result["answer"],
                },
                "source_chunk_ids": [
                    source.get("chunk_id") for source in result["sources"]
                ],
                "chunks": result["sources"],
            }
        )
        print_result(result)

    return {
        "input_path": str(resolve_path(input_path)),
        "metadata": {
            "input_path": str(resolve_path(input_path)),
            "input_metadata": data.get("metadata", {}),
            "answer_model": ANSWER_MODEL,
            "context_field": context_field,
            "max_chunks": max_chunks,
            "query_count": len(answered_queries),
        },
        "summary": {
            "answered_count": len(answered_queries),
            "skipped_count": skipped_count,
        },
        "queries": answered_queries,
    }


def answer_and_save(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    context_field: str = DEFAULT_CONTEXT_FIELD,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    only_query: Optional[str] = None,
) -> Dict[str, Any]:
    output = answer_queries(
        input_path=input_path,
        context_field=context_field,
        max_chunks=max_chunks,
        only_query=only_query,
    )
    save_json(output_path, output)

    return {
        "input_path": output["input_path"],
        "output_path": str(resolve_path(output_path)),
        "answered_count": output["summary"]["answered_count"],
        "skipped_count": output["summary"]["skipped_count"],
    }


def answer_question_from_file(
    question: str,
    input_path: str = DEFAULT_INPUT_PATH,
    context_field: str = DEFAULT_CONTEXT_FIELD,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> Dict[str, Any]:
    data = load_reranked_results(input_path)
    return answer_question(
        question=question,
        results_data=data,
        context_field=context_field,
        max_chunks=max_chunks,
    )


def print_result(result: Dict[str, Any]) -> None:
    print("\n==============================")
    print("QUESTION")
    print("==============================")
    print(result["question"])

    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(textwrap.fill(result["answer"], width=100))

    print("\n==============================")
    print("SOURCES")
    print("==============================")

    if not result["sources"]:
        print("No sources found.")
        return

    for idx, source in enumerate(result["sources"], start=1):
        print(f"\n[Source {idx}]")
        print("Title:", source.get("title", "Untitled"))
        similarity = source.get("similarity")
        if isinstance(similarity, (int, float)):
            print("Similarity:", round(similarity, 4))
        rerank_score = source.get("rerank_score")
        if isinstance(rerank_score, (int, float)):
            print("Rerank score:", round(rerank_score, 4))
        print("Content preview:", source.get("content", "")[:300].replace("\n", " "))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Answer EVAL_QUERIES using chunks already saved in a reranked "
            "benchmark results file, then write a separate answer file."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--context-field", default=DEFAULT_CONTEXT_FIELD)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument(
        "--query",
        default=None,
        help="Answer only this query instead of every query in EVAL_QUERIES.",
    )

    args = parser.parse_args()

    summary = answer_and_save(
        input_path=args.input,
        output_path=args.output,
        context_field=args.context_field,
        max_chunks=args.max_chunks,
        only_query=args.query,
    )
    print("\nSaved answers:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
