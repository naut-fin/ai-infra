import os
import textwrap
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from retrieval_augmented_generation.sample_rag.rag_supabase import search_documents

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")

DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.4


def format_context(chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for idx, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"""
[Source {idx}]
Title: {chunk["title"]}
Similarity: {chunk["similarity"]:.4f}
Content:
{chunk["content"]}
""".strip()
        )

    return "\n\n".join(context_blocks)


def answer_question(
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    retrieved_chunks = search_documents(
        query=question,
        match_count=top_k,
        similarity_threshold=similarity_threshold,
    )

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "I don't know. I could not find relevant context in the knowledge base.",
            "sources": [],
        }

    context = format_context(retrieved_chunks)

    prompt = f"""
You are a grounded RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
- If the context contains the answer, answer clearly and concisely.
- If the context does not contain the answer, say: "I don't know based on the provided context."
- Do not use outside knowledge.
- Cite sources inline using [Source 1], [Source 2], etc.
- Do not mention similarity scores in the answer.

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
        "sources": retrieved_chunks,
    }


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
        print("Title:", source["title"])
        print("Similarity:", round(source["similarity"], 4))
        print("Content preview:", source["content"][:300].replace("\n", " "))


if __name__ == "__main__":
    questions = [
        "What is photosynthesis?",
        "Who was Albert Einstein?",
        "What caused World War II?",
        "How do volcanoes form?",
        "What is gravity?",
    ]

    for q in questions:
        result = answer_question(q)
        print_result(result)