import argparse
import json
import os
import time
import statistics
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT_PATH = "benchmark_results/context_quality_judge_fixed_overlap_hnsw_reranked_top10.json"


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


SYSTEM_PROMPT = """
You are an expert evaluator for RAG retrieval quality.

Your job is to judge whether the retrieved context is useful for answering the user's query.

Important rules:
- Judge ONLY the provided retrieved context.
- Do NOT use outside knowledge.
- Do NOT judge the final answer, because no final answer is provided.
- Penalize irrelevant, noisy, duplicated, fragmented, or contradictory context.
- Reward context that is relevant, complete, coherent, and sufficient to answer the query.
- Return strict JSON only.
""".strip()


USER_PROMPT_TEMPLATE = """
User query:
{query}

Retrieved context:
{context}

Evaluate the retrieved context quality.

Scoring rubric:

answerability:
- "yes": context contains enough information to answer the query well
- "partial": context contains some useful information but misses important pieces
- "no": context does not contain enough useful information

context_relevance_score:
1 = mostly irrelevant
2 = weakly relevant
3 = partially relevant
4 = mostly relevant
5 = directly relevant

context_completeness_score:
1 = cannot answer the query
2 = major missing information
3 = partial answer possible
4 = mostly complete
5 = complete enough to answer well

context_coherence_score:
1 = fragmented / hard to follow
2 = choppy with major discontinuities
3 = understandable but somewhat fragmented
4 = mostly coherent
5 = coherent and easy to use

noise_score:
1 = very little noise
2 = small amount of noise
3 = moderate noise
4 = high noise
5 = mostly noisy / distracting

follow the below failure_type rules only to define failure_type:

none:
Use when answerability is "yes".

corpus_gap:
Use when retrieved context is on-topic, but the source material appears too shallow or missing the specific information needed.

retrieval_miss:
Use when retrieved context is mostly about the wrong topic or wrong entity.

fragmented_context:
Use when relevant information appears present, but chunk boundaries/fragments make the context hard to use.

noisy_context:
Use when relevant information is present but buried under too much irrelevant content.

insufficient_detail:
Use when context is relevant and coherent, but answerability is partial.

ambiguous_query:
Use when the user query is too vague to know what information is required.

Return strict JSON with this schema:
{{
  "answerability": "yes" | "partial" | "no",
  "context_relevance_score": 1,
  "context_completeness_score": 1,
  "context_coherence_score": 1,
  "noise_score": 1,
  "missing_information": "short explanation",
  "failure_type": "none" | "corpus_gap" | "retrieval_miss" | "fragmented_context" | "noisy_context" | "insufficient_detail" | "ambiguous_query",
  "reason": "short explanation"
}}
""".strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Handles occasional markdown-wrapped JSON.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return json.loads(text)


def build_context(reranked_results: List[Dict[str, Any]], max_chunks: int = 10) -> str:
    parts = []

    for idx, result in enumerate(reranked_results[:max_chunks], start=1):
        title = result.get("title", "")
        content = result.get("content", "")
        chunk_id = result.get("chunk_id", "")
        similarity = result.get("similarity")
        rerank_score = result.get("rerank_score")
        original_rank = result.get("original_rank")

        header = (
            f"[Chunk {idx}]\n"
            f"Title: {title}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Original retrieval rank: {original_rank}\n"
            f"Embedding similarity: {similarity}\n"
            f"Rerank score: {rerank_score}\n"
            f"Content:\n{content}"
        )

        parts.append(header)

    return "\n\n---\n\n".join(parts)


def judge_context_quality(
        query: str,
        context: str,
        model: str,
        max_retries: int = 4,
        sleep_base_seconds: float = 2.0,
) -> Dict[str, Any]:
    prompt = USER_PROMPT_TEMPLATE.format(
        query=query,
        context=context,
    )

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

            raw = response.choices[0].message.content or "{}"
            parsed = safe_json_loads(raw)

            usage = response.usage
            parsed["_judge_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

            return parsed

        except Exception as e:
            last_error = e
            sleep_seconds = sleep_base_seconds * attempt
            print(
                f"Judge call failed attempt={attempt}/{max_retries} "
                f"error={repr(e)} sleeping={sleep_seconds}s"
            )
            time.sleep(sleep_seconds)

    return {
        "answerability": "error",
        "context_relevance_score": None,
        "context_completeness_score": None,
        "context_coherence_score": None,
        "noise_score": None,
        "missing_information": "",
        "reason": f"Judge failed after retries: {repr(last_error)}",
        "_judge_usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }


def mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def percentile_nearest_rank(values: List[float], percentile: float) -> Optional[float]:
    values = sorted([v for v in values if v is not None])
    if not values:
        return None

    import math

    rank = math.ceil(percentile * len(values))
    rank = max(1, min(rank, len(values)))
    return values[rank - 1]


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    judgments = [r.get("judgment", {}) for r in results]

    relevance = [j.get("context_relevance_score") for j in judgments]
    completeness = [j.get("context_completeness_score") for j in judgments]
    coherence = [j.get("context_coherence_score") for j in judgments]
    noise = [j.get("noise_score") for j in judgments]

    answerability_counts = {
        "yes": 0,
        "partial": 0,
        "no": 0,
        "error": 0,
    }

    failure_type_counts = {
        "none": 0,
        "corpus_gap": 0,
        "retrieval_miss": 0,
        "fragmented_context": 0,
        "noisy_context" :0,
        "insufficient_detail": 0,
        "ambiguous_query": 0,
        "error": 0
    }

    for judgment in judgments:
        value = judgment.get("answerability", "error")
        if value not in answerability_counts:
            value = "error"
        answerability_counts[value] += 1

        failure_value = judgment.get("failure_type", "error")
        if failure_value not in failure_type_counts:
            failure_value = "error"
        failure_type_counts[failure_value] +=1

    total = len(results)

    def pct(count: int) -> float:
        return count / total if total else 0.0

    total_tokens = [
        (j.get("_judge_usage") or {}).get("total_tokens")
        for j in judgments
        if (j.get("_judge_usage") or {}).get("total_tokens") is not None
    ]

    return {
        "count": total,

        "answerability_counts": answerability_counts,
        "answerability_pct": {
            key: pct(value)
            for key, value in answerability_counts.items()
        },

        "context_relevance": {
            "avg": mean(relevance),
            "median": statistics.median([v for v in relevance if v is not None]) if any(v is not None for v in relevance) else None,
            "p10": percentile_nearest_rank(relevance, 0.10),
            "min": min([v for v in relevance if v is not None], default=None),
        },

        "context_completeness": {
            "avg": mean(completeness),
            "median": statistics.median([v for v in completeness if v is not None]) if any(v is not None for v in completeness) else None,
            "p10": percentile_nearest_rank(completeness, 0.10),
            "min": min([v for v in completeness if v is not None], default=None),
        },

        "context_coherence": {
            "avg": mean(coherence),
            "median": statistics.median([v for v in coherence if v is not None]) if any(v is not None for v in coherence) else None,
            "p10": percentile_nearest_rank(coherence, 0.10),
            "min": min([v for v in coherence if v is not None], default=None),
        },

        "noise": {
            "avg": mean(noise),
            "median": statistics.median([v for v in noise if v is not None]) if any(v is not None for v in noise) else None,
            "p90": percentile_nearest_rank(noise, 0.90),
            "max": max([v for v in noise if v is not None], default=None),
        },

        "token_usage": {
            "avg_total_tokens": mean(total_tokens),
            "total_tokens": sum(total_tokens) if total_tokens else 0,
        },

        "failure_type_counts": failure_type_counts,
        "failure_type_pct": {
            key: pct(value)
            for key, value in failure_type_counts.items()
        },
    }


def load_existing_results(output_path: str) -> Dict[str, Any]:
    if not os.path.exists(output_path):
        return {
            "metadata": {},
            "summary": {},
            "results": [],
        }

    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to reranked retrieval results JSON.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI judge model.",
    )


    parser.add_argument(
        "--strategy",
        default="fixed_overlap_v1",
        help="Chunking strategy label for metadata.",
    )

    parser.add_argument(
        "--retrieval-path",
        default="benchmark_results_ef_search_120_hnsw",
        help="Retrieval path label for metadata.",
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file and skip already judged queries.",
    )

    args = parser.parse_args()

    output_path = f"benchmark_results/context_quality_judge_{args.strategy}_hnsw_reranked_top10.json"

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    queries = input_data.get("queries", [])

    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    existing = load_existing_results(output_path) if args.resume else {
        "metadata": {},
        "summary": {},
        "results": [],
    }

    existing_by_query = {
        item.get("query"): item
        for item in existing.get("results", [])
        if item.get("query")
    }

    results = list(existing.get("results", [])) if args.resume else []

    for idx, item in enumerate(queries, start=1):
        query = item.get("query", "")

        if args.resume and query in existing_by_query:
            print(f"[{idx}/{len(queries)}] Skipping already judged query: {query}")
            continue

        reranked_results = item.get("reranked_results", [])

        context = build_context(
            reranked_results=reranked_results,
            max_chunks=10,
        )

        print(f"[{idx}/{len(queries)}] Judging query: {query}")

        started = time.perf_counter()

        judgment = judge_context_quality(
            query=query,
            context=context,
            model=args.model,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

        result = {
            "query": query,
            "candidate_count": item.get("candidate_count"),
            "rerank_top_k": item.get("rerank_top_k"),
            "rerank_latency_ms": item.get("rerank_latency_ms"),
            "judge_latency_ms": elapsed_ms,
            "top_titles": [
                result.get("title")
                for result in reranked_results[:10]
            ],
            "top_chunk_ids": [
                result.get("chunk_id")
                for result in reranked_results[:10]
            ],
            "judgment": judgment,
        }

        results.append(result)

        report = {
            "metadata": {
                "input_path": args.input,
                "judge_model": args.model,
                "chunking_strategy": args.strategy,
                "retrieval_path": args.retrieval_path,
                "num_queries": len(queries),
            },
            "summary": summarize(results),
            "results": results,
        }


        save_json(output_path, report)

    final_report = {
        "metadata": {
            "input_path": args.input,
            "judge_model": args.model,
            "chunking_strategy": args.strategy,
            "retrieval_path": args.retrieval_path,
            "num_queries": len(queries),
        },
        "summary": summarize(results),
        "results": results,
    }

    save_json(output_path, final_report)

    print("Saved judge results:")
    print(output_path)

    print("Summary:")
    print(json.dumps(final_report["summary"], indent=2))


if __name__ == "__main__":
    main()