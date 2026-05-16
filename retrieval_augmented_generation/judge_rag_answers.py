import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_INPUT_PATH = "benchmark_results/answers_context_aware_ef_search_120_hnsw.json"
DEFAULT_OUTPUT_PATH = "benchmark_results/rag_answer_judge_context_aware_ef_search_120_hnsw.json"
DEFAULT_MAX_CHUNKS = 10

# Current GPT-4.1 mini API pricing from the OpenAI model docs:
# https://platform.openai.com/docs/models/gpt-4.1-mini
DEFAULT_INPUT_COST_PER_1M = 0.40
DEFAULT_OUTPUT_COST_PER_1M = 1.60

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


SYSTEM_PROMPT = """
You are an expert evaluator for a retrieval-augmented generation (RAG) pipeline.

Your job is to judge the final answer using only the user query, retrieved context,
and generated answer provided to you.

Rules:
- Do not use outside knowledge.
- Treat the retrieved context as the only allowed source of truth.
- Penalize unsupported claims, contradictions, missing caveats, vague answers,
  and answers that cite sources that do not support the claims.
- Evaluate retrieval quality separately from answer quality.
- Return strict JSON only.
""".strip()


USER_PROMPT_TEMPLATE = """
User query:
{query}

Generated answer:
{answer}

Retrieved context:
{context}

Evaluate these RAG pipeline metrics.

Metric definitions:

answerability:
- Measures final usefulness of the generated answer for the user query.
- Consider whether the answer is direct, clear, and usable.

groundedness:
- Measures hallucination risk.
- Reward answers whose claims are supported by the retrieved context.
- Penalize unsupported facts, overstatements, and contradictions.

completeness:
- Measures whether the generated answer covers the important parts of the query.
- Penalize answers that are correct but only partial.

context_relevance:
- Measures retrieval quality.
- Reward context that is topically focused on the query.
- Penalize irrelevant, noisy, or wrong-entity chunks.

context_completeness:
- Measures chunking and retrieval sufficiency.
- Reward context that contains enough information to fully answer the query.
- Penalize missing key facts or fragmented context that makes the answer hard.

Use this score scale for all numeric metrics:
1 = poor
2 = weak
3 = acceptable / partial
4 = good
5 = excellent

Return strict JSON with this schema:
{{
  "answerability": "yes" | "partial" | "no",
  "answerability_score": 1,
  "groundedness_score": 1,
  "completeness_score": 1,
  "context_relevance_score": 1,
  "context_completeness_score": 1,
  "unsupported_claims": ["short claim not supported by context"],
  "missing_information": ["short missing point"],
  "failure_type": "none" | "unanswerable_from_context" | "retrieval_miss" | "fragmented_context" | "noisy_context" | "incomplete_answer" | "hallucinated_answer" | "ambiguous_query",
  "reason": "short explanation"
}}
""".strip()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    script_relative = Path(__file__).resolve().parent / path
    if script_relative.exists():
        return script_relative

    return script_relative if not candidate.is_absolute() else candidate


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return json.loads(text)


def save_json(path: str, data: Dict[str, Any]) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_json(path: str) -> Dict[str, Any]:
    input_path = resolve_path(path)
    with input_path.open(encoding="utf-8") as f:
        return json.load(f)


def build_context(chunks: List[Dict[str, Any]], max_chunks: int) -> str:
    blocks = []

    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
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
        blocks.append(
            f"""
[Source {idx}]
Title: {chunk.get("title", "Untitled")}
Chunk ID: {chunk.get("chunk_id", "")}
{similarity_line}{rerank_line}Content:
{chunk.get("content", "")}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


def compute_llm_cost_usd(
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> Optional[float]:
    if prompt_tokens is None or completion_tokens is None:
        return None

    input_cost = (prompt_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (completion_tokens / 1_000_000) * output_cost_per_1m
    return input_cost + output_cost


def judge_answer(
    query: str,
    answer: str,
    context: str,
    model: str,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
    max_retries: int = 4,
    sleep_base_seconds: float = 2.0,
) -> Dict[str, Any]:
    prompt = USER_PROMPT_TEMPLATE.format(
        query=query,
        answer=answer,
        context=context,
    )

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            latency_ms = (time.perf_counter() - started) * 1000

            raw = response.choices[0].message.content or "{}"
            parsed = safe_json_loads(raw)

            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            judge_cost_usd = compute_llm_cost_usd(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_cost_per_1m=input_cost_per_1m,
                output_cost_per_1m=output_cost_per_1m,
            )

            parsed["_judge_metadata"] = {
                "model": model,
                "latency_ms": latency_ms,
                "cost_usd": judge_cost_usd,
                "input_cost_per_1m": input_cost_per_1m,
                "output_cost_per_1m": output_cost_per_1m,
            }
            parsed["_judge_usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
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
        "answerability_score": None,
        "groundedness_score": None,
        "completeness_score": None,
        "context_relevance_score": None,
        "context_completeness_score": None,
        "unsupported_claims": [],
        "missing_information": [],
        "failure_type": "error",
        "reason": f"Judge failed after retries: {repr(last_error)}",
        "_judge_metadata": {
            "model": model,
            "latency_ms": None,
            "cost_usd": None,
            "input_cost_per_1m": input_cost_per_1m,
            "output_cost_per_1m": output_cost_per_1m,
        },
        "_judge_usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }


def mean(values: List[Optional[float]]) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def percentile_nearest_rank(
    values: List[Optional[float]],
    percentile: float,
) -> Optional[float]:
    filtered = sorted(v for v in values if v is not None)
    if not filtered:
        return None

    import math

    rank = math.ceil(percentile * len(filtered))
    rank = max(1, min(rank, len(filtered)))
    return filtered[rank - 1]


def summarize_metric(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    filtered = [v for v in values if v is not None]
    return {
        "avg": mean(values),
        "median": statistics.median(filtered) if filtered else None,
        "p10": percentile_nearest_rank(values, 0.10),
        "p90": percentile_nearest_rank(values, 0.90),
        "min": min(filtered, default=None),
        "max": max(filtered, default=None),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    judgments = [r.get("judgment", {}) for r in results]

    answerability_counts = {
        "yes": 0,
        "partial": 0,
        "no": 0,
        "error": 0,
    }
    failure_type_counts: Dict[str, int] = {}

    for judgment in judgments:
        answerability = judgment.get("answerability", "error")
        if answerability not in answerability_counts:
            answerability = "error"
        answerability_counts[answerability] += 1

        failure_type = judgment.get("failure_type", "error")
        failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1

    total = len(results)

    def pct(count: int) -> float:
        return count / total if total else 0.0

    judge_latencies = [
        (j.get("_judge_metadata") or {}).get("latency_ms")
        for j in judgments
    ]
    judge_costs = [
        (j.get("_judge_metadata") or {}).get("cost_usd")
        for j in judgments
    ]
    prompt_tokens = [
        (j.get("_judge_usage") or {}).get("prompt_tokens")
        for j in judgments
    ]
    completion_tokens = [
        (j.get("_judge_usage") or {}).get("completion_tokens")
        for j in judgments
    ]

    return {
        "count": total,
        "answerability_counts": answerability_counts,
        "answerability_pct": {
            key: pct(value)
            for key, value in answerability_counts.items()
        },
        "answerability": summarize_metric([
            j.get("answerability_score") for j in judgments
        ]),
        "groundedness": summarize_metric([
            j.get("groundedness_score") for j in judgments
        ]),
        "completeness": summarize_metric([
            j.get("completeness_score") for j in judgments
        ]),
        "context_relevance": summarize_metric([
            j.get("context_relevance_score") for j in judgments
        ]),
        "context_completeness": summarize_metric([
            j.get("context_completeness_score") for j in judgments
        ]),
        "judge_latency_ms": summarize_metric(judge_latencies),
        "judge_cost_per_query_usd": summarize_metric(judge_costs),
        "judge_cost_total_usd": sum(v for v in judge_costs if v is not None),
        "judge_token_usage": {
            "avg_prompt_tokens": mean(prompt_tokens),
            "avg_completion_tokens": mean(completion_tokens),
            "total_prompt_tokens": sum(v for v in prompt_tokens if v is not None),
            "total_completion_tokens": sum(v for v in completion_tokens if v is not None),
        },
        "failure_type_counts": failure_type_counts,
        "failure_type_pct": {
            key: pct(value)
            for key, value in failure_type_counts.items()
        },
    }


def load_existing_results(output_path: str) -> Dict[str, Any]:
    path = resolve_path(output_path)
    if not path.exists():
        return {
            "metadata": {},
            "summary": {},
            "results": [],
        }

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_answer(item: Dict[str, Any]) -> str:
    answer_response = item.get("answer_response") or {}
    return answer_response.get("answer") or item.get("answer") or ""


def evaluate_answers(
    input_path: str,
    output_path: str,
    model: str,
    max_chunks: int,
    max_queries: Optional[int],
    resume: bool,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> Dict[str, Any]:
    input_data = load_json(input_path)
    queries = input_data.get("queries", [])

    if max_queries is not None:
        queries = queries[:max_queries]

    existing = load_existing_results(output_path) if resume else {
        "metadata": {},
        "summary": {},
        "results": [],
    }
    existing_by_query = {
        item.get("query"): item
        for item in existing.get("results", [])
        if item.get("query")
    }

    results = list(existing.get("results", [])) if resume else []

    for idx, item in enumerate(queries, start=1):
        query = item.get("query", "")

        if resume and query in existing_by_query:
            print(f"[{idx}/{len(queries)}] Skipping already judged query: {query}")
            continue

        answer = extract_answer(item)
        chunks = item.get("chunks", [])
        context = build_context(chunks=chunks, max_chunks=max_chunks)

        print(f"[{idx}/{len(queries)}] Judging answer: {query}")

        judgment = judge_answer(
            query=query,
            answer=answer,
            context=context,
            model=model,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )

        result = {
            "query": query,
            "answer_model": (item.get("answer_response") or {}).get("model"),
            "answer": answer,
            "source_chunk_ids": item.get("source_chunk_ids", []),
            "source_titles": [
                chunk.get("title")
                for chunk in chunks[:max_chunks]
            ],
            "source_count": len(chunks[:max_chunks]),
            "judgment": judgment,
            "note": (
                "Latency and cost/query are measured for the judge call. "
                "Answer-generation latency/cost are not in the input answer file."
            ),
        }
        results.append(result)

        report = {
            "metadata": {
                "input_path": str(resolve_path(input_path)),
                "input_metadata": input_data.get("metadata", {}),
                "judge_model": model,
                "max_chunks": max_chunks,
                "num_queries": len(queries),
                "input_cost_per_1m": input_cost_per_1m,
                "output_cost_per_1m": output_cost_per_1m,
            },
            "summary": summarize(results),
            "results": results,
        }
        save_json(output_path, report)

    final_report = {
        "metadata": {
            "input_path": str(resolve_path(input_path)),
            "input_metadata": input_data.get("metadata", {}),
            "judge_model": model,
            "max_chunks": max_chunks,
            "num_queries": len(queries),
            "input_cost_per_1m": input_cost_per_1m,
            "output_cost_per_1m": output_cost_per_1m,
        },
        "summary": summarize(results),
        "results": results,
    }
    save_json(output_path, final_report)
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Judge generated RAG answers using query, answer, and retrieved "
            "chunks from an answer JSON file."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--input-cost-per-1m",
        type=float,
        default=DEFAULT_INPUT_COST_PER_1M,
        help="Judge model input token cost per 1M tokens.",
    )
    parser.add_argument(
        "--output-cost-per-1m",
        type=float,
        default=DEFAULT_OUTPUT_COST_PER_1M,
        help="Judge model output token cost per 1M tokens.",
    )

    args = parser.parse_args()

    report = evaluate_answers(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        max_chunks=args.max_chunks,
        max_queries=args.max_queries,
        resume=args.resume,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )

    print("Saved RAG answer judge results:")
    print(resolve_path(args.output))
    print("Summary:")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
