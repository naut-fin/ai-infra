# RAG Project Recap and Q&A

This document is a closing recap for the Retrieval Augmented Generation experiments in this folder. It is written as a reference for a future reader who wants to understand what was built, why it was built, what the results mean, and what questions to ask before extending the project.

## Executive Summary

This project built an end-to-end RAG evaluation pipeline over Simple Wikipedia using Supabase Postgres, pgvector, OpenAI embeddings, Cohere reranking, OpenAI-based judges, and JSON benchmark artifacts.

The final core pipeline is:

1. Ingest Wikipedia articles into `documents` and `document_chunks`.
2. Embed chunks with `text-embedding-3-small`.
3. Retrieve candidates with pgvector HNSW at `ef_search=120`.
4. Rerank retrieved candidates with Cohere `rerank-v3.5`.
5. Generate grounded answers with `gpt-4.1-mini`.
6. Judge both retrieved context quality and final answer quality with `gpt-4.1-mini`.
7. Compare context judge and answer judge outcomes to detect questionable cases.

The most important final result:

- Context-aware chunking improved context answerability over fixed-overlap chunking.
- Final answer judging reported 137 yes, 15 partial, and 46 no answers over 198 queries.
- The final context-vs-answer comparison found 24 cases where context was only partial but the answer judge still marked the answer as yes.
- There were 0 cases where context was no and answer was yes.
- There were 0 cases where context was yes and answer was no.

## Key Final Metrics

### Context-aware HNSW plus rerank context judge

Source file:

`benchmark_results/context_quality_judge_context_aware_chunk_v1_hnsw_reranked_top10.json`

Results over 198 queries:

- Context answerability yes: 113, or 57.1 percent
- Context answerability partial: 44, or 22.2 percent
- Context answerability no: 41, or 20.7 percent
- Average context relevance: 3.80
- Average context completeness: 3.59
- Average context coherence: 4.86
- Average noise score: 1.60
- Main failure types: 42 retrieval misses and 43 insufficient-detail cases

### Fixed-overlap HNSW plus rerank context judge

Source file:

`benchmark_results/context_quality_judge_fixed_overlap_v1_hnsw_reranked_top10.json`

Results over 198 queries:

- Context answerability yes: 93, or 47.0 percent
- Context answerability partial: 59, or 29.8 percent
- Context answerability no: 46, or 23.2 percent
- Average context relevance: 3.67
- Average context completeness: 3.31
- Average context coherence: 4.40
- Average noise score: 2.06

Conclusion: context-aware chunking produced more answerable, more coherent, and less noisy contexts than fixed overlap.

### Neighbor-expanded context-aware retrieval

Source file:

`benchmark_results/context_quality_judge_context_aware_chunk_neighbor_expanded_v1_hnsw_reranked_top10.json`

Results over 198 queries:

- Context answerability yes: 115, or 58.1 percent
- Context answerability partial: 43, or 21.7 percent
- Context answerability no: 40, or 20.2 percent

Conclusion: neighbor expansion gave a small improvement over plain context-aware HNSW, but not a dramatic one. It slightly increased noise and only shifted a few queries.

### Final answer judge

Source file:

`benchmark_results/rag_answer_judge_context_aware_ef_search_120_hnsw.json`

Results over 198 queries:

- Answerability yes: 137, or 69.2 percent
- Answerability partial: 15, or 7.6 percent
- Answerability no: 46, or 23.2 percent
- Average answerability score: 3.87
- Average groundedness score: 4.49
- Average completeness score: 3.61
- Average judged context relevance: 4.14
- Average judged context completeness: 3.66
- Judge cost total: about 0.304 USD for 198 answer-judge calls

Failure type counts:

- none: 100
- incomplete_answer: 47
- unanswerable_from_context: 48
- retrieval_miss: 3

Conclusion: answers were usually grounded when retrieval worked. The dominant weakness was still upstream retrieval or incomplete context, not uncontrolled hallucination.

### Context judge vs answer judge comparison

Source file:

`benchmark_results/context_vs_answer_judge_comparison_context_aware_hnsw.json`

Results over 198 matched queries:

- A. context partial -> answer yes: 24
- B. context no -> answer yes: 0
- C. context yes -> answer no: 0
- Questionable context not yes but answer yes: 24

Transition matrix:

```json
{
  "yes": {
    "yes": 113,
    "partial": 0,
    "no": 0,
    "error": 0,
    "missing": 0
  },
  "partial": {
    "yes": 24,
    "partial": 11,
    "no": 9,
    "error": 0,
    "missing": 0
  },
  "no": {
    "yes": 0,
    "partial": 4,
    "no": 37,
    "error": 0,
    "missing": 0
  }
}
```

Conclusion: the most suspicious category is not direct no-context leakage. It is partial-context cases where the answer model or answer judge considered the final answer good enough. These should be manually reviewed before making strong claims about leakage.

## Architecture Questions and Answers

### Q: What is the project trying to prove?

A: It is trying to evaluate a practical RAG stack, not just build a demo. The core question is: given a fixed query set, corpus, embedding model, vector index, reranker, and answer model, how often does the system retrieve enough context and produce grounded answers?

The project separates evaluation into several layers:

- Retrieval quality: did HNSW retrieve candidates close to exact flat search?
- Reranking quality: did the reranker improve the final top results?
- Context quality: is the retrieved top context enough to answer?
- Answer quality: did the final model answer clearly and stay grounded?
- Judge agreement: do context and answer judges agree, and where do they disagree?

### Q: What corpus did we use?

A: The ingestion scripts use the Hugging Face Wikipedia dataset `wikipedia`, config `20220301.simple`, which is Simple English Wikipedia. The main evaluation query set contains 198 broad educational and technical questions from `questions.py`.

Simple Wikipedia is useful because it is manageable and easy to inspect, but it creates known corpus gaps for modern infrastructure topics such as RAG, vector databases, Kubernetes, Docker, Hadoop, Spark, and some systems concepts.

### Q: Why did some modern technical questions fail?

A: Many of those topics are either missing from Simple Wikipedia or represented by ambiguous words. For example:

- `Spark` may retrieve electrical sparks instead of Apache Spark.
- `RAG` may retrieve unrelated acronym meanings instead of Retrieval Augmented Generation.
- `Vector database` may retrieve general vector or database pages but not the concept of a vector DB.
- `Kubernetes`, `Docker`, `Hadoop`, and similar terms may not be covered well in the corpus.

This is not only a retrieval problem. It is partly a corpus coverage problem.

### Q: What are the main database tables?

A: The SQL schema uses two main tables:

- `documents`: stores article-level metadata, title, source URL, and creation time.
- `document_chunks`: stores chunk text, token counts, metadata, embeddings, and chunk indexes.

The important SQL files are:

- `sql/documents.sql`: `documents` table shape.
- `sql/document_chunks.sql`: `document_chunks` table shape.
- `sql/match_document_chunks.sql`: vector search RPC with runtime `hnsw.ef_search`.
- `sql/match_documents_hnsw.sql`: older HNSW search RPC shape.
- `sql/hybrid_search  BM25 + Semantic Searcg/*.sql`: lexical search columns, trigger, indexes, backfill, and RPC.

### Q: What does HNSW do in this project?

A: HNSW is the approximate nearest neighbor index used by pgvector to speed up vector search. Instead of scanning all embeddings exactly, it searches an approximate graph. The tradeoff is speed versus recall.

The runtime parameter `ef_search` controls how much search effort HNSW spends per query:

- Higher `ef_search`: better recall, higher latency.
- Lower `ef_search`: lower latency, more missed candidates.

The project settled on `ef_search=120` as the main default for later experiments.

### Q: Why compare flat search to HNSW?

A: Flat search is treated as an exact baseline. HNSW is approximate, so it needs to be compared against flat search to measure whether it misses important nearest neighbors.

The comparison scripts compute recall and NDCG against flat results. This gives a practical signal for whether HNSW approximation is harming retrieval quality.

### Q: What were the HNSW latency numbers for the final context-aware run?

A: For `benchmark_results_context_aware_ef_search_120_hnsw.json`:

- Retrieval p50 across queries: 97.34 ms
- Retrieval p95 across queries: 328.71 ms
- Retrieval average across queries: 140.27 ms

These numbers measure retrieval against Supabase/Postgres, not full end-to-end answer generation.

### Q: Why did we add reranking?

A: Vector retrieval returns a candidate set, but the raw vector order is not always the best final context order. Cohere Rerank was added to reorder retrieved chunks by query relevance.

The main reranker script is `rerank_retrieval_results.py`. It takes retrieved candidates, builds compact reranker documents with title and content, sends them to Cohere `rerank-v3.5`, and stores the final `reranked_results`.

For the final context-aware HNSW run:

- Candidate limit: 50
- Rerank top-k: 10
- Average rerank latency: 163.82 ms
- Rerank p50: 147.15 ms
- Rerank p95: 208.49 ms

### Q: What is context-aware chunking?

A: Context-aware chunking tries to split documents at semantic boundaries instead of blindly cutting every 256 tokens. The implementation in `ingest_wikipedia_context_aware_chunk_overlap.py` does several things:

- Detects likely section headings.
- Ignores low-value sections such as references and external links.
- Splits prose into paragraphs and sentences.
- Handles timeline-like and list-like pages differently.
- Adds title and section path into the embedding text.
- Repairs orphan or fragment chunks.
- Avoids storing repeated overlap inside every chunk.

The goal was to reduce mid-sentence cuts, duplicate overlap, and fragmented context.

### Q: Why not just use fixed token overlap?

A: Fixed token overlap is simple, but it creates several problems:

- Many chunks begin or end mid-sentence.
- Overlap duplicates text across neighboring chunks.
- Retrieved context can become repetitive or noisy.
- The LLM may see fragments without enough local meaning.

The chunk-quality reports illustrate this:

Fixed overlap:

- 93.3 percent of chunks ended mid-sentence.
- 50.0 percent had fragment-like starts.
- Average adjacent token Jaccard overlap was 0.198.

Context-aware:

- 24.2 percent of chunks ended mid-sentence.
- 7.5 percent had fragment-like starts.
- Average adjacent token Jaccard overlap was 0.140.

Context-aware chunking was not perfect, but it reduced fragmentation substantially.

### Q: Why did we add neighbor expansion?

A: Context-aware chunks avoid duplicated overlap, but that means a retrieved chunk may lack the previous or next sentence needed for continuity. Neighbor expansion fetches adjacent chunks at query time, usually `N-1`, `N`, and `N+1`, and builds a larger passage for reranking and answer generation.

The implementation is in `benchmark_retrieval_with_neighbors.py`.

The result was a small context answerability improvement:

- Plain context-aware: 113 yes
- Neighbor-expanded context-aware: 115 yes

The gain was small enough that the simpler plain context-aware path remained reasonable.

### Q: What is hybrid retrieval?

A: Hybrid retrieval combines semantic vector search with lexical full-text search. The implementation is in `benchmark_hybrid_retrieval.py` and the SQL files under `sql/hybrid_search  BM25 + Semantic Searcg/`.

The hybrid path:

1. Runs semantic HNSW search.
2. Runs Postgres full-text lexical search.
3. Merges candidates by chunk ID.
4. Tracks whether each candidate came from semantic, lexical, or both.
5. Sends merged candidates to a reranker.

For the top 50 semantic plus top 50 lexical run:

- Average hybrid candidate count: 61.84
- Average semantic and lexical overlap: 6.13
- Retrieval total p50: 161.59 ms
- Retrieval total p95: 200.56 ms

Reranked top 10 source breakdown:

- Average semantic-only in top-k: 6.80
- Average lexical-only in top-k: 0.67
- Average both-source in top-k: 2.53

Interpretation: lexical retrieval added some useful candidates, but semantic retrieval still dominated most final top-k contexts.

### Q: Why add routed retrieval?

A: Always running hybrid retrieval costs more than semantic-only retrieval. Routed retrieval tries to run hybrid only when a query has lexical signals such as acronyms, years, quoted phrases, numbers, code-like terms, or rare exact terms.

The implementation is in `route_retrieval.py`, using the evaluation set in `routing_eval_queries.py`.

Final routed retrieval summary:

- Query count: 50
- Routed hybrid: 21
- Routed semantic: 29
- Accuracy against hand labels: 92 percent
- Average latency: 412.31 ms
- Average candidate count: 52.18

This is a good prototype, but it needs more evaluation before production use.

## File-by-File Guide

### `README.md`

The operational command guide. It explains environment variables, ingestion, flat versus HNSW experiments, `ef_search` tuning, reranking, context judging, and chunking comparisons.

### `questions.py`

Defines `EVAL_QUERIES`, the 198-query benchmark set. These queries intentionally include science, history, economics, computing, distributed systems, databases, vector search, and RAG topics.

### `ingest_wikipedia_fixed_chunk_overlap.py`

Ingests Simple Wikipedia with fixed 256-token chunks and 24-token overlap. This was the baseline chunking strategy.

### `ingest_wikipedia_context_aware_chunk_overlap.py`

Ingests Simple Wikipedia with section-aware and sentence-aware logic. It detects headings, article types, lists, timelines, and orphan chunks. It embeds enriched text containing title, section path, previous context, and content.

This is the most important ingestion script.

### `analyze_chunk_quality.py`

Reads chunks from Supabase and computes chunk hygiene metrics:

- token size distribution
- short chunks
- lowercase starts
- punctuation starts
- mid-sentence endings
- fragment-like starts
- short-label chunks
- duplicate ratio
- adjacent chunk overlap

This script was used to confirm that context-aware chunking materially improved chunk quality.

### `benchmark_retrieval.py`

Runs flat and HNSW vector retrieval benchmarks over `EVAL_QUERIES`. It records embedding latency, retrieval latency summaries, top results, and result IDs. It also compares HNSW to flat using recall and NDCG.

Important caveat: true flat search requires dropping the HNSW index before the flat run. Otherwise Postgres may still use the index.

### `benchmark_retrieval_with_neighbors.py`

Runs HNSW retrieval and expands each seed result with neighboring chunks from the same document. This addresses continuity loss from non-overlapping context-aware chunks.

### `benchmark_hybrid_retrieval.py`

Runs semantic HNSW plus lexical full-text search, merges and dedupes candidates, tracks semantic and lexical sources, and writes a candidate set that can be consumed by the reranker.

### `rerank_retrieval_results.py`

Reranks normal retrieval outputs with Cohere Rerank. It supports flat/HNSW reranking, custom input/output files, candidate limits, and comparison of reranked HNSW against reranked flat outputs.

### `rerank_hybrid_retrieval_results.py`

Reranks hybrid retrieval outputs and tracks how many candidates in the final top-k came from semantic-only, lexical-only, or both retrieval sources.

### `judge_context_quality.py`

Uses an OpenAI judge model to evaluate retrieved context alone. It does not evaluate the generated answer. It scores:

- answerability
- relevance
- completeness
- coherence
- noise
- failure type

This script answers: did retrieval give the model enough useful context?

### `answer_question.py`

Generates actual RAG answers from a reranked results file. It asks the answer model to use only the provided context and to say `I don't know based on the provided context.` when context is insufficient.

### `judge_rag_answers.py`

Judges generated answers using the query, answer, and retrieved context. It scores:

- answerability
- groundedness
- completeness
- context relevance
- context completeness
- unsupported claims
- missing information
- failure type

This script answers: did the final RAG response work?

### `compare_context_answer_judges.py`

Compares per-query context judge output against final answer judge output. This was added to catch potentially suspicious cases where the context judge says the context is not fully answerable, but the answer judge says the final answer is answerable.

The key suspicious condition is:

`context_answerability != yes` and `answer_answerability == yes`

The final count was 24, all from context partial to answer yes. There were no context no to answer yes cases.

### `analyze_retrieval_failures.py`

Prints and stores worst flat-vs-HNSW failures by recall. Useful for manual inspection of cases where HNSW diverges from flat search.

### `classify_retrieval_failures.py`

Applies heuristics to classify flat-vs-HNSW failures into:

- healthy
- corpus_gap
- semantic_ambiguity
- ann_miss
- minor_ranking_difference
- uncertain

For the classified recall@10 artifact, 82.32 percent were healthy, 13.64 percent were semantic ambiguity, 2.02 percent were corpus gaps, and only 0.51 percent were labeled ANN miss.

### `plot_ef_search_benchmarks.py`

Builds SVG plots from ef_search benchmark result triplets. It visualizes recall and latency tradeoffs.

### `route_retrieval.py`

Routes queries to semantic-only or hybrid retrieval based on cheap lexical signals. It imports existing benchmark functions and emits a common candidate format for reranking.

### `routing_eval_queries.py`

Contains 50 labeled routing evaluation queries. Half are exact-match-heavy hybrid candidates; half are broad semantic queries.

### `sample_rag/rag_supabase.py`

Small early sample RAG implementation. It demonstrates chunking, embedding, inserting, and searching simple documents. It is useful for learning but not the final benchmark pipeline.

### `sample_rag/test_search.py`

Small test harness for `sample_rag/rag_supabase.py`.

### `sample_inputs/*.txt`

Small local text inputs used for early experiments or demonstrations.

### `sql/*.sql`

Database setup and RPC files:

- `documents.sql`: documents table.
- `document_chunks.sql`: chunks table and HNSW index note.
- `match_document_chunks.sql`: main vector search RPC.
- `match_documents_hnsw.sql`: older vector search RPC.
- `hnsw_index.sql`: placeholder or empty index file.

### `sql/hybrid_search  BM25 + Semantic Searcg/*.sql`

Hybrid lexical search setup:

- `alter_table.sql`: adds `title`, `section_path`, `search_text`, and `search_vector`.
- `trigger_for_hybrid_search_columns.sql`: maintains search fields on insert/update.
- `update_document_chunks.sql`: backfills search fields for existing rows.
- `indexes_for_hybrid_retrieval.sql`: GIN and support indexes.
- `bm25_style_retrieval.sql`: lexical retrieval RPC using Postgres full-text ranking.

## Evaluation Questions and Answers

### Q: What is the difference between context answerability and answer answerability?

A: Context answerability asks whether retrieved context alone is enough to answer the query. Answer answerability asks whether the generated final answer is useful, clear, and complete, given the context.

They are related but not identical. A partial context can still lead to a good-enough answer if the query is broad and the partial context contains the key points. That is why the 24 partial-to-yes cases need review but are not automatically proof of leakage.

### Q: What did the leakage check show?

A: It showed no direct cases where context was judged no but the answer was judged yes. That is the strongest potential leakage signal, and it was 0.

The weaker signal was 24 cases where context was partial and answer was yes. These are questionable because the answer model was instructed not to use prior knowledge, but they may also reflect stricter context judging or looser answer judging.

### Q: Why can answer judge yes be higher than context judge yes?

A: Several reasons:

- The context judge may require a complete answer, while the answer judge may accept a concise useful answer.
- The answer may synthesize scattered partial evidence better than the context-only judge expected.
- The query may be broad enough that partial context is practically sufficient.
- The answer judge may be too lenient.
- The answer model may have used prior knowledge despite instructions.

Manual review is required before labeling these as leakage.

### Q: What were the main sources of failure?

A: The main sources were:

- Corpus gaps for modern technical topics.
- Semantic ambiguity, especially short or overloaded terms.
- Insufficient detail in retrieved context.
- Some retrieval misses, often because the corpus did not contain the expected concept or because the query meant a domain-specific sense.

The failure classification artifact showed only a tiny number of clear ANN misses. Most failures were not because HNSW was broken.

### Q: Did reranking solve retrieval?

A: No. Reranking improves ordering only if the right candidate appears in the candidate set. If retrieval does not fetch any relevant candidate, the reranker cannot fix it.

The most important practical lesson is:

Candidate recall comes before reranker quality.

### Q: Did hybrid retrieval solve corpus gaps?

A: No. Hybrid retrieval helps when exact terms exist in the corpus but semantic search misses or misranks them. It cannot retrieve facts that are absent from the corpus.

Hybrid is useful for:

- acronyms
- dates
- numeric identifiers
- exact entity names
- rare technical terms

It is less useful for broad conceptual questions where semantic search already performs well.

### Q: Did context-aware chunking solve all chunking problems?

A: No. It improved chunk quality and context answerability, but some chunks still ended mid-sentence or lacked enough detail. Some pages are lists, timelines, stubs, or tables expressed as plain text, so chunking remains imperfect.

Still, the improvement over fixed overlap was clear enough to prefer context-aware chunking.

## How to Reproduce the Main Pipeline

Run from:

```bash
cd retrieval_augmented_generation
source venv/bin/activate
```

Required environment variables:

```bash
OPENAI_API_KEY=...
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
COHERE_API_KEY=...
```

Main flow:

```bash
python ingest_wikipedia_context_aware_chunk_overlap.py --max-articles 20000
python benchmark_retrieval.py --mode hnsw --top-k 50
python rerank_retrieval_results.py \
  --mode hnsw \
  --input benchmark_results/benchmark_results_context_aware_ef_search_120_hnsw.json \
  --output benchmark_results/reranked_results_context_aware_ef_search_120_hnsw.json \
  --candidate-limit 50 \
  --rerank-top-k 10
python judge_context_quality.py \
  --input benchmark_results/reranked_results_context_aware_ef_search_120_hnsw.json \
  --strategy context_aware_chunk_v1 \
  --retrieval-path benchmark_results_ef_search_120_hnsw \
  --resume
python answer_question.py
python judge_rag_answers.py --resume
python compare_context_answer_judges.py
```

For neighbor-expanded retrieval:

```bash
python benchmark_retrieval_with_neighbors.py \
  --top-k 50 \
  --ef-search 120 \
  --neighbor-window 1 \
  --output benchmark_results/benchmark_results_context_aware_neighbor_expanded_ef_search_120_hnsw.json \
  --resume
```

For hybrid retrieval:

```bash
python benchmark_hybrid_retrieval.py \
  --semantic-top-k 50 \
  --lexical-top-k 50 \
  --ef-search 120

python rerank_hybrid_retrieval_results.py \
  --input benchmark_results/benchmark_results_section_aware_v2_hybrid_semantic_top_50_lexical_top_50_ef_search_120.json \
  --output benchmark_results/reranked_results_section_aware_v2_hybrid_semantic_top_50_lexical_top_50_to_top_10.json \
  --candidate-limit 100 \
  --rerank-top-k 10
```

## What We Learned

### Q: What was the biggest technical lesson?

A: RAG quality is mostly bottlenecked by context availability, not final answer generation. When the correct context appears in the top reranked results, the answer model usually performs well and stays grounded. When the context is absent, incomplete, or ambiguous, answer quality drops.

### Q: What was the biggest evaluation lesson?

A: One metric is not enough. Retrieval recall, context answerability, final answer groundedness, and context-vs-answer consistency each reveal different problems.

Flat-vs-HNSW recall can say the ANN index is healthy, while context judging can still show the corpus does not answer the query.

### Q: What was the biggest chunking lesson?

A: Token overlap is a blunt tool. It helps preserve continuity, but it creates duplicated and fragmented context. Context-aware chunking produced cleaner chunks and better judged answerability.

### Q: What was the biggest corpus lesson?

A: Retrieval cannot compensate for a missing corpus. Simple Wikipedia is good for general educational questions but weak for modern AI and infrastructure terms. The project should not be judged as a general RAG system for all technical topics unless the corpus is expanded.

### Q: What was the biggest judge lesson?

A: LLM judges are useful but should not be treated as absolute truth. The 24 context-partial-to-answer-yes cases show that judge definitions and thresholds matter. For high-stakes claims, manually inspect disagreement cases.

### Q: What was the biggest production lesson?

A: Production RAG should be modular:

- chunking strategy
- retrieval strategy
- reranking
- answer generation
- context judging
- answer judging
- disagreement analysis

Keeping those stages separate made this project debuggable.

## Recommended Next Questions

These are the questions a future user should ask before continuing the project:

1. Should the corpus be expanded beyond Simple Wikipedia?
2. Should modern technical docs be added for RAG, vector DBs, Kubernetes, Docker, Spark, Hadoop, and distributed systems?
3. Should hybrid retrieval become the default only for acronym/date/exact-match queries?
4. Should the answer model be forced to quote exact supporting source spans before answering?
5. Should the 24 partial-to-yes cases be manually labeled as acceptable synthesis, judge leniency, or possible prior-knowledge leakage?
6. Should context judge and answer judge rubrics be calibrated against human labels?
7. Should we add a second judge model or majority-vote judging for disagreement cases?
8. Should neighbor expansion be kept, removed, or made conditional based on chunk metadata?
9. Should chunking include table/list-specific representations instead of treating everything as prose?
10. Should reranking compare Cohere Rerank against OpenAI or local rerankers?
11. Should the pipeline store source spans and citations more rigorously?
12. Should benchmark queries be split into in-corpus and out-of-corpus sets?
13. Should evaluation track answer refusal quality separately from answer correctness?
14. Should HNSW `ef_search` be tuned per latency budget rather than fixed globally?
15. Should routed retrieval be evaluated against final answer quality, not only route labels?

## Final Project State

The project is in a good state for a learning and benchmarking RAG system. It has:

- Multiple chunking strategies.
- HNSW retrieval with `ef_search` tuning.
- Flat-vs-HNSW comparison logic.
- Reranking.
- Hybrid retrieval.
- Routed retrieval.
- Context judging.
- Answer generation.
- Answer judging.
- Context-vs-answer disagreement analysis.
- Stored benchmark artifacts and plots.

The strongest conclusion is that context-aware chunking plus HNSW retrieval plus reranking is a solid baseline for the Simple Wikipedia corpus. The remaining failures are mostly corpus coverage, ambiguity, and insufficient retrieved detail rather than obvious answer-model hallucination.

The project is ready to close as an experiment, and the best next version would focus less on pipeline mechanics and more on corpus quality, human-labeled evaluation, and stricter evidence-backed answer generation.
