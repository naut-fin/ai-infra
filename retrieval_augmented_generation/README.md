# Retrieval Augmented Generation Experiments

This folder contains ingestion, retrieval benchmarking, reranking, and context-quality evaluation scripts for the Supabase/pgvector RAG experiments.

Run commands from this directory:

```bash
cd retrieval_augmented_generation
source venv/bin/activate
```

Required environment variables are loaded from `.env`:

```bash
OPENAI_API_KEY=...
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
COHERE_API_KEY=...
```

## Main Files

- `ingest_wikipedia_fixed_chunk_overlap.py`: ingests Simple Wikipedia using fixed-size chunks with token overlap.
- `ingest_wikipedia_context_aware_chunk_overlap.py`: ingests Simple Wikipedia using section/context-aware chunks.
- `benchmark_retrieval.py`: runs flat vs HNSW retrieval benchmarks and compares HNSW against flat ground truth.
- `benchmark_retrieval_with_neighbors.py`: runs HNSW retrieval, then expands each seed chunk with neighboring chunks.
- `rerank_retrieval_results.py`: reranks benchmark result files with Cohere Rerank and compares flat vs HNSW after reranking.
- `judge_context_quality.py`: uses an OpenAI judge model to evaluate whether reranked context answers the query well.
- `plot_ef_search_benchmarks.py`: plots ef_search latency/recall tradeoffs from benchmark result JSON files.
- `sql/match_document_chunks.sql`: RPC used for HNSW search with `hnsw.ef_search`.
- `sql/document_chunks.sql`: table shape and the HNSW index statement.

## Flat vs HNSW Experiment

Use `benchmark_retrieval.py` for the flat vs HNSW retrieval experiment.

Important: when measuring flat search, drop the HNSW index first. Both flat and HNSW modes order by vector distance; the difference is whether PostgreSQL can use the HNSW index. If the index exists during the flat run, the flat baseline may not be exact.

In Supabase SQL editor, drop the HNSW index before the flat run:

```sql
DROP INDEX IF EXISTS public.document_chunks_embedding_hnsw_idx;
```

Run the flat baseline:

```bash
python benchmark_retrieval.py --mode flat --top-k 50
```

Recreate the HNSW index before the HNSW run:

```sql
CREATE INDEX document_chunks_embedding_hnsw_idx
ON public.document_chunks
USING hnsw (embedding vector_cosine_ops);
```

Run HNSW:

```bash
python benchmark_retrieval.py --mode hnsw --top-k 50
```

Compare HNSW against flat:

```bash
python benchmark_retrieval.py --compare
```

Expected output files:

- Flat: `benchmark_results/benchmark_results_context_aware_flat.json`
- HNSW: `benchmark_results/benchmark_results_context_aware_ef_search_120_hnsw.json`
- Comparison: `benchmark_results/compare_context_aware_flat_vs_hnsw.json`

## Tuning HNSW

The main HNSW runtime knob is `ef_search`, set inside the Supabase RPC:

```sql
PERFORM set_config('hnsw.ef_search', ef_search::text, true);
```

Higher `ef_search` usually improves recall and reranked quality, but it increases database work and latency. Lower `ef_search` is cheaper and faster, but can miss relevant chunks.

To tune HNSW:

1. Keep `top-k`, dataset, embeddings, and query set fixed.
2. Run one exact flat benchmark as ground truth.
3. Run HNSW at multiple `ef_search` values.
4. Compare HNSW recall/NDCG against flat.
5. Pick the lowest `ef_search` that reaches acceptable recall and reranked quality within the latency budget.

Current scripts use a hard-coded `ef_search_list` in both `benchmark_retrieval.py` and `rerank_retrieval_results.py`. To sweep values, edit:

```python
ef_search_list = [20, 40, 60, 80, 120, 160, 200, 300, 400]
```

Then run:

```bash
python benchmark_retrieval.py --mode flat --top-k 50
python benchmark_retrieval.py --mode hnsw --top-k 50
python benchmark_retrieval.py --compare
```

For reranked recall/quality during the sweep:

```bash
python rerank_retrieval_results.py \
  --mode flat \
  --input benchmark_results/benchmark_results_context_aware_flat.json \
  --output benchmark_results/reranked_results_context_aware_flat.json \
  --candidate-limit 20 \
  --rerank-top-k 10

python rerank_retrieval_results.py \
  --mode hnsw \
  --input benchmark_results/benchmark_results_context_aware_ef_search_120_hnsw.json \
  --output benchmark_results/reranked_results_context_aware_ef_search_120_hnsw.json \
  --candidate-limit 20 \
  --rerank-top-k 10

python rerank_retrieval_results.py --compare --compare-top-k 10
```

Plot the ef_search tradeoff:

```bash
python plot_ef_search_benchmarks.py \
  --results-dir benchmark_results/ef_search_results \
  --output-dir benchmark_results/plots
```

When choosing the final value, compare:

- Retrieval recall/NDCG vs flat.
- Reranked recall/NDCG vs flat.
- Retrieval p50/p95 latency.
- Cost from extra database work and any repeated reranker/judge calls used during evaluation.

`ef_search=120` is the current default used by the scripts and context-quality experiments.

## Context-Aware vs Fixed-Overlap Chunking

This experiment compares fixed overlap chunking against context-aware chunking. Use `ef_search=120` and `top-k=50` for both sides.

Before each ingestion run, clear or recreate the Supabase `documents` and `document_chunks` data so each strategy is evaluated on its own indexed corpus. After ingestion, create the HNSW index before benchmarking:

```sql
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
ON public.document_chunks
USING hnsw (embedding vector_cosine_ops);
```

### Fixed-Overlap Run

Ingest fixed overlap chunks:

```bash
python ingest_wikipedia_fixed_chunk_overlap.py
```

Run retrieval with `top-k=50` and `ef_search=120`:

```bash
python benchmark_retrieval.py --mode hnsw --top-k 50
```

Rerank the same result file:

```bash
python rerank_retrieval_results.py \
  --mode hnsw \
  --input benchmark_results/benchmark_results_context_aware_ef_search_120_hnsw.json \
  --output benchmark_results/reranked_results_context_aware_ef_search_120_hnsw.json \
  --candidate-limit 20 \
  --rerank-top-k 10
```

Judge context quality:

```bash
python judge_context_quality.py \
  --input benchmark_results/reranked_results_context_aware_ef_search_120_hnsw.json \
  --strategy fixed_overlap_v1 \
  --retrieval-path benchmark_results_context_aware_ef_search_120_hnsw \
  --resume
```

### Context-Aware Run

Clear/recreate the Supabase data, then ingest context-aware chunks:

```bash
python ingest_wikipedia_context_aware_chunk_overlap.py --max-articles 20000
```

Run HNSW retrieval with neighbor expansion. This is the preferred context-aware retrieval path because it expands each seed chunk with nearby chunks:

```bash
python benchmark_retrieval_with_neighbors.py \
  --top-k 50 \
  --ef-search 120 \
  --neighbor-window 1 \
  --output benchmark_results/benchmark_results_context_aware_neighbor_expanded_ef_search_120_hnsw.json \
  --resume
```

Rerank the neighbor-expanded result file:

```bash
python rerank_retrieval_results.py \
  --mode hnsw \
  --input benchmark_results/benchmark_results_context_aware_neighbor_expanded_ef_search_120_hnsw.json \
  --output benchmark_results/reranked_results_context_aware_neighbor_expanded_ef_search_120_hnsw.json \
  --candidate-limit 20 \
  --rerank-top-k 10
```

Judge context quality:

```bash
python judge_context_quality.py \
  --input benchmark_results/reranked_results_context_aware_neighbor_expanded_ef_search_120_hnsw.json \
  --strategy context_aware_chunk_neighbor_expanded_v1 \
  --retrieval-path benchmark_results_context_aware_neighbor_expanded_ef_search_120_hnsw \
  --resume
```

Compare the resulting judge files:

- `benchmark_results/context_quality_judge_fixed_overlap_v1_hnsw_reranked_top10.json`
- `benchmark_results/context_quality_judge_context_aware_chunk_neighbor_expanded_v1_hnsw_reranked_top10.json`

## Notes

- `benchmark_retrieval_with_neighbors.py` checkpoints after every query and supports `--resume`.
- `judge_context_quality.py` also supports `--resume`.
- Keep `top-k`, `ef_search`, candidate limit, rerank top-k, query set, and article count fixed when comparing chunking strategies.
- For flat vs HNSW, always drop the HNSW index before the flat run and recreate it before the HNSW run.
