# AI Infrastructure Experiments

This repository contains hands-on AI/ML infrastructure experiments across retrieval, model training, evaluation, and domain-specific pipelines. The projects are intentionally practical: each folder contains runnable code, experiment scripts, and notes about the tradeoffs being tested.

## Projects

- `retrieval_augmented_generation/`: Supabase/pgvector RAG experiments covering ingestion, HNSW retrieval, neighbor expansion, hybrid semantic + lexical retrieval, Cohere reranking, and OpenAI-based context-quality judging.
- `credit_default/`: neural-network experiments for credit-card default prediction, including preprocessing, architecture search, regularization, threshold tuning, and evaluation.
- `imdb_sentiment_analysis/`: sentiment-analysis training and prediction scripts.
- `finance-news-summarizer/`: finance-news summarization pipeline with prompt loading, inference, validation, evaluation, and repair prompts.

## Current Focus: RAG Retrieval Quality

The most developed experiment is in `retrieval_augmented_generation/`. It compares several retrieval strategies:

- Exact flat vector search vs HNSW approximate vector search.
- HNSW `ef_search` latency/recall tradeoffs.
- Fixed-overlap chunking vs context-aware chunking.
- Neighbor-expanded retrieval for section-aware chunks.
- Hybrid retrieval that merges semantic HNSW candidates with BM25-style lexical candidates, then reranks the merged candidate set.

See `retrieval_augmented_generation/README.md` for setup, SQL scripts, benchmark commands, expected output files, and evaluation notes.

## Setup

Each project folder owns its own dependencies and workflow. Start with the folder-specific README or requirements file before running scripts.

For the RAG experiments:

```bash
cd retrieval_augmented_generation
source venv/bin/activate
```

The RAG scripts expect environment variables such as `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `COHERE_API_KEY` to be present in `.env`.

## Purpose

This repo is for learning and validating AI infrastructure patterns end to end: data preparation, retrieval design, model evaluation, latency/quality measurement, reranking, and production-oriented tradeoffs.
