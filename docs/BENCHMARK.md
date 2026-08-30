# Benchmark report (v0.1 full corpus)

- Date: 2026-08-30
- Corpus: 20 Project Gutenberg public-domain sci-fi books, 8,256 chunks
- Golden QA: 110 items (auto-drafted by DeepSeek, then programmatically validated: evidence quotes must appear verbatim in the indexed chapter)
- Model: `deepseek-v4-flash` (generation and LLM rerank arms)
- Repro:
  - corpus: `uv run stellar-index build`
  - BM25 retrieval: `uv run stellar-index eval --retrieval-only`
  - dense+rerank: `uv run python scripts/retrieval_bench.py`
  - LLM rerank: `uv run python scripts/retrieval_bench_llm.py --rerank llm`

## Retrieval (110 QA, evidence-chunk ranking within the correct book)

| configuration | Hit@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 |
|---|---|---|---|---|---|
| BM25 (SQLite FTS5) | 0.182 | 0.298 | 0.389 | 0.067 | 0.253 |
| BM25 + bge-small dense + bge-reranker-base | 0.155 | 0.317 | 0.426 | 0.073 | 0.249 |
| BM25 + DeepSeek LLM pointwise rerank | **0.309** | **0.412** | **0.452** | **0.096** | **0.377** |

## Generation (rag, BM25-only, 110 QA)

| metric | value |
|---|---|
| Accuracy (strict keyphrase match) | 0.291 |
| Mean citation precision | 0.765 |
| Answerable=false rate | 25.5% |
| Total model cost | $0.182 |
| Mean latency | 12.75 s |

## Findings

1. On this corpus, BM25 is a strong sparse baseline; local dense + bge-reranker-base improves Recall@10 slightly but lowers Hit@1.
2. DeepSeek LLM rerank is the best current reranker on our evidence-chunk task (+70% Hit@1 and +49% MRR vs BM25), at the cost of one extra LLM call per query.
3. Generation accuracy is the weak point: many auto-drafted QA items are strict or ambiguous. Next iteration must improve retrieval-to-generation context selection, answer matching, and curate the golden set before treating accuracy as a headline metric.
4. Citation precision 0.765 shows the citation checker is working; fabricated quotes are surfaced instead of silently accepted.

## Next benchmark iteration

- Run generation with `--rerank llm` on the full set (already 7/110 rows, slow but in progress).
- Curate the 110 QA: drop ambiguous items, normalize acceptable answers, split dev/test.
- Add Self-Route and long-context generation arms on a 20-question subset.
