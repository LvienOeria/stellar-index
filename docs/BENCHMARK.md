# Benchmark report (v0.1 fixture smoke)

- Date: 2026-08-30
- Corpus: 2 synthetic fixture books + full Project Gutenberg rebuild (20 books, 8,256 chunks)
- Model: `deepseek-v4-flash` for generation
- QA: 10-question fixture golden set
- Repro: `uv run stellar-index eval --fixtures [--dense --rerank full]`

## Retrieval-only (gold evidence chunks)

| configuration | Hit@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| BM25 | 0.900 | 1.000 | 1.000 | 0.200 | 0.933 | 0.950 |
| BM25 + bge-small dense + bge-reranker-base | 0.900 | 1.000 | 1.000 | 0.200 | 0.950 | 0.963 |

## Generation (rag, BM25-only, 10 QA)

| metric | value |
|---|---|
| Accuracy | 0.900 |
| Mean citation precision | 1.000 |
| Total cost | $0.0041 |
| Mean latency | 2.26 s |

## Full corpus

- 20 public-domain books from Project Gutenberg Science Fiction bookshelf.
- `uv run stellar-index build` downloads and indexes them; texts are never committed.
- Full 100-question golden QA set is the next milestone (current fixture has 10).
