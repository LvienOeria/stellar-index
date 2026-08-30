# Benchmark report (v0.2 curated split)

- Date: 2026-08-30
- Corpus: 20 Project Gutenberg public-domain sci-fi books, 8,256 chunks
- Golden QA: 110 drafted → 98 curated by LLM auditor → 20 dev / 78 test
- Model: `deepseek-v4-flash`
- Answer correctness: exact keyphrase or token-overlap recall ≥ 0.6 against accepted keyphrases
- Citation precision: fraction of citations whose quote is verifiably in the claimed chapter

## Retrieval (evidence-chunk ranking, curated set)

| configuration | Hit@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 |
|---|---|---|---|---|---|
| BM25 (SQLite FTS5) | 0.182 | 0.298 | 0.389 | 0.067 | 0.253 |
| BM25 + bge-small dense + bge-reranker-base | 0.155 | 0.317 | 0.426 | 0.073 | 0.249 |
| BM25 + DeepSeek LLM pointwise rerank | **0.309** | **0.412** | **0.452** | **0.096** | **0.377** |

## Generation on held-out test set (78 QA)

| mode | accuracy | citation precision | total cost | mean latency | upgraded to long-context |
|---|---|---|---|---|---|
| RAG (BM25 + query rewrite) | 0.641 | 0.878 | $0.132 | 12.8 s | 0 |
| RAG (BM25 + rewrite + LLM rerank) | 0.487 | 0.893 | $0.130* | 26.2 s | 0 |
| Long-context (first 20 subset) | 0.700 | 0.950 | $0.342 | — | 20 |
| **Self-Route (default)** | **0.744** | **0.987** | $0.374 | 13.7 s | 13 |

*LLM-rerank retrieval call cost is not yet included in that row.

## Product decision

Self-Route is the default answering mode: it starts with cheap BM25+rewrite RAG, upgrades only uncertain questions to full-book long context. On the curated test set it beats pure RAG by **+10.3 points accuracy** and **+10.9 points citation precision**, and beats long-context on both accuracy and cost by upgrading only 13/78 questions.

## Repro

```bash
uv run stellar-index build
uv run stellar-index eval --retrieval-only --qa-file data/splits/test.json
uv run python scripts/generation_bench.py --qa-file data/splits/test.json --mode self-route --rerank none --query-strategy rewrite
```

## Known limitations

- Golden QA is LLM-audited, not fully human-curated.
- Long-context comparison in the table is the first 20 test items; full long-context run is queued.
- Accuracy uses keyphrase recall, not a full LLM judge rubric.
