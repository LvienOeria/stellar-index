# stellar-index

> 中文：面向公版科幻小说的深读助手——每个答案带原文引用，并用 DeepSeek V4 的 1M 上下文检验 RAG 是否仍然必要。
> English: A citation-grounded deep-reading assistant for public-domain sci-fi novels; compares RAG, 1M-token long context, and self-routing retrieval.

## Why

DeepSeek V4 can fit an entire novel in its 1M-token context. Does RAG still pay for itself? `stellar-index` builds a reproducible retrieval stack (BM25 + bge-small-en-v1.5 + bge-reranker-base), a long-context arm, and a Self-Route hybrid, then measures answer quality, citation precision, token cost, and latency on a golden QA set.

## Quickstart

```bash
uv venv .venv
uv pip install -e .
# BM25-only path works without model downloads:
uv run stellar-index build --fixtures
uv run stellar-index ask --book fixtures/the_last_observatory --question "Who is Commander Voss?"
uv run stellar-index eval --fixtures
# Full retrieval stack (downloads ~1.25 GB models):
uv pip install -e '.[retrieval]'
python -m spacy download en_core_web_sm
uv run stellar-index build --fixtures --dense
```

## Architecture

```text
src/stellarindex/
  corpus.py    download / clean / chapter-aware parent-child chunking
  search.py    BM25 (SQLite FTS5) + optional LanceDB dense + cross-encoder rerank
  llm.py       DeepSeek V4 client
  qa.py        RAG / long-context / Self-Route answering modes
  evals.py     retrieval, generation, citation, efficiency metrics
  ui.py        Streamlit app
fixtures/      2 synthetic public-domain-style mini books + 10 golden QA (offline CI)
```

## Evaluation

- 100 golden QA (20 dev / 80 test) in the full corpus; fixtures ship 10 QA for CI.
- Metrics: Recall@5/10, Precision@5, Hit@1, MRR@10, nDCG@10, exact-match/F1, citation precision/recall, groundedness, relevance, token cost, latency.
- Every RAG answer includes `quote` and chapter/paragraph location verified by an independent checker.

## PM Artifacts

- `docs/PRD.md`
- `docs/ADR/`
- `docs/BENCHMARK.md` (generated)

## License

MIT. Book texts are not committed; the full corpus is rebuilt from Project Gutenberg with `data/books.toml`.
