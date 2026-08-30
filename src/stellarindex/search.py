from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .corpus import Book, Chunk, chunk_book


@dataclass
class SearchResult:
    chunk_id: str
    book_id: str
    chapter_idx: int
    parent_id: str
    text: str
    parent_text: str
    score: float
    rank_source: str


class IndexStore:
    """SQLite-backed index. FTS5 supplies BM25; optional LanceDB supplies dense."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS books (
                book_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT NOT NULL,
                source TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL,
                chapter_idx INTEGER NOT NULL,
                parent_id TEXT NOT NULL,
                text TEXT NOT NULL,
                parent_text TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                start_sentence INTEGER NOT NULL,
                end_sentence INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                book_id UNINDEXED,
                chapter_idx UNINDEXED,
                parent_id UNINDEXED,
                text
            );
            """
        )
        self.conn.commit()

    def rebuild(self, books: list[Book], settings: Settings) -> int:
        self.conn.executescript("DELETE FROM books; DELETE FROM chunks; DELETE FROM chunks_fts;")
        rows: list[tuple[Any, ...]] = []
        for book in books:
            self.conn.execute(
                "INSERT INTO books(book_id, title, author, source) VALUES (?,?,?,?)",
                (book.book_id, book.title, book.author, book.source),
            )
            chunks = chunk_book(
                book,
                child_target=settings.child_tokens,
                parent_target=settings.parent_tokens,
            )
            parents: dict[str, str] = {}
            for chunk in chunks:
                parents[chunk.parent_id] = parents.get(chunk.parent_id, "") + " " + chunk.text
            for chunk in chunks:
                rows.append(
                    (
                        chunk.chunk_id,
                        chunk.book_id,
                        chunk.chapter_idx,
                        chunk.parent_id,
                        chunk.text,
                        parents[chunk.parent_id].strip(),
                        chunk.token_estimate,
                        chunk.start_sentence,
                        chunk.end_sentence,
                    )
                )
        self.conn.executemany(
            """INSERT INTO chunks(
                chunk_id, book_id, chapter_idx, parent_id, text, parent_text,
                token_estimate, start_sentence, end_sentence)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.executemany(
            "INSERT INTO chunks_fts(chunk_id, book_id, chapter_idx, parent_id, text) VALUES (?,?,?,?,?)",
            [(r[0], r[1], r[2], r[3], r[4]) for r in rows],
        )
        self.conn.commit()
        return len(rows)

    def close(self) -> None:
        self.conn.close()


def _fts_query(query: str) -> str:
    tokens = [t for t in re.findall(r"[A-Za-z0-9_']+", query.lower()) if t not in {"the", "a", "an", "of", "in", "on"}]
    if not tokens:
        tokens = [t for t in re.findall(r"[A-Za-z0-9_']+", query.lower())]
    return " OR ".join(f'"{token}"' for token in tokens[:12])


def bm25_search(store: IndexStore, query: str, limit: int = 50, book_id: str | None = None) -> list[SearchResult]:
    fts = _fts_query(query)
    if book_id:
        rows = store.conn.execute(
            """
            SELECT c.*, bm25(chunks_fts) AS score
            FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ? AND c.book_id = ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (fts, book_id, limit),
        ).fetchall()
    else:
        rows = store.conn.execute(
            """
            SELECT c.*, bm25(chunks_fts) AS score
            FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (fts, limit),
        ).fetchall()
    results = [_row_to_result(row, "bm25", raw_score=True) for row in rows]
    return _normalize_scores(results)


def _row_to_result(row: sqlite3.Row, source: str, raw_score: bool = False) -> SearchResult:
    score = float(row["score"])
    if raw_score:
        score = -score
    return SearchResult(
        chunk_id=row["chunk_id"],
        book_id=row["book_id"],
        chapter_idx=int(row["chapter_idx"]),
        parent_id=row["parent_id"],
        text=row["text"],
        parent_text=row["parent_text"],
        score=score,
        rank_source=source,
    )


def _normalize_scores(results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return results
    min_score = min(r.score for r in results)
    max_score = max(r.score for r in results)
    span = max_score - min_score or 1.0
    for r in results:
        r.score = (r.score - min_score) / span
    return results


def dense_search(
    query: str,
    chunks: list[dict[str, Any]],
    embeddings: Any,
    limit: int = 100,
) -> list[tuple[str, float]]:
    """Optional dense arm. `embeddings` exposes `.encode(list[str])` and `.similarity(q, d)`.

    The full LanceDB implementation lives behind this small interface so the
    BM25-only path never imports torch or lancedb.
    """
    corpus = [c["text"] for c in chunks]
    query_vec = embeddings.encode([query])
    doc_vecs = embeddings.encode(corpus, batch_size=32)
    scores = embeddings.similarity(query_vec, doc_vecs).tolist()[0]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
    return [(chunks[i]["chunk_id"], float(scores[i])) for i in ranked]


def rrf_fuse(
    bm25_results: list[SearchResult],
    dense_ranked: list[tuple[str, float]],
    k: int = 60,
    limit: int = 50,
) -> list[SearchResult]:
    scores: dict[str, tuple[SearchResult, float]] = {}
    for rank, result in enumerate(bm25_results):
        scores[result.chunk_id] = (result, 1.0 / (k + rank + 1))
    for rank, (chunk_id, _score) in enumerate(dense_ranked):
        if chunk_id in scores:
            result, old = scores[chunk_id]
            scores[chunk_id] = (result, old + 1.0 / (k + rank + 1))
    ranked = sorted(scores.values(), key=lambda item: item[1], reverse=True)[:limit]
    for result, fused in ranked:
        result.score = fused
        result.rank_source = "rrf"
    return [result for result, _ in ranked]


_RERANK_MODELS: dict[str, Any] = {}


class Reranker:
    """Cross-encoder reranker with a fast fallback and optional LLM arm."""

    def __init__(self, settings: Settings, mode: str = "full"):
        self.settings = settings
        self.mode = mode
        self._model = None
        if mode in {"full", "fast"}:
            model_name = (
                settings.fast_reranker_model if mode == "fast" else settings.reranker_model
            )
            if model_name in _RERANK_MODELS:
                self._model = _RERANK_MODELS[model_name]
                return
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._model = CrossEncoder(model_name)
                _RERANK_MODELS[model_name] = self._model
            except Exception:
                self._model = None

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if self._model is None or not results:
            return results
        max_chars = 1_800  # ~450 tokens; keeps long-novel reranking memory-bounded
        pairs = [(query, (r.parent_text or r.text)[:max_chars]) for r in results]
        batch_size = 8 if self.mode == "full" else 16
        batch_size = min(batch_size, len(pairs) or 1)
        scores = self._model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        for result, score in zip(results, scores, strict=True):
            result.score = float(score)
            result.rank_source = f"cross-encoder:{self.mode}"
        return sorted(results, key=lambda r: r.score, reverse=True)


def llm_rerank(query: str, results: list[SearchResult], client: Any) -> list[SearchResult]:
    """Experimental DeepSeek pointwise rerank arm."""
    if not results:
        return results
    payload = {
        "query": query,
        "candidates": [
            {"chunk_id": r.chunk_id, "text": r.parent_text[:1200]} for r in results[:20]
        ],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval reranker. Output strict JSON: "
                '{"scores": [{"chunk_id": "...", "relevance": 0.0}]}. '
                "Relevance is from 0 to 1."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    data = client.complete_json(messages, temperature=0.0)
    score_map = {item.get("chunk_id"): float(item.get("relevance", 0.0)) for item in data.get("scores", [])}
    for result in results:
        result.score = score_map.get(result.chunk_id, result.score)
        result.rank_source = "llm-rerank"
    return sorted(results, key=lambda r: r.score, reverse=True)


def hybrid_search(
    query: str,
    store: IndexStore,
    settings: Settings,
    *,
    dense: bool = False,
    rerank: str = "none",
    llm_client: Any = None,
    book_id: str | None = None,
) -> list[SearchResult]:
    bm25_results = bm25_search(store, query, limit=100, book_id=book_id)
    if os.getenv("STELLAR_DEBUG"):
        print("debug: bm25", len(bm25_results), flush=True)
    reranker_obj = Reranker(settings, mode=rerank) if rerank in {"full", "fast"} else None
    if dense:
        # Dense arm requires the optional retrieval extra. It is deliberately
        # best-effort: failure falls back to BM25 and records the fallback.
        try:
            from .dense import DenseIndex

            dense_index = DenseIndex(settings)
            if os.getenv("STELLAR_DEBUG"):
                print("debug: dense loaded", flush=True)
            ranked = dense_index.search(query, top_k=100)
            if os.getenv("STELLAR_DEBUG"):
                print("debug: dense ranked", len(ranked), flush=True)
            results = rrf_fuse(bm25_results, ranked, k=settings.rrf_k, limit=settings.top_k_hybrid)
        except Exception:
            results = bm25_results[: settings.top_k_hybrid]
    else:
        results = bm25_results[: settings.top_k_hybrid]
    if book_id:
        results = [r for r in results if r.book_id == book_id]
    if reranker_obj is not None:
        if os.getenv("STELLAR_DEBUG"):
            print("debug: before rerank", len(results), flush=True)
        results = reranker_obj.rerank(query, results)
        if os.getenv("STELLAR_DEBUG"):
            print("debug: after rerank", flush=True)
    elif rerank == "llm" and llm_client is not None:
        results = llm_rerank(query, results, llm_client)
    return results
