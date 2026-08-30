from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .corpus import Book, load_fixture_books
from .qa import QABot
from .search import IndexStore, bm25_search, hybrid_search

FIXTURE_QA_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "qa" / "fixture_qa.json"


@dataclass
class QAItem:
    id: str
    book_id: str
    type: str
    question: str
    answers: list[str]
    evidence: list[str]
    evidence_chapter: int


@dataclass
class EvalResult:
    qa_id: str
    mode: str
    retrieval: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    efficiency: dict[str, Any] = field(default_factory=dict)


def load_qa(path: Path | None = None) -> list[QAItem]:
    raw = json.loads((path or FIXTURE_QA_PATH).read_text(encoding="utf-8"))
    items = []
    fields = {f for f in QAItem.__dataclass_fields__}
    for idx, item in enumerate(raw):
        item.setdefault("id", f"{item.get('book_id', 'unknown')}-{idx:03d}")
        item.setdefault("book_id", "unknown")
        item.setdefault("type", "single-chapter")
        item.setdefault("answers", [])
        item.setdefault("evidence", [])
        item.setdefault("evidence_chapter", 1)
        items.append(QAItem(**{k: v for k, v in item.items() if k in fields}))
    return items


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return " ".join(text.split())


def find_evidence_chunks(store: IndexStore, evidence: str, book_id: str | None = None) -> list[str]:
    """Locate gold chunks with normalized text matching.

    Project Gutenberg texts contain italics markup (`_word_`) and curly quotes;
    normalization removes that noise before matching. Parent chunks are matched
    first because evidence quotes often cross child chunk boundaries.
    """
    needle = _normalize(evidence)
    if not needle:
        return []
    rows = store.conn.execute(
        "SELECT chunk_id, parent_id, text, parent_text FROM chunks"
        + (" WHERE book_id=?" if book_id else ""),
        (book_id,) if book_id else (),
    ).fetchall()
    hits: set[str] = set()
    parent_hits: set[str] = set()
    for row in rows:
        if needle in _normalize(row["text"]):
            hits.add(row["chunk_id"])
        if needle in _normalize(row["parent_text"]):
            parent_hits.add(row["parent_id"])
    if parent_hits:
        first_by_parent: dict[str, str] = {}
        for row in rows:
            if row["parent_id"] in parent_hits:
                first_by_parent.setdefault(row["parent_id"], row["chunk_id"])
        hits = set(first_by_parent.values())
    return sorted(hits)[:1]


def retrieval_metrics(
    store: IndexStore,
    qa: list[QAItem],
    settings: Settings,
    *,
    dense: bool = False,
    rerank: str = "none",
) -> dict[str, float]:
    per_query = []
    for item in qa:
        gold_ids = set()
        for evidence in item.evidence:
            gold_ids.update(find_evidence_chunks(store, evidence, item.book_id))
        if not gold_ids:
            continue
        results = hybrid_search(item.question, store, settings, dense=dense, rerank=rerank, book_id=item.book_id)
        ranked_ids = [r.chunk_id for r in results]
        hit1 = 1.0 if ranked_ids and ranked_ids[0] in gold_ids else 0.0
        recall5 = len(set(ranked_ids[:5]) & gold_ids) / len(gold_ids)
        recall10 = len(set(ranked_ids[:10]) & gold_ids) / len(gold_ids)
        precision5 = len(set(ranked_ids[:5]) & gold_ids) / 5.0
        rr = 0.0
        for rank, chunk_id in enumerate(ranked_ids[:10], start=1):
            if chunk_id in gold_ids:
                rr = 1.0 / rank
                break
        dcg = 0.0
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), 10)))
        for rank, chunk_id in enumerate(ranked_ids[:10], start=1):
            if chunk_id in gold_ids:
                dcg += 1.0 / math.log2(rank + 1)
        per_query.append(
            {
                "hit1": hit1,
                "recall5": recall5,
                "recall10": recall10,
                "precision5": precision5,
                "mrr10": rr,
                "ndcg10": dcg / idcg if idcg else 0.0,
            }
        )
    if not per_query:
        return {}
    keys = list(per_query[0])
    return {key: sum(x[key] for x in per_query) / len(per_query) for key in keys}


def _answer_matches(item: QAItem, answer: str) -> bool:
    normalized = _normalize(answer)
    if any(_normalize(accepted) in normalized for accepted in item.answers if accepted):
        return True
    # Token-overlap fallback: accepted answers are key phrases, not full
    # sentences; require most key-phrase tokens to appear in the answer.
    candidate_tokens = set(normalized.split())
    for accepted in item.answers:
        accepted_tokens = set(_normalize(accepted).split())
        if not accepted_tokens:
            continue
        recall = len(accepted_tokens & candidate_tokens) / len(accepted_tokens)
        if recall >= 0.6:
            return True
    return False


def citation_precision(book: Book, answer: Any) -> tuple[float, list[str]]:
    if not answer.citations:
        return 0.0, []
    ok = 0
    failures = []
    full = book.full_text
    for citation in answer.citations:
        quote = _normalize(citation.quote)
        if not quote or quote not in _normalize(full):
            failures.append(citation.quote[:120])
        else:
            ok += 1
    return ok / len(answer.citations), failures


def generation_metrics(item: QAItem, book: Book, answer: Any) -> dict[str, Any]:
    correct = _answer_matches(item, answer.answer)
    precision, failures = citation_precision(book, answer)
    evidence_hits = sum(1 for ev in item.evidence if _normalize(ev) in _normalize(answer.answer))
    return {
        "correct": correct,
        "citation_precision": precision,
        "evidence_hits": evidence_hits,
        "citation_failures": failures,
        "answerable": answer.answerable,
        "confidence": answer.confidence,
    }


def run_eval(
    settings: Settings,
    qa: list[QAItem],
    books: list[Book],
    store: IndexStore,
    *,
    dense: bool = False,
    rerank: str = "none",
    modes: tuple[str, ...] = ("rag",),
    max_questions: int | None = None,
) -> list[EvalResult]:
    bot = QABot(settings, store)
    book_map = {book.book_id: book for book in books}
    results: list[EvalResult] = []
    for item in qa[:max_questions]:
        book = book_map.get(item.book_id)
        if book is None:
            continue
        for mode in modes:
            started = time.time()
            if mode == "rag":
                answer = bot.ask_rag(item.question, dense=dense, rerank=rerank)
            elif mode == "long-context":
                answer = bot.ask_longctx(item.question, book)
            elif mode == "self-route":
                answer = bot.ask_self_route(item.question, book, dense=dense, rerank=rerank)
            else:
                continue
            results.append(
                EvalResult(
                    qa_id=item.id,
                    mode=mode,
                    generation=generation_metrics(item, book, answer),
                    efficiency={
                        "input_tokens": answer.input_tokens,
                        "output_tokens": answer.output_tokens,
                        "cost_usd": answer.cost_usd,
                        "wall_seconds": answer.wall_seconds,
                    },
                    retrieval={"citations": [c.__dict__ for c in answer.citations]},
                )
            )
    return results


def aggregate_results(results: list[EvalResult]) -> dict[str, Any]:
    by_mode: dict[str, list[EvalResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)
    summary = {}
    for mode, mode_results in by_mode.items():
        correct = [r.generation.get("correct") for r in mode_results]
        citation_precisions = [r.generation.get("citation_precision", 0.0) for r in mode_results]
        costs = [r.efficiency.get("cost_usd", 0.0) for r in mode_results]
        summary[mode] = {
            "questions": len(mode_results),
            "accuracy": sum(bool(x) for x in correct) / len(correct) if correct else 0.0,
            "mean_citation_precision": sum(citation_precisions) / len(citation_precisions)
            if citation_precisions
            else 0.0,
            "total_cost_usd": sum(costs),
            "mean_wall_seconds": sum(r.efficiency.get("wall_seconds", 0.0) for r in mode_results)
            / len(mode_results),
        }
    return summary
