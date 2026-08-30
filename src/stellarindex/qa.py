from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .corpus import Book
from .llm import DeepSeekClient, estimate_cost
from .search import SearchResult, hybrid_search


@dataclass
class Citation:
    quote: str
    book_id: str
    chapter_idx: int
    parent_id: str


@dataclass
class Answer:
    question: str
    answer: str
    mode: str
    citations: list[Citation] = field(default_factory=list)
    answerable: bool | None = None
    confidence: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def citation_text(self) -> str:
        return "\n".join(
            f"[{i + 1}] {c.book_id} chapter {c.chapter_idx + 1}: {c.quote}" for i, c in enumerate(self.citations)
        )


def _context_blocks(results: list[SearchResult], top_k: int = 5) -> tuple[str, list[Citation]]:
    citations: list[Citation] = []
    blocks = []
    for rank, result in enumerate(results[:top_k], start=1):
        block = (
            f"<passage id=\"{rank}\" book=\"{result.book_id}\" "
            f"chapter=\"{result.chapter_idx + 1}\" parent=\"{result.parent_id}\">\n"
            f"{result.parent_text}\n</passage>"
        )
        blocks.append(block)
        citations.append(
            Citation(
                quote=result.text[:300],
                book_id=result.book_id,
                chapter_idx=result.chapter_idx,
                parent_id=result.parent_id,
            )
        )
    return "\n\n".join(blocks), citations


def _answer_prompt(question: str, context: str) -> str:
    return (
        "Answer the question about the provided passages. Output strict JSON with this shape:\n"
        '{"answer": "...", "quotes": [{"quote": "exact short quote from passage", '
        '"book": "...", "chapter": 1, "parent": "..."}], "answerable": true, "confidence": 0.0, '
        '"reason": "..."}\n'
        "Rules: every claim in `answer` must be supported by `quotes`; quote text must be copied "
        "verbatim from a passage; confidence is 0.0 to 1.0; set answerable=false only if the passages "
        "do not contain enough information.\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {question}"
    )


def _long_prompt(question: str, book: Book) -> str:
    return (
        "You are reading one complete public-domain book. Answer the question and output strict JSON:\n"
        '{"answer": "...", "quotes": [{"quote": "exact quote", "chapter": 1}], "answerable": true, '
        '"confidence": 0.0, "reason": "..."}\n'
        "Quote verbatim from the text and identify the chapter number.\n\n"
        f"<book title=\"{book.title}\">\n{book.full_text}\n</book>\n\n"
        f"Question: {question}"
    )


class QABot:
    def __init__(self, settings: Settings, store: Any):
        self.settings = settings
        self.store = store
        self.llm = DeepSeekClient(settings)

    def _run_json(self, prompt: str, question: str, mode: str, started: float) -> Answer:
        messages = [
            {
                "role": "system",
                "content": "You are a meticulous literary research assistant. Always return strict JSON.",
            },
            {"role": "user", "content": prompt},
        ]
        reply = self.llm.complete_json_reply(messages, temperature=0.0)
        data = reply.data
        citations = []
        for item in data.get("quotes", []):
            citations.append(
                Citation(
                    quote=str(item.get("quote", ""))[:500],
                    book_id=str(item.get("book", "")),
                    chapter_idx=int(item.get("chapter", 1)) - 1,
                    parent_id=str(item.get("parent", "")),
                )
            )
        answer = Answer(
            question=question,
            answer=str(data.get("answer", "")),
            mode=mode,
            citations=citations,
            answerable=bool(data.get("answerable", True)) if isinstance(data.get("answerable"), bool) else None,
            confidence=float(data.get("confidence", 0.0)) if data.get("confidence") is not None else None,
            raw=data,
        )
        answer.wall_seconds = time.time() - started
        answer.input_tokens = reply.input_tokens
        answer.output_tokens = reply.output_tokens
        answer.cost_usd = estimate_cost(self.settings.model, reply.input_tokens, reply.output_tokens)
        return answer

    def ask_rag(self, question: str, *, dense: bool = False, rerank: str = "none") -> Answer:
        started = time.time()
        results = hybrid_search(
            question,
            self.store,
            self.settings,
            dense=dense,
            rerank=rerank,
            llm_client=self.llm,
        )
        context, _ = _context_blocks(results, top_k=min(5, len(results)))
        prompt = _answer_prompt(question, context)
        answer = self._run_json(prompt, question, f"rag:{rerank}:{'dense' if dense else 'bm25'}", started)
        return answer

    def ask_longctx(self, question: str, book: Book) -> Answer:
        started = time.time()
        prompt = _long_prompt(question, book)
        answer = self._run_json(prompt, question, "long-context", started)
        for citation in answer.citations:
            if not citation.book_id:
                citation.book_id = book.book_id
        return answer

    def ask_self_route(self, question: str, book: Book, *, dense: bool = False, rerank: str = "full") -> Answer:
        started = time.time()
        results = hybrid_search(question, self.store, self.settings, dense=dense, rerank=rerank, llm_client=self.llm)
        context, _ = _context_blocks(results, top_k=5)
        prompt = _answer_prompt(question, context)
        answer = self._run_json(prompt, question, "self-route", started)
        if answer.answerable is False or (answer.confidence is not None and answer.confidence < 0.6):
            return self.ask_longctx(question, book)
        return answer
