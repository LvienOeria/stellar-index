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
class QueryPlan:
    retrieval_query: str
    sub_questions: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


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

    def _plan_query(self, question: str, strategy: str) -> QueryPlan:
        if strategy not in {"rewrite", "decompose"}:
            return QueryPlan(retrieval_query=question)
        messages = [
            {
                "role": "system",
                "content": (
                    "You convert literary questions into effective retrieval queries. "
                    "Output strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Convert this question into a short keyword-rich English retrieval query. "
                    "If the question has multiple parts, also return up to 3 sub_questions.\n"
                    'JSON shape: {"retrieval_query": "...", "sub_questions": ["..."]}\n\n'
                    f"Question: {question}"
                ),
            },
        ]
        reply = self.llm.complete_json_reply(messages, temperature=0.0, max_tokens=800)
        data = reply.data
        return QueryPlan(
            retrieval_query=str(data.get("retrieval_query") or question).strip() or question,
            sub_questions=[str(q) for q in data.get("sub_questions", [])][:3],
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
        )

    def _search_with_plan(
        self,
        question: str,
        plan: QueryPlan,
        *,
        dense: bool,
        rerank: str,
        book_id: str | None = None,
    ) -> list[SearchResult]:
        merged: list[SearchResult] = []
        seen: set[str] = set()
        queries = [plan.retrieval_query, *plan.sub_questions[:2]]
        for query in queries:
            results = hybrid_search(
                query,
                self.store,
                self.settings,
                dense=dense,
                rerank=rerank,
                llm_client=self.llm,
                book_id=book_id,
            )
            for result in results:
                if result.chunk_id not in seen:
                    seen.add(result.chunk_id)
                    merged.append(result)
        if not merged:
            results = hybrid_search(
                question,
                self.store,
                self.settings,
                dense=dense,
                rerank=rerank,
                llm_client=self.llm,
                book_id=book_id,
            )
            for result in results:
                if result.chunk_id not in seen:
                    seen.add(result.chunk_id)
                    merged.append(result)
        return merged

    def _apply_plan_usage(self, answer: Answer, plan: QueryPlan) -> None:
        answer.input_tokens += plan.input_tokens
        answer.output_tokens += plan.output_tokens
        answer.cost_usd += estimate_cost(
            self.settings.model, plan.input_tokens, plan.output_tokens
        )

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
            try:
                chapter_idx = int(item.get("chapter", 1)) - 1
            except (TypeError, ValueError):
                chapter_idx = 0
            citations.append(
                Citation(
                    quote=str(item.get("quote", ""))[:500],
                    book_id=str(item.get("book", "")),
                    chapter_idx=chapter_idx,
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

    def ask_rag(
        self,
        question: str,
        *,
        dense: bool = False,
        rerank: str = "none",
        query_strategy: str = "rewrite",
        book_id: str | None = None,
    ) -> Answer:
        started = time.time()
        plan = self._plan_query(question, query_strategy)
        results = self._search_with_plan(question, plan, dense=dense, rerank=rerank, book_id=book_id)
        context, _ = _context_blocks(results, top_k=min(5, len(results)))
        prompt = _answer_prompt(question, context)
        label = f"rag:{rerank}:{'dense' if dense else 'bm25'}:{query_strategy}"
        answer = self._run_json(prompt, question, label, started)
        self._apply_plan_usage(answer, plan)
        return answer

    def ask_longctx(self, question: str, book: Book) -> Answer:
        started = time.time()
        prompt = _long_prompt(question, book)
        answer = self._run_json(prompt, question, "long-context", started)
        for citation in answer.citations:
            if not citation.book_id:
                citation.book_id = book.book_id
        return answer

    def ask_self_route(
        self,
        question: str,
        book: Book,
        *,
        dense: bool = False,
        rerank: str = "none",
        query_strategy: str = "rewrite",
    ) -> Answer:
        started = time.time()
        plan = self._plan_query(question, query_strategy)
        results = self._search_with_plan(question, plan, dense=dense, rerank=rerank, book_id=book.book_id)
        context, _ = _context_blocks(results, top_k=5)
        prompt = _answer_prompt(question, context)
        answer = self._run_json(prompt, question, "self-route", started)
        self._apply_plan_usage(answer, plan)
        if answer.answerable is False or (answer.confidence is not None and answer.confidence < 0.6):
            upgraded = self.ask_longctx(question, book)
            upgraded.input_tokens += answer.input_tokens
            upgraded.output_tokens += answer.output_tokens
            upgraded.cost_usd += answer.cost_usd
            upgraded.wall_seconds += answer.wall_seconds
            return upgraded
        return answer
