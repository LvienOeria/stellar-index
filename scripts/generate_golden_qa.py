#!/usr/bin/env python3
"""Draft 5 golden QA items per indexed book with DeepSeek, then validate them.

Usage:
    uv run python scripts/generate_golden_qa.py [--books 5]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from stellarindex.config import Settings
from stellarindex.corpus import load_gutenberg_book
from stellarindex.llm import DeepSeekClient

PROMPT = """You are creating a reading-comprehension evaluation set from a public-domain novel.
Output strict JSON with a "questions" array of exactly 5 objects:
{"book_id": "...", "type": "single-chapter"|"cross-chapter"|"thematic",
 "question": "...", "answers": ["short acceptable key phrase", "..."],
 "evidence": ["verbatim quote copied from the text", "..."],
 "evidence_chapter": 1}
Rules:
- 2 single-chapter factual questions, 2 cross-chapter questions, 1 thematic question.
- `evidence` must be copied verbatim from the provided text (20 words or fewer each).
- `answers` must be short phrases that would unambiguously appear in a correct answer.
- chapter numbers are 1-based.
- Questions must be answerable from the text alone; no external knowledge.

<book title="%s">
%s
</book>
"""


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--books", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/golden_qa.json")
    args = parser.parse_args()

    settings = Settings(data_dir=Path(args.data_dir), raw_dir=Path(args.data_dir) / "raw")
    client = DeepSeekClient(settings)
    conn = sqlite3.connect(settings.index_dir / "gutenberg.db")
    rows = conn.execute("SELECT book_id, title, author FROM books ORDER BY book_id").fetchall()
    if args.books:
        rows = rows[: args.books]

    all_items = []
    for book_id, title, author in rows:
        raw = settings.raw_dir / f"pg{book_id}.txt"
        if not raw.exists():
            continue
        book = load_gutenberg_book(raw, book_id, title, author)
        excerpt = book.full_text[:60_000]
        reply = client.complete_json_reply(
            [
                {"role": "system", "content": "You are an evaluation-set author. Return strict JSON only."},
                {"role": "user", "content": PROMPT % (title, excerpt)},
            ],
            temperature=0.4,
            max_tokens=8_000,
        )
        items = reply.data.get("questions", [])
        for item in items:
            if not isinstance(item, dict):
                continue
            item["book_id"] = str(book_id)
            item["type"] = item.get("type", "single-chapter")
            raw_answers = item.get("answers", [])
            raw_evidence = item.get("evidence", [])
            raw_chapter = item.get("evidence_chapter", 1)
            if isinstance(raw_chapter, list):
                raw_chapter = raw_chapter[0] if raw_chapter else 1
            item["answers"] = [str(a) for a in raw_answers][:5]
            item["evidence"] = [str(e) for e in raw_evidence][:3]
            try:
                item["evidence_chapter"] = int(raw_chapter)
            except (TypeError, ValueError):
                item["evidence_chapter"] = 1
        all_items.extend(items)
        print(f"{book_id} {title}: {len(items)} items", flush=True)

    validated = []
    dropped = []
    for item in all_items:
        raw = settings.raw_dir / f"pg{item['book_id']}.txt"
        if not raw.exists():
            dropped.append((item["question"], "missing raw"))
            continue
        book = load_gutenberg_book(raw, item["book_id"])
        if not item["question"] or not item["answers"] or not item["evidence"]:
            dropped.append((item["question"], "missing fields"))
            continue
        # Locate the first evidence quote in any chapter instead of trusting
        # the drafted chapter number (PG chapter detection varies by edition).
        chapter = max(1, min(item["evidence_chapter"], len(book.chapters)))
        found_chapter = None
        for idx, ch in enumerate(book.chapters, start=1):
            if all(normalize(e) in normalize(ch) for e in item["evidence"]):
                found_chapter = idx
                break
        if found_chapter is None:
            dropped.append((item["question"], "evidence not found in any chapter"))
            continue
        chapter = found_chapter
        if any(q["question"] == item["question"] for q in validated):
            dropped.append((item["question"], "duplicate"))
            continue
        item["evidence_chapter"] = chapter
        validated.append(item)

    Path(args.out).write_text(json.dumps(validated, indent=2, ensure_ascii=False) + "\n")
    print(f"validated={len(validated)} dropped={len(dropped)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
