from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import Settings

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "books"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“(])")


@dataclass
class Book:
    book_id: str
    title: str
    author: str
    chapters: list[str] = field(default_factory=list)
    source: str = "fixture"

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.chapters)


@dataclass
class Chunk:
    chunk_id: str
    book_id: str
    chapter_idx: int
    parent_id: str | None
    text: str
    token_estimate: int
    start_sentence: int
    end_sentence: int


_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy  # type: ignore

        _NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser", "lemmatizer"])
    return _NLP


def _sentences(text: str) -> list[str]:
    try:
        return [sent.text.strip() for sent in _get_nlp()(text).sents if sent.text.strip()]
    except Exception:
        return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _group_sentences(sentences: list[str], target: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if current and _token_estimate(current + " " + sent) > target:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_book(book: Book, child_target: int = 256, parent_target: int = 600) -> list[Chunk]:
    """Chapter-aware, sentence-boundary, parent-child chunking."""
    chunks: list[Chunk] = []
    for chapter_idx, chapter in enumerate(book.chapters):
        sentences = _sentences(chapter)
        parents = _group_sentences(sentences, parent_target)
        sentence_cursor = 0
        for parent_idx, parent in enumerate(parents):
            parent_id = f"{book.book_id}:c{chapter_idx}:p{parent_idx}"
            parent_sentences = _sentences(parent)
            children = _group_sentences(parent_sentences, child_target)
            local = 0
            for child_idx, child in enumerate(children):
                child_sentences = _sentences(child)
                chunks.append(
                    Chunk(
                        chunk_id=f"{parent_id}:ch{child_idx}",
                        book_id=book.book_id,
                        chapter_idx=chapter_idx,
                        parent_id=parent_id,
                        text=child,
                        token_estimate=_token_estimate(child),
                        start_sentence=sentence_cursor + local,
                        end_sentence=sentence_cursor + local + len(child_sentences) - 1,
                    )
                )
                local += len(child_sentences)
            sentence_cursor += len(parent_sentences)
    return chunks


def parent_text(chunks: list[Chunk], parent_id: str) -> str:
    return " ".join(c.text for c in chunks if c.parent_id == parent_id)


def load_fixture_books() -> list[Book]:
    books: list[Book] = []
    for path in sorted(FIXTURES_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        title = path.stem.replace("_", " ").title()
        author = "fixture"
        body_start = 0
        for idx, line in enumerate(lines[:10]):
            if line.startswith("Title: "):
                title = line[7:].strip()
            elif line.startswith("Author: "):
                author = line[8:].strip()
            elif line.startswith("---"):
                body_start = idx + 1
                break
        chapters = [
            c.strip()
            for c in re.split(r"^#\s*CHAPTER", "\n".join(lines[body_start:]), flags=re.M)
            if c.strip()
        ]
        if not chapters:
            chapters = [text]
        books.append(
            Book(book_id=path.stem, title=title, author=author, chapters=chapters, source="fixture")
        )
    return books


def download_gutenberg(book_id: str, settings: Settings, mirror: str | None = None) -> Path:
    base = mirror or "https://www.gutenberg.org"
    url = f"{base}/cache/epub/{book_id}/pg{book_id}.txt"
    target = settings.raw_dir / f"pg{book_id}.txt"
    if target.exists():
        return target
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "stellar-index/0.1 (public-domain research assistant)"}
    with httpx.Client(follow_redirects=True, headers=headers, timeout=60) as client:
        response = client.get(url)
        response.raise_for_status()
        target.write_text(response.text, encoding="utf-8")
    return target


def load_gutenberg_book(path: Path, book_id: str, title: str | None = None, author: str | None = None) -> Book:
    text = path.read_text(encoding="utf-8")
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ]
    starts = [text.find(m) for m in start_markers if text.find(m) >= 0]
    ends = [text.find(m) for m in end_markers if text.find(m) >= 0]
    start = max(starts) if starts else 0
    end = min(ends) if ends else len(text)
    if start:
        start = text.find("\n", start) + 1
    body = text[start:end]
    chapters = [
        c.strip()
        for c in re.split(r"\n\s*(?:CHAPTER|Chapter)\s+[IVXLC0-9]+[.:]?", body)
        if len(c.strip()) > 200
    ]
    if not chapters:
        chapters = [body]
    return Book(
        book_id=book_id,
        title=title or path.stem.replace("pg", "Book "),
        author=author or "Project Gutenberg",
        chapters=chapters,
        source="gutenberg",
    )
