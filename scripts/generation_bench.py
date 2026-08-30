from __future__ import annotations

import argparse
import json
from pathlib import Path

from stellarindex.config import Settings
from stellarindex.corpus import load_gutenberg_book
from stellarindex.evals import generation_metrics, load_qa
from stellarindex.qa import QABot
from stellarindex.search import IndexStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--rerank", default="none")
    args = parser.parse_args()

    settings = Settings(data_dir=Path("data"), raw_dir=Path("data/raw"), index_dir=Path("data/index"))
    store = IndexStore(Path("data/index/gutenberg.db"))
    qa = load_qa(Path("data/golden_qa.json"))
    bot = QABot(settings, store)
    book_cache = {}
    out = Path(f"results/generation-rows-{args.rerank}-{'dense' if args.dense else 'bm25'}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("a", encoding="utf-8") as fh:
        for idx, item in enumerate(qa):
            if idx < args.start:
                continue
            if args.limit and idx >= args.start + args.limit:
                break
            if item.book_id not in book_cache:
                raw = settings.raw_dir / f"pg{item.book_id}.txt"
                book_cache[item.book_id] = load_gutenberg_book(raw, item.book_id) if raw.exists() else None
            book = book_cache[item.book_id]
            if book is None:
                continue
            answer = bot.ask_rag(item.question, dense=args.dense, rerank=args.rerank)
            gen = generation_metrics(item, book, answer)
            row = {
                "qa_id": item.id,
                "book_id": item.book_id,
                "type": item.type,
                "mode": answer.mode,
                "generation": gen,
                "efficiency": {
                    "input_tokens": answer.input_tokens,
                    "output_tokens": answer.output_tokens,
                    "cost_usd": answer.cost_usd,
                    "wall_seconds": answer.wall_seconds,
                },
                "answer": answer.answer[:800],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{idx + 1}/{len(qa)} {item.id} ok={gen['correct']} cit={gen['citation_precision']:.2f}", flush=True)


if __name__ == "__main__":
    main()
