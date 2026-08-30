from __future__ import annotations

import argparse
import json
from pathlib import Path

from stellarindex.config import Settings
from stellarindex.corpus import load_gutenberg_book
from stellarindex.llm import DeepSeekClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/golden_qa.json")
    parser.add_argument("--output", default="data/golden_qa_curated.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    settings = Settings(data_dir=Path("data"), raw_dir=Path("data/raw"))
    client = DeepSeekClient(settings)
    items = json.loads(Path(args.input).read_text())
    out_path = Path(args.output)
    curated = json.loads(out_path.read_text()) if out_path.exists() else []

    for idx, item in enumerate(items):
        if idx < args.start:
            continue
        if args.limit and idx >= args.start + args.limit:
            break
        raw = settings.raw_dir / f"pg{item['book_id']}.txt"
        if not raw.exists():
            continue
        book = load_gutenberg_book(raw, item["book_id"])
        chapter_idx = max(1, min(int(item.get("evidence_chapter", 1)), len(book.chapters)))
        chapter = book.chapters[chapter_idx - 1]
        # Compact the chapter around evidence to bound prompt size.
        evidence = item["evidence"][0] if item["evidence"] else ""
        pos = chapter.lower().find(evidence.lower()[:40]) if evidence else -1
        window = chapter if pos < 0 else chapter[max(0, pos - 1200): pos + 1600]
        prompt = (
            "You are auditing a reading-comprehension evaluation item. Output strict JSON:\n"
            '{"keep": true, "clarity": 5, "answerability": 5, "answers": ["short key phrase"], '
            '"reason": "..."}\n'
            "Rules: keep=true only when the question is unambiguous, the accepted answers are correct, "
            "and the question is answerable from the chapter below. Normalize `answers` to short "
            "unambiguous key phrases (max 3).\n\n"
            f"<item>\n{json.dumps(item, ensure_ascii=False)}\n</item>\n\n"
            f"<chapter>\n{window}\n</chapter>"
        )
        reply = client.complete_json_reply(
            [
                {"role": "system", "content": "You are an evaluation-set auditor. Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        data = reply.data
        keep = bool(data.get("keep", False))
        clarity = int(data.get("clarity", 0) or 0)
        answerability = int(data.get("answerability", 0) or 0)
        if keep and clarity >= 4 and answerability >= 4:
            answers = [str(a) for a in data.get("answers", [])][:3]
            if answers:
                item["answers"] = answers
            item["curation"] = {
                "clarity": clarity,
                "answerability": answerability,
                "reason": str(data.get("reason", ""))[:300],
            }
            curated.append(item)
            print(f"{idx + 1}/{len(items)} KEEP {item['id']} clarity={clarity}", flush=True)
        else:
            print(f"{idx + 1}/{len(items)} DROP {item['id']} clarity={clarity} ans={answerability} reason={str(data.get('reason',''))[:120]}", flush=True)
        out_path.write_text(json.dumps(curated, indent=2, ensure_ascii=False) + "\n")

    print(f"curated={len(curated)} -> {out_path}")


if __name__ == "__main__":
    main()
