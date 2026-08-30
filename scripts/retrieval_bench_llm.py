from __future__ import annotations

import gc
import json
from pathlib import Path

from stellarindex.config import Settings
from stellarindex.evals import _normalize, find_evidence_chunks, load_qa
from stellarindex.search import IndexStore, hybrid_search


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--rerank", default="full")
    args = parser.parse_args()

    settings = Settings(data_dir=Path("data"), raw_dir=Path("data/raw"), index_dir=Path("data/index"))
    store = IndexStore(Path("data/index/gutenberg.db"))
    qa = load_qa(Path("data/golden_qa.json"))
    out = Path("results/retrieval-llm-rows.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line).get("qa_id"))
    with out.open("a", encoding="utf-8") as fh:
        for idx, item in enumerate(qa):
            if idx < args.start:
                continue
            if args.limit and idx >= args.start + args.limit:
                break
            if item.id in done:
                continue
            gold_ids = set()
            for evidence in item.evidence:
                gold_ids.update(find_evidence_chunks(store, evidence, item.book_id))
            if not gold_ids:
                row = {"qa_id": item.id, "book_id": item.book_id, "error": "no-gold"}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                continue
            try:
                results = hybrid_search(
                    item.question,
                    store,
                    settings,
                    dense=args.dense,
                    rerank=args.rerank,
                    llm_client=__import__("stellarindex.llm", fromlist=["DeepSeekClient"]).DeepSeekClient(settings),
                    book_id=item.book_id,
                )
                ranked_ids = [r.chunk_id for r in results]
                rr = 0.0
                for rank, chunk_id in enumerate(ranked_ids[:10], start=1):
                    if chunk_id in gold_ids:
                        rr = 1.0 / rank
                        break
                row = {
                    "qa_id": item.id,
                    "book_id": item.book_id,
                    "gold": sorted(gold_ids),
                    "hit1": int(ranked_ids and ranked_ids[0] in gold_ids),
                    "recall5": len(set(ranked_ids[:5]) & gold_ids) / len(gold_ids),
                    "recall10": len(set(ranked_ids[:10]) & gold_ids) / len(gold_ids),
                    "precision5": len(set(ranked_ids[:5]) & gold_ids) / 5.0,
                    "mrr10": rr,
                    "rank": ([i + 1 for i, c in enumerate(ranked_ids[:10]) if c in gold_ids] or [None])[0],
                }
            except Exception as exc:  # noqa: BLE001
                row = {"qa_id": item.id, "book_id": item.book_id, "error": f"{type(exc).__name__}: {exc}"}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            gc.collect()
            print(f"{idx + 1}/{len(qa)} {item.id}", flush=True)
    store.close()


if __name__ == "__main__":
    main()
