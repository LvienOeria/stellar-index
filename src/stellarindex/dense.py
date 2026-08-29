from __future__ import annotations

from pathlib import Path

from .config import Settings


_DENSE_CACHE: dict[tuple[str, str], "DenseIndex"] = {}


class DenseIndex:
    """Optional LanceDB dense index. Kept isolated from the BM25-only path."""

    def __new__(cls, settings: Settings):
        key = (str(settings.index_dir), settings.embedding_model)
        if key not in _DENSE_CACHE:
            _DENSE_CACHE[key] = super().__new__(cls)
        return _DENSE_CACHE[key]

    def __init__(self, settings: Settings):
        if getattr(self, "_initialized", False):
            return
        self.settings = settings
        self._table = None
        self._encoder = None
        try:
            import lancedb  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            db = lancedb.connect(str(settings.index_dir / "lancedb"))
            table_name = "chunks"
            if table_name in db.table_names():
                self._table = db.open_table(table_name)
            else:
                raise RuntimeError("dense index not built; run `stellar-index build --dense` first")
            self._encoder = SentenceTransformer(settings.embedding_model)
            self._initialized = True
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"dense index unavailable: {exc}") from exc

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        if self._table is None or self._encoder is None:
            return []
        vec = self._encoder.encode([query], normalize_embeddings=True).tolist()[0]
        rows = self._table.search(vec).limit(top_k).to_list()
        return [(str(row["chunk_id"]), float(row.get("_distance") or 0.0)) for row in rows]


def build_dense_index(store: Any, settings: Settings) -> int:
    """Build the LanceDB table from chunks already present in IndexStore."""
    import lancedb  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    rows = store.conn.execute("SELECT chunk_id, text FROM chunks").fetchall()
    if not rows:
        raise RuntimeError("no chunks in store; build the BM25 index first")
    encoder = SentenceTransformer(settings.embedding_model)
    texts = [row["text"] for row in rows]
    vectors = encoder.encode(texts, batch_size=settings.dense_batch_size, normalize_embeddings=True)
    db = lancedb.connect(str(settings.index_dir / "lancedb"))
    data = [
        {"chunk_id": row["chunk_id"], "text": row["text"], "vector": vectors[i].tolist()}
        for i, row in enumerate(rows)
    ]
    db.create_table("chunks", data=data, mode="overwrite")
    return len(data)
