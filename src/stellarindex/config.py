from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")
    results_dir: Path = Path("results")
    model: str = os.getenv("STELLAR_MODEL", "deepseek-v4-flash")
    api_key: str | None = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    base_url: str | None = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL"))
    temperature: float = 0.2
    max_tokens: int = 8_192
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    fast_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    child_tokens: int = 256
    parent_tokens: int = 600
    top_k_hybrid: int = 50
    rrf_k: int = 60
    dense_batch_size: int = 32
    rerank_batch_size: int = 16


def resolve_paths(settings: Settings) -> None:
    for name in ("data_dir", "raw_dir", "index_dir", "results_dir"):
        path = Path(getattr(settings, name))
        path.mkdir(parents=True, exist_ok=True)
