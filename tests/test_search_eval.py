import shutil
from pathlib import Path

from stellarindex.config import Settings
from stellarindex.corpus import load_fixture_books
from stellarindex.evals import load_qa, retrieval_metrics
from stellarindex.search import IndexStore


def test_retrieval_metrics_on_fixtures(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", index_dir=tmp_path / "index")
    store = IndexStore(settings.index_dir / "fixtures.db")
    store.rebuild(load_fixture_books(), settings)
    qa = load_qa()
    metrics = retrieval_metrics(store, qa, settings)
    store.close()
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert metrics["recall10"] >= 0.9
    assert metrics["mrr10"] > 0.5
