from __future__ import annotations

import json
import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import Settings
from .corpus import download_gutenberg, load_fixture_books, load_gutenberg_book
from .dense import build_dense_index
from .evals import aggregate_results, load_qa, retrieval_metrics, run_eval
from .qa import QABot
from .search import IndexStore

console = Console()


def _settings(data_dir: str, model: str, env_file: str | None) -> Settings:
    if env_file and Path(env_file).exists():
        for raw in Path(env_file).read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"'))
    return Settings(
        data_dir=Path(data_dir),
        raw_dir=Path(data_dir) / "raw",
        index_dir=Path(data_dir) / "index",
        results_dir=Path(data_dir) / "results",
        model=model,
    )


def _fixture_store(settings: Settings) -> tuple[IndexStore, list]:
    books = load_fixture_books()
    store = IndexStore(settings.index_dir / "fixtures.db")
    store.rebuild(books, settings)
    return store, books


@click.group()
def main() -> None:
    """stellar-index: citation-grounded deep reading for public-domain sci-fi."""


@main.command("build")
@click.option("--fixtures", is_flag=True, help="Build the two synthetic fixture books instead of downloading.")
@click.option("--dense", is_flag=True, help="Also build the local LanceDB dense index (downloads bge-small-en).")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--model", default="deepseek-v4-flash", show_default=True)
@click.option("--env-file", default=".env", show_default=True)
def build_cmd(fixtures: bool, dense: bool, data_dir: str, model: str, env_file: str) -> None:
    settings = _settings(data_dir, model, env_file)
    if fixtures:
        store, books = _fixture_store(settings)
        console.print(f"[green]built[/green] {len(books)} fixture books")
    else:
        books_toml = Path("data/books.toml")
        if not books_toml.exists():
            raise click.ClickException("data/books.toml missing; use --fixtures or add Project Gutenberg ids")
        import tomllib

        spec = tomllib.loads(books_toml.read_text())
        books = []
        store = IndexStore(settings.index_dir / "gutenberg.db")
        for entry in spec.get("books", []):
            book_id = str(entry["gutenberg_id"])
            path = download_gutenberg(book_id, settings)
            books.append(load_gutenberg_book(path, book_id, entry.get("title"), entry.get("author")))
        store.rebuild(books, settings)
        console.print(f"[green]built[/green] {len(books)} Gutenberg books")
    if dense:
        n = build_dense_index(store, settings)
        console.print(f"[green]dense index built[/green] {n} vectors")
    store.close()


@main.command("ask")
@click.option("--book", "book_id", default="the_last_observatory", show_default=True)
@click.option("--question", required=True)
@click.option("--mode", type=click.Choice(["rag", "long-context", "self-route"]), default="rag", show_default=True)
@click.option("--dense", is_flag=True)
@click.option("--rerank", type=click.Choice(["none", "full", "fast", "llm"]), default="none", show_default=True)
@click.option("--fixtures", is_flag=True, help="Use the synthetic fixture corpus.")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--model", default="deepseek-v4-flash", show_default=True)
@click.option("--env-file", default=".env", show_default=True)
def ask_cmd(
    book_id: str,
    question: str,
    mode: str,
    dense: bool,
    rerank: str,
    fixtures: bool,
    data_dir: str,
    model: str,
    env_file: str,
) -> None:
    settings = _settings(data_dir, model, env_file)
    store, books = _fixture_store(settings) if fixtures else (IndexStore(settings.index_dir / "gutenberg.db"), [])
    if fixtures:
        book = next(b for b in books if b.book_id == book_id)
    else:
        rows = store.conn.execute("SELECT * FROM books WHERE book_id=?", (book_id,)).fetchall()
        if not rows:
            raise click.ClickException(f"unknown book {book_id}; run build first")
        raw = settings.raw_dir / f"pg{book_id}.txt"
        book = load_gutenberg_book(raw, book_id, rows[0]["title"], rows[0]["author"])
    bot = QABot(settings, store)
    if mode == "rag":
        answer = bot.ask_rag(question, dense=dense, rerank=rerank)
    elif mode == "long-context":
        answer = bot.ask_longctx(question, book)
    else:
        answer = bot.ask_self_route(question, book, dense=dense, rerank=rerank)
    console.print(f"[bold]Answer ({answer.mode})[/bold]: {answer.answer}")
    console.print(answer.citation_text())
    console.print(
        f"tokens={answer.input_tokens}/{answer.output_tokens} cost=${answer.cost_usd:.6f} "
        f"seconds={answer.wall_seconds:.2f}"
    )
    store.close()


@main.command("eval")
@click.option("--fixtures", is_flag=True, help="Evaluate the 10-question fixture QA set.")
@click.option("--retrieval-only", is_flag=True, help="Skip LLM generation; only compute retrieval metrics.")
@click.option("--dense", is_flag=True)
@click.option("--rerank", type=click.Choice(["none", "full", "fast", "llm"]), default="none", show_default=True)
@click.option("--modes", default="rag", show_default=True, help="Comma-separated: rag,long-context,self-route")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--model", default="deepseek-v4-flash", show_default=True)
@click.option("--env-file", default=".env", show_default=True)
def eval_cmd(
    fixtures: bool,
    retrieval_only: bool,
    dense: bool,
    rerank: str,
    modes: str,
    data_dir: str,
    model: str,
    env_file: str,
) -> None:
    settings = _settings(data_dir, model, env_file)
    if fixtures:
        store, books = _fixture_store(settings)
        qa = load_qa()
    else:
        store = IndexStore(settings.index_dir / "gutenberg.db")
        qa = load_qa(Path("data/golden_qa.json"))
        books = []
    retrieval = retrieval_metrics(store, qa, settings, dense=dense, rerank=rerank)
    console.print_json(json.dumps({"retrieval": retrieval}))
    if retrieval_only:
        store.close()
        return
    mode_list = tuple(m.strip() for m in modes.split(",") if m.strip())
    results = run_eval(settings, qa, books, store, dense=dense, rerank=rerank, modes=mode_list)
    summary = aggregate_results(results)
    out = settings.results_dir / "fixture-eval.json" if fixtures else settings.results_dir / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"retrieval": retrieval, "summary": summary}, indent=2))
    console.print_json(json.dumps({"retrieval": retrieval, "summary": summary}))
    store.close()


if __name__ == "__main__":
    main()
