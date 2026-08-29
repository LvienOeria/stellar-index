from __future__ import annotations

from pathlib import Path

import streamlit as st

from .config import Settings
from .corpus import load_fixture_books
from .qa import QABot
from .search import IndexStore


@st.cache_resource
def get_store(index_dir: str) -> IndexStore:
    store = IndexStore(Path(index_dir) / "fixtures.db")
    store.rebuild(load_fixture_books(), Settings(index_dir=Path(index_dir)))
    return store


def main() -> None:
    st.set_page_config(page_title="Stellar Index", layout="wide")
    st.title("✨ Stellar Index · 星海索引")
    st.caption("Citation-grounded deep reading for public-domain sci-fi. Every answer must point back to the text.")

    index_dir = st.sidebar.text_input("index dir", "data/index")
    store = get_store(index_dir)
    books = {b.book_id: b for b in load_fixture_books()}
    book_id = st.sidebar.selectbox("book", list(books))
    mode = st.sidebar.selectbox("mode", ["rag", "long-context", "self-route"])
    rerank = st.sidebar.selectbox("rerank", ["none", "full", "fast", "llm"])
    dense = st.sidebar.checkbox("dense retrieval", value=False)

    question = st.text_input("question", value="Who kept the observatory mirror polished for nineteen years?")
    settings = Settings(index_dir=Path(index_dir))
    bot = QABot(settings, store)
    if st.button("Ask") and question:
        with st.spinner("Reading the archive..."):
            if mode == "rag":
                answer = bot.ask_rag(question, dense=dense, rerank=rerank)
            elif mode == "long-context":
                answer = bot.ask_longctx(question, books[book_id])
            else:
                answer = bot.ask_self_route(question, books[book_id], dense=dense, rerank=rerank)
        st.markdown("### Answer")
        st.write(answer.answer)
        st.markdown("### Citations")
        for i, citation in enumerate(answer.citations, start=1):
            st.markdown(
                f"**[{i}] {citation.book_id} · chapter {citation.chapter_idx + 1}** — "
                f"*{citation.quote}*"
            )
        st.caption(
            f"mode={answer.mode} · tokens={answer.input_tokens}/{answer.output_tokens} · "
            f"cost=${answer.cost_usd:.6f} · {answer.wall_seconds:.2f}s"
        )


if __name__ == "__main__":
    main()
