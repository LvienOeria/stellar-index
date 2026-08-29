from stellarindex.corpus import chunk_book, load_fixture_books


def test_fixture_books_load():
    books = load_fixture_books()
    assert len(books) == 2
    for book in books:
        assert len(book.chapters) == 3
        assert book.full_text


def test_chunking_respects_chapter_and_parent():
    books = load_fixture_books()
    chunks = chunk_book(books[0], child_target=256, parent_target=600)
    assert chunks
    chapter_ids = {c.chapter_idx for c in chunks}
    assert chapter_ids == {0, 1, 2}
    for chunk in chunks:
        assert chunk.parent_id
        assert chunk.start_sentence <= chunk.end_sentence
        assert chunk.text.strip()
