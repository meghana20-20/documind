import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingest import chunk_text, clean_text


def test_chunk_text_respects_size():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert all(len(c) <= 200 + 50 for c in chunks)
    assert len(chunks) > 1


def test_chunk_text_overlap():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    # consecutive chunks should share some overlapping content
    assert chunks[0][-10:] in text
    assert len(chunks) >= 2


def test_clean_text_collapses_whitespace():
    dirty = "Hello\n\n\n\nWorld    Test"
    cleaned = clean_text(dirty)
    assert "\n\n\n" not in cleaned
    assert "    " not in cleaned


def test_chunk_text_raises_on_bad_overlap():
    try:
        chunk_text("hello world", chunk_size=10, overlap=10)
        assert False, "should have raised"
    except ValueError:
        pass
