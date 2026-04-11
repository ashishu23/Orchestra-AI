"""Unit tests for RRF and chunker — pure functions, no external deps."""
import pytest

from rag.rrf import rrf_combine
from rag.chunker import chunk_text


def test_rrf_combine_merges_and_ranks():
    list_a = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
    list_b = [{"id": "b", "score": 0.95}, {"id": "c", "score": 0.7}]

    result = rrf_combine([list_a, list_b], k=60)
    ids = [r["id"] for r in result]

    # "b" appears in both lists, should rank highest
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_rrf_combine_single_list():
    items = [{"id": str(i), "score": float(i)} for i in range(5)]
    result = rrf_combine([items], k=60)
    assert [r["id"] for r in result] == [str(i) for i in range(5)]


def test_rrf_scores_attached():
    items = [{"id": "x", "score": 0.5}]
    result = rrf_combine([items])
    assert "rrf_score" in result[0]


def test_chunk_text_basic():
    text = " ".join(["word"] * 600)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "id" in chunk
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["chunk_index"] >= 0


def test_chunk_text_overlap():
    text = "A B C D E F G H I J"
    chunks = chunk_text(text, chunk_size=3, overlap=1)
    # With overlap, chunks should share tokens at boundaries
    assert len(chunks) >= 2


def test_chunk_text_metadata_preserved():
    chunks = chunk_text("hello world " * 50, metadata={"source": "test.pdf", "type": "pdf"})
    for chunk in chunks:
        assert chunk["metadata"]["source"] == "test.pdf"
        assert chunk["metadata"]["type"] == "pdf"
