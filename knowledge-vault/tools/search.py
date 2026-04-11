from rag.qdrant_store import QdrantStore
from rag.rrf import rrf_combine

# Injected at startup via knowledge-vault/main.py
_store: QdrantStore | None = None


def set_store(store: QdrantStore):
    global _store
    _store = store


def _get_store() -> QdrantStore:
    if _store is None:
        raise RuntimeError("QdrantStore has not been initialized")
    return _store


async def search_semantic(query: str, top_k: int = 5) -> list[dict]:
    """Pure dense-vector semantic search."""
    store = _get_store()
    return await store.search_vector(query, top_k)


async def search_keyword(query: str, top_k: int = 5) -> list[dict]:
    """Sparse keyword (BM25-style) search."""
    store = _get_store()
    return await store.search_keyword(query, top_k)


async def search_hybrid(
    query: str, top_k: int = 5, alpha: float = 0.6
) -> list[dict]:
    """
    Hybrid search using Reciprocal Rank Fusion of dense + sparse results.

    alpha is reserved for future score blending; RRF handles fusion by default.
    """
    store = _get_store()
    vector_hits = await store.search_vector(query, top_k * 2)
    keyword_hits = await store.search_keyword(query, top_k * 2)
    fused = rrf_combine([vector_hits, keyword_hits], k=60)
    return fused[:top_k]
