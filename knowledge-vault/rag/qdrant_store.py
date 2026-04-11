from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    NamedVector,
    NamedSparseVector,
    SearchRequest,
    Filter,
)

from .embedder import Embedder


class QdrantStore:
    """
    Qdrant async store supporting dense vector search and sparse (BM25-style)
    keyword search. Requires Qdrant >= 1.7 for sparse vectors.
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        dense_dim: int,
        embedder: Embedder,
        api_key: str | None = None,
    ):
        self._client = AsyncQdrantClient(url=url, api_key=api_key or None)
        self._collection = collection_name
        self._dense_dim = dense_dim
        self._embedder = embedder

    async def ensure_collection(self):
        """Create the collection if it does not exist."""
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    "dense": VectorParams(
                        size=self._dense_dim, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )

    async def upsert(
        self,
        doc_id: str,
        text: str,
        metadata: dict,
    ):
        """Embed text and upsert a point with both dense and sparse vectors."""
        dense_vec = await self._embedder.embed_one(text)
        sparse_vec = self._build_sparse(text)

        await self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=doc_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": sparse_vec,
                    },
                    payload={"text": text, **metadata},
                )
            ],
        )

    async def search_vector(self, query: str, top_k: int) -> list[dict]:
        """Dense vector (semantic) search."""
        query_vec = await self._embedder.embed_one(query)
        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=NamedVector(name="dense", vector=query_vec),
            limit=top_k,
        )
        return [{"id": str(h.id), "score": h.score, **h.payload} for h in hits]

    async def search_keyword(self, query: str, top_k: int) -> list[dict]:
        """Sparse (BM25-style) keyword search."""
        sparse_vec = self._build_sparse(query)
        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(
                    indices=sparse_vec.indices, values=sparse_vec.values
                ),
            ),
            limit=top_k,
        )
        return [{"id": str(h.id), "score": h.score, **h.payload} for h in hits]

    def _build_sparse(self, text: str) -> SparseVector:
        """
        Simple TF-based sparse vector. Each unique token maps to a hash-based
        index; value is normalized term frequency. Suitable for BM25-style
        retrieval without external tokenizer dependencies.
        """
        tokens = text.lower().split()
        freq: dict[int, float] = {}
        for token in tokens:
            idx = abs(hash(token)) % 100_000
            freq[idx] = freq.get(idx, 0.0) + 1.0
        total = sum(freq.values()) or 1.0
        indices = list(freq.keys())
        values = [v / total for v in freq.values()]
        return SparseVector(indices=indices, values=values)
