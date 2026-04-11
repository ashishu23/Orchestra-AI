from rag.chunker import chunk_text, chunk_pdf
from rag.qdrant_store import QdrantStore
from tools.search import _get_store


async def ingest_document(text: str, metadata: dict | None = None) -> dict:
    """
    Chunk raw text and upsert all chunks into Qdrant.
    Returns a summary of ingested chunks.
    """
    from config import settings

    store = _get_store()
    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        metadata=metadata or {},
    )
    for chunk in chunks:
        await store.upsert(
            doc_id=chunk["id"],
            text=chunk["text"],
            metadata=chunk["metadata"],
        )
    return {"ingested_chunks": len(chunks), "source": (metadata or {}).get("source", "unknown")}


async def ingest_pdf(path: str) -> dict:
    """
    Extract text from a PDF, chunk it, and upsert into Qdrant.
    Returns a summary of ingested chunks.
    """
    from config import settings

    store = _get_store()
    chunks = chunk_pdf(
        path,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    for chunk in chunks:
        await store.upsert(
            doc_id=chunk["id"],
            text=chunk["text"],
            metadata=chunk["metadata"],
        )
    return {"ingested_chunks": len(chunks), "source": path}
