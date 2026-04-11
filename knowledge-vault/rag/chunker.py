import hashlib
import uuid
from typing import Iterator

import tiktoken
from pypdf import PdfReader


def _sliding_window(
    tokens: list[int], chunk_size: int, overlap: int
) -> Iterator[list[int]]:
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        yield tokens[start : start + chunk_size]
        if start + chunk_size >= len(tokens):
            break


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
) -> list[dict]:
    """Split text into overlapping token chunks. Returns list of chunk dicts."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    metadata = metadata or {}
    chunks = []
    for i, window in enumerate(_sliding_window(tokens, chunk_size, overlap)):
        chunk_text_str = enc.decode(window)
        chunk_id = str(uuid.UUID(
            hashlib.sha256(
                f"{metadata.get('source', '')}:{i}:{chunk_text_str[:64]}".encode()
            ).hexdigest()[:32]
        ))
        chunks.append(
            {
                "id": chunk_id,
                "text": chunk_text_str,
                "metadata": {**metadata, "chunk_index": i},
            }
        )
    return chunks


def chunk_pdf(
    path: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """Extract text from a PDF and split into chunks."""
    reader = PdfReader(path)
    pages_text = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n".join(pages_text)
    return chunk_text(
        full_text,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata={"source": path, "type": "pdf", "total_pages": len(reader.pages)},
    )
