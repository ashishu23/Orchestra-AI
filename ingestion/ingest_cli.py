#!/usr/bin/env python3
"""
Offline ingestion CLI: process PDF files and ingest them into the Knowledge-Vault MCP.

Usage:
    python ingest_cli.py --pdf ./docs/paper.pdf
    python ingest_cli.py --pdf ./docs/ --vault-url http://localhost:8001
"""
import argparse
import asyncio
import json
import pathlib
import uuid

import httpx

try:
    from pypdf import PdfReader
except ImportError:
    raise SystemExit(
        "pypdf is required: pip install pypdf"
    )


def extract_pdf_text(path: pathlib.Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )


async def ingest_document(vault_url: str, text: str, metadata: dict) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {"name": "ingest_document", "arguments": {"text": text, "metadata": metadata}},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{vault_url}/mcp", json=payload)
        resp.raise_for_status()
        data = resp.json()
    if "error" in data:
        raise RuntimeError(f"MCP error: {data['error']}")
    return data["result"]


async def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Orchestra AI Knowledge-Vault")
    parser.add_argument("--pdf", required=True, help="Path to a PDF file or directory of PDFs")
    parser.add_argument("--vault-url", default="http://localhost:8001", help="Knowledge-Vault base URL")
    args = parser.parse_args()

    pdf_path = pathlib.Path(args.pdf)
    pdfs = list(pdf_path.glob("**/*.pdf")) if pdf_path.is_dir() else [pdf_path]

    if not pdfs:
        print(f"No PDF files found at: {pdf_path}")
        return

    print(f"Found {len(pdfs)} PDF(s) to ingest into {args.vault_url}")
    total_chunks = 0

    for pdf in pdfs:
        print(f"  Ingesting: {pdf.name} ...", end=" ", flush=True)
        text = extract_pdf_text(pdf)
        metadata = {"source": pdf.name, "path": str(pdf.absolute())}
        result = await ingest_document(args.vault_url, text, metadata)
        chunks = result.get("ingested_chunks", 0)
        total_chunks += chunks
        print(f"{chunks} chunks")

    print(f"\nDone. Total chunks ingested: {total_chunks}")


if __name__ == "__main__":
    asyncio.run(main())
