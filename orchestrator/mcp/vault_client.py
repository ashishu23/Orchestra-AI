from .client import MCPClient


class VaultClient:
    """Typed wrapper for Knowledge-Vault MCP tools."""

    def __init__(self, client: MCPClient):
        self._c = client

    async def search_semantic(self, query: str, top_k: int = 5) -> list[dict]:
        return await self._c.call("search_semantic", {"query": query, "top_k": top_k})

    async def search_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.6) -> list[dict]:
        return await self._c.call("search_hybrid", {"query": query, "top_k": top_k, "alpha": alpha})

    async def search_keyword(self, query: str, top_k: int = 5) -> list[dict]:
        return await self._c.call("search_keyword", {"query": query, "top_k": top_k})

    async def ingest_document(self, text: str, metadata: dict) -> dict:
        return await self._c.call("ingest_document", {"text": text, "metadata": metadata})

    async def ingest_pdf(self, path: str) -> dict:
        return await self._c.call("ingest_pdf", {"path": path})
