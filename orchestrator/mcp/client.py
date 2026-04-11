import uuid
from typing import Any

import httpx


class MCPError(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"MCP error {code}: {message}")


class MCPClient:
    """Generic async JSON-RPC 2.0 client for MCP servers."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._url = base_url.rstrip("/") + "/mcp"
        self._client = httpx.AsyncClient(timeout=timeout)

    async def call(self, method_name: str, arguments: dict) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": method_name, "arguments": arguments},
        }
        resp = await self._client.post(self._url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise MCPError(data["error"]["code"], data["error"]["message"])
        return data["result"]

    async def list_tools(self) -> list[dict]:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list",
            "params": {},
        }
        resp = await self._client.post(self._url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise MCPError(data["error"]["code"], data["error"]["message"])
        return data["result"]

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
