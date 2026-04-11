"""Unit tests for the generic MCP JSON-RPC 2.0 client."""
import json

import httpx
import pytest
import respx

from mcp.client import MCPClient, MCPError


BASE_URL = "http://vault-test:8001"


@pytest.mark.asyncio
@respx.mock
async def test_call_success():
    respx.post(f"{BASE_URL}/mcp").mock(
        return_value=httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "result": [{"id": "abc", "text": "test chunk"}],
            },
        )
    )
    async with MCPClient(BASE_URL) as client:
        result = await client.call("search_hybrid", {"query": "test", "top_k": 3})
    assert isinstance(result, list)
    assert result[0]["id"] == "abc"


@pytest.mark.asyncio
@respx.mock
async def test_call_raises_mcp_error():
    respx.post(f"{BASE_URL}/mcp").mock(
        return_value=httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "error": {"code": -32602, "message": "Invalid params"},
            },
        )
    )
    async with MCPClient(BASE_URL) as client:
        with pytest.raises(MCPError) as exc_info:
            await client.call("search_hybrid", {})
    assert exc_info.value.code == -32602
