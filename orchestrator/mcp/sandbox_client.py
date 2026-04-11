from .client import MCPClient


class SandboxClient:
    """Typed wrapper for Executive-Sandbox MCP tools."""

    def __init__(self, client: MCPClient):
        self._c = client

    async def execute_python(self, code: str) -> dict:
        """
        Execute Python code in the sandbox.
        Returns: {"stdout": str, "error": str | None, "success": bool}
        """
        return await self._c.call("execute_python", {"code": code})
