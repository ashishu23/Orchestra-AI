import inspect
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tools.execute import execute_python

TOOLS = {
    "execute_python": execute_python,
}


async def _call_tool(name: str, arguments: dict):
    if name not in TOOLS:
        return None, {"code": -32601, "message": f"Tool not found: {name}"}
    try:
        fn = TOOLS[name]
        result = await fn(**arguments) if inspect.iscoroutinefunction(fn) else fn(**arguments)
        return result, None
    except Exception as e:
        return None, {"code": -32000, "message": str(e)}


app = FastAPI()


@app.post("/mcp")
async def mcp_handler(request: Request):
    body = await request.json()
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    if method == "tools/list":
        result = [{"name": name} for name in TOOLS]
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    if method == "tools/call":
        result, error = await _call_tool(params.get("name"), params.get("arguments", {}))
        if error:
            return {"jsonrpc": "2.0", "id": req_id, "error": error}
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "executive-sandbox"}
