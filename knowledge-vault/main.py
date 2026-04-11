import inspect
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from config import settings
from rag.embedder import Embedder
from rag.qdrant_store import QdrantStore
from tools.search import set_store, search_semantic, search_hybrid, search_keyword
from tools.ingest import ingest_document, ingest_pdf

TOOLS = {
    "search_semantic": search_semantic,
    "search_hybrid": search_hybrid,
    "search_keyword": search_keyword,
    "ingest_document": ingest_document,
    "ingest_pdf": ingest_pdf,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = Embedder(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        api_key=settings.google_api_key or None,
    )
    store = QdrantStore(
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
        dense_dim=settings.dense_dim,
        embedder=embedder,
        api_key=settings.qdrant_api_key or None,
    )
    await store.ensure_collection()
    set_store(store)
    yield


app = FastAPI(lifespan=lifespan)


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
    return {"status": "ok", "service": "knowledge-vault"}
