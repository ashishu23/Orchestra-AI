import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from agents.graph import build_graph
from agents.state import OrchestraState

# ── Prometheus metrics ──────────────────────────────────────────────────────────
RESEARCH_REQUESTS = Counter(
    "orchestra_research_requests_total", "Total research requests"
)
SELF_CORRECTIONS = Histogram(
    "orchestra_self_corrections",
    "Self-correction count per request",
    buckets=[0, 1, 2, 3, 4, 5],
)
VERIFIED_CHUNKS = Histogram(
    "orchestra_verified_chunks",
    "Verified context chunks per request",
    buckets=[0, 1, 2, 3, 5, 8, 13],
)

# ── App setup ───────────────────────────────────────────────────────────────────
graph = build_graph()
app = FastAPI(title="Orchestra AI Orchestrator", version="1.0.0")


class ResearchRequest(BaseModel):
    query: str
    stream: bool = False


def _initial_state(query: str) -> OrchestraState:
    return {
        "query": query,
        "messages": [],
        "plan": [],
        "current_step_index": 0,
        "retrieved_chunks": [],
        "execution_results": [],
        "verified_context": [],
        "hallucination_flags": [],
        "final_answer": None,
        "self_correction_count": 0,
        "status": "planning",
    }


@app.post("/research")
async def research(req: ResearchRequest):
    RESEARCH_REQUESTS.inc()
    initial_state = _initial_state(req.query)
    config = {"configurable": {"thread_id": "single"}}

    if req.stream:
        async def event_stream():
            async for event in graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    final_state = await graph.ainvoke(initial_state, config=config)
    corrections = final_state["self_correction_count"]
    chunks = len(final_state["verified_context"])
    SELF_CORRECTIONS.observe(corrections)
    VERIFIED_CHUNKS.observe(chunks)

    return {
        "answer": final_state["final_answer"],
        "status": final_state["status"],
        "self_corrections": corrections,
        "verified_chunks": chunks,
        "plan_steps": len(final_state["plan"]),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "orchestrator"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
