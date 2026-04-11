# Orchestra AI

A distributed multi-agent system for high-fidelity technical research. Orchestra AI uses the **Model Context Protocol (MCP)** to decouple reasoning from tool execution, **LangGraph** to orchestrate a self-correcting Plan-and-Execute loop, and **Kubernetes** to deploy all components as isolated, scalable microservices.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Reference](#component-reference)
   - [Orchestrator](#orchestrator)
   - [Knowledge-Vault MCP](#knowledge-vault-mcp)
   - [Executive-Sandbox MCP](#executive-sandbox-mcp)
   - [Qdrant Vector DB](#qdrant-vector-db)
4. [Agentic Flow](#agentic-flow)
5. [MCP Protocol](#mcp-protocol)
6. [Project Structure](#project-structure)
7. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Local Development (Docker Compose)](#local-development-docker-compose)
   - [Kubernetes Deployment (Minikube)](#kubernetes-deployment-minikube)
8. [Ingesting Documents](#ingesting-documents)
9. [API Reference](#api-reference)
10. [Configuration](#configuration)
11. [Security Model](#security-model)
12. [Running Tests](#running-tests)
13. [KPIs & Observability](#kpis--observability)
14. [Technology Stack](#technology-stack)

---

## Overview

Orchestra AI is designed to answer deep technical research questions by:

1. **Planning** a multi-step retrieval strategy from a natural-language query.
2. **Executing** each step through MCP tool calls — either semantic search over a knowledge base or Python code execution to verify numerical claims.
3. **Verifying** every result with a Critic Agent that detects hallucinations and irrelevant responses.
4. **Self-correcting** by automatically escalating retrieval strategies (semantic → hybrid → keyword) when verification fails.
5. **Synthesizing** a final grounded answer from only the verified context.

Unlike standard RAG pipelines, the LLM never has direct database access. All retrieval and execution go through strictly typed MCP tool schemas, making every tool call auditable and isolated.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     axiom-mesh (K8s namespace)               │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              Agentic Orchestrator (:8000)           │     │
│  │                                                     │     │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │     │
│  │   │ Planner  │→ │Executor  │→ │  Critic / Synth  │  │     │
│  │   │  Agent   │  │  Agent   │  │      Agent       │  │     │
│  │   └──────────┘  └────┬─────┘  └──────────────────┘  │     │
│  │        ↑             │ JSON-RPC 2.0 (MCP)           │     │
│  └────────┼─────────────┼──────────────────────────────┘     │
│           │             │                                    │
│    ┌──────┴──────┐  ┌───┴──────────────────┐                 │
│    │  Knowledge  │  │  Executive-Sandbox   │                 │
│    │  Vault MCP  │  │       MCP            │                 │
│    │   (:8001)   │  │      (:8002)         │                 │
│    │             │  │                      │                 │
│    │  Hybrid     │  │  Python Interpreter  │                 │
│    │  Search     │  │  (no internet,       │                 │
│    │  (RRF)      │  │   resource-limited)  │                 │
│    └──────┬──────┘  └──────────────────────┘                 │
│           │                                                  │
│    ┌──────┴──────┐                                           │
│    │   Qdrant    │                                           │
│    │  (:6333)    │  StatefulSet + PVC                        │
│    │  dense+     │                                           │
│    │  sparse     │                                           │
│    └─────────────┘                                           │
└──────────────────────────────────────────────────────────────┘
```

The system follows a **Micro-Agentic Architecture**:

- The **Orchestrator** is the "brain" — it plans, coordinates, and synthesizes.
- The **MCP servers** are the "hands" — they do the actual work (search, execute) in isolated containers.
- The **LLM never touches the database directly** — only through MCP tool schemas.

---

## Component Reference

### Orchestrator

**Path:** `orchestrator/`  
**Port:** 8000  
**Image:** `orchestra-ai/orchestrator`

The central coordinator built with **LangGraph**. Exposes a FastAPI HTTP server with:

- `POST /research` — submit a query, receive a structured answer
- `GET /health` — liveness check
- `GET /metrics` — Prometheus metrics

Internally runs a compiled `StateGraph` with five nodes: `planner`, `executor`, `critic`, `correction`, and `synthesizer`. All agent state flows through the shared `OrchestraState` TypedDict.

**Key files:**

| File | Purpose |
|------|---------|
| `agents/state.py` | `OrchestraState` and `ResearchStep` TypedDicts — the single source of truth |
| `agents/graph.py` | Assembles the LangGraph `StateGraph` with conditional routing |
| `agents/planner.py` | Calls LLM to generate a JSON step plan |
| `agents/executor.py` | Dispatches plan steps to MCP servers |
| `agents/critic.py` | Evaluates results; drives correction and synthesis |
| `mcp/client.py` | Generic async JSON-RPC 2.0 HTTP client |
| `mcp/vault_client.py` | Typed wrapper for Knowledge-Vault tools |
| `mcp/sandbox_client.py` | Typed wrapper for Executive-Sandbox tools |
| `prompts/planner.md` | System prompt constraining plan format (max 6 steps, JSON only) |
| `prompts/critic.md` | System prompt for PASS/FAIL verdict with calibration rules |

---

### Knowledge-Vault MCP

**Path:** `knowledge-vault/`  
**Port:** 8001  
**Image:** `orchestra-ai/knowledge-vault`

A **FastMCP** server wrapping Qdrant. Provides hybrid retrieval by combining dense vector search (semantic) with sparse BM25-style keyword search, fused using **Reciprocal Rank Fusion (RRF)**.

**Exposed MCP tools:**

| Tool | Arguments | Description |
|------|-----------|-------------|
| `search_semantic` | `query: str, top_k: int` | Pure dense-vector cosine similarity search |
| `search_hybrid` | `query: str, top_k: int, alpha: float` | Dense + sparse results fused with RRF |
| `search_keyword` | `query: str, top_k: int` | Sparse TF-based keyword search |
| `ingest_document` | `text: str, metadata: dict` | Chunk raw text and upsert into Qdrant |
| `ingest_pdf` | `path: str` | Extract PDF text, chunk, and upsert |

**RAG pipeline:**

```
PDF / raw text
      ↓
  chunker.py       — token-based sliding window (512 tok, 64 tok overlap)
      ↓
  embedder.py      — Google text-embedding-004 or sentence-transformers
      ↓
  qdrant_store.py  — upsert with dense vector + sparse TF vector
      ↓
  search_hybrid    — dense hits + keyword hits → RRF → top-k results
```

**Key files:**

| File | Purpose |
|------|---------|
| `rag/chunker.py` | `chunk_pdf` / `chunk_text` using tiktoken cl100k_base |
| `rag/rrf.py` | Pure `rrf_combine` — score = Σ 1/(60 + rank_i) |
| `rag/embedder.py` | Async embedding with Google or sentence-transformers fallback |
| `rag/qdrant_store.py` | Qdrant CRUD: collection creation, upsert, dense/sparse search |
| `tools/search.py` | MCP tool implementations for all three search modes |
| `tools/ingest.py` | MCP tool implementations for document and PDF ingestion |

---

### Executive-Sandbox MCP

**Path:** `executive-sandbox/`  
**Port:** 8002  
**Image:** `orchestra-ai/executive-sandbox`

A hardened Python execution environment for **verifying numerical claims** retrieved from documents (e.g., recomputing statistics, validating benchmarks). It runs as a non-root user with all Linux capabilities dropped.

**Exposed MCP tools:**

| Tool | Arguments | Description |
|------|-----------|-------------|
| `execute_python` | `code: str` | Execute Python code, return stdout and error |

**Isolation model:**

Each execution spawns a **fresh child process** via `multiprocessing.Process`. Before running any user code, the child applies:

- `RLIMIT_CPU` — hard cap of 10 CPU seconds
- `RLIMIT_AS` — virtual memory cap of 128 MB
- Wall-clock timeout of 15 seconds (parent kills child if exceeded)
- Restricted builtins: `open`, `exec`, `eval`, `__import__`, `breakpoint`, `input` are blocked

At the Kubernetes layer, a `NetworkPolicy` enforces zero egress from sandbox pods — generated code cannot reach the internet under any circumstances.

**Key files:**

| File | Purpose |
|------|---------|
| `sandbox/limits.py` | Resource limit constants |
| `sandbox/runner.py` | `run_sandboxed(code)` — subprocess isolation with `resource.setrlimit` |
| `tools/execute.py` | `execute_python` MCP tool wrapping the runner |

---

### Qdrant Vector DB

**Image:** `qdrant/qdrant:v1.9.0`  
**Ports:** 6333 (REST), 6334 (gRPC)

Deployed as a **StatefulSet** with a PersistentVolumeClaim for durable storage. The `research_docs` collection uses:

- `dense` vectors — 768-dimensional, cosine distance (text-embedding-004 output)
- `sparse` vectors — for BM25-style keyword retrieval (Qdrant >= 1.7)

---

## Agentic Flow

```
User query
    │
    ▼
┌─────────┐
│ Planner │  LLM generates a JSON array of up to 6 ResearchSteps
└────┬────┘  (tool + arguments for each step)
     │
     ▼
┌──────────┐
│ Executor │  Dispatches current step to vault or sandbox MCP
└────┬─────┘
     │
     ▼
┌────────┐
│ Critic │  LLM evaluates result: PASS or FAIL
└────┬───┘
     │
     ├──[PASS, more steps remain]──────────────────→ Executor (next step)
     │
     ├──[PASS, all steps done]─────────────────────→ Synthesizer → Final answer
     │
     ├──[FAIL, retries remain]──→ Correction node ──→ Executor (retry with
     │                            (escalates tool:     different strategy)
     │                             semantic→hybrid
     │                             →keyword)
     │
     └──[FAIL, max retries (3)]──────────────────── → END (status: failed)
```

**Self-correction strategy escalation:**

| Previous tool | Next tool on FAIL |
|---------------|-------------------|
| `search_semantic` | `search_hybrid` |
| `search_hybrid` | `search_keyword` |
| `search_keyword` | `search_keyword` (last resort) |

The `self_correction_count` in `OrchestraState` tracks how many corrections occurred — exposed as a KPI in the API response and as a Prometheus histogram.

---

## MCP Protocol

All tool calls from the orchestrator to MCP servers use **JSON-RPC 2.0** over HTTP POST to the `/mcp` endpoint.

**Request format:**

```json
{
  "jsonrpc": "2.0",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "tools/call",
  "params": {
    "name": "search_hybrid",
    "arguments": {
      "query": "YOLOv11 inference latency benchmarks",
      "top_k": 5
    }
  }
}
```

**Success response:**

```json
{
  "jsonrpc": "2.0",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "result": [
    {
      "id": "abc123",
      "text": "YOLOv11 achieves 7ms latency on A100...",
      "score": 0.94,
      "rrf_score": 0.031,
      "source": "yolo_benchmark_2024.pdf",
      "chunk_index": 12
    }
  ]
}
```

**Error response:**

```json
{
  "jsonrpc": "2.0",
  "id": "...",
  "error": { "code": -32602, "message": "Invalid params: top_k must be > 0" }
}
```

---

## Project Structure

```
Orchestra AI/
├── .env.example                        # All environment variable documentation
├── .gitignore
├── docker-compose.yml                  # Full local dev stack
│
├── orchestrator/                       # LangGraph orchestrator (port 8000)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                         # FastAPI: POST /research, GET /health, GET /metrics
│   ├── config.py                       # pydantic-settings Settings
│   ├── agents/
│   │   ├── state.py                    # OrchestraState TypedDict
│   │   ├── planner.py                  # Researcher Agent
│   │   ├── executor.py                 # Executor Agent
│   │   ├── critic.py                   # Critic + Correction + Synthesizer Agents
│   │   └── graph.py                    # StateGraph assembly
│   ├── mcp/
│   │   ├── client.py                   # Generic JSON-RPC 2.0 client
│   │   ├── vault_client.py             # Knowledge-Vault typed wrapper
│   │   └── sandbox_client.py           # Executive-Sandbox typed wrapper
│   ├── prompts/
│   │   ├── planner.md
│   │   └── critic.md
│   └── tests/
│       ├── test_graph.py               # Happy path, correction loop, max-retry tests
│       └── test_mcp_client.py
│
├── knowledge-vault/                    # RAG MCP service (port 8001)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── config.py
│   ├── rag/
│   │   ├── chunker.py                  # PDF/text → overlapping token chunks
│   │   ├── rrf.py                      # Reciprocal Rank Fusion
│   │   ├── embedder.py                 # Google / sentence-transformers
│   │   └── qdrant_store.py             # Qdrant async CRUD + sparse vectors
│   ├── tools/
│   │   ├── search.py                   # search_semantic / hybrid / keyword
│   │   └── ingest.py                   # ingest_document / ingest_pdf
│   └── tests/
│       └── test_search.py              # RRF unit tests, chunker overlap tests
│
├── executive-sandbox/                  # Code execution MCP service (port 8002)
│   ├── Dockerfile                      # Non-root user, all capabilities dropped
│   ├── requirements.txt
│   ├── main.py
│   ├── config.py
│   ├── sandbox/
│   │   ├── limits.py                   # CPU/mem/time constants
│   │   └── runner.py                   # Subprocess isolation + resource.setrlimit
│   ├── tools/
│   │   └── execute.py                  # execute_python MCP tool
│   └── tests/
│       └── test_execute.py             # Timeout, memory, stdout, error tests
│
├── ingestion/
│   └── ingest_cli.py                   # CLI for bulk PDF ingestion
│
├── helm/
│   └── orchestra/
│       ├── Chart.yaml
│       ├── values.yaml                 # Production defaults
│       ├── values.dev.yaml             # Minikube resource overrides
│       └── templates/
│           ├── namespace.yaml          # axiom-mesh namespace
│           ├── secrets.yaml            # API keys as K8s Secret
│           ├── orchestrator/           # Deployment + Service
│           ├── knowledge-vault/        # Deployment + Service
│           ├── executive-sandbox/      # Deployment + Service + HPA
│           ├── qdrant/                 # StatefulSet + Service + PVC
│           └── network-policy/
│               ├── sandbox-egress-deny.yaml   # BLOCKS all sandbox egress
│               └── internal-allow.yaml        # Intra-namespace communication
│
└── scripts/
    ├── setup_minikube.sh
    ├── build_images.sh
    ├── deploy.sh
    └── port_forward.sh
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- (For Kubernetes) Minikube or K3s, kubectl, Helm 3

### Local Development (Docker Compose)

**1. Clone and configure environment:**

```bash
cd "Orchestra AI"
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
GEMINI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here     # same key, used for embeddings
# Or if using Anthropic:
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here
```

**2. Start the full stack:**

```bash
docker compose up --build
```

This starts four services in dependency order:

```
qdrant (healthy) → knowledge-vault (healthy) → executive-sandbox (healthy) → orchestrator
```

**3. Verify all services are running:**

```bash
curl http://localhost:8000/health   # orchestrator
curl http://localhost:8001/health   # knowledge-vault
curl http://localhost:8002/health   # executive-sandbox
curl http://localhost:6333/healthz  # qdrant
```

**4. Ingest some documents:**

```bash
python ingestion/ingest_cli.py --pdf ./your-paper.pdf --vault-url http://localhost:8001
```

**5. Run a research query:**

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the inference latency of YOLOv11 on an A100 GPU?"}'
```

**Example response:**

```json
{
  "answer": "According to retrieved benchmarks, YOLOv11 achieves 7ms inference latency on an NVIDIA A100 GPU at batch size 1...",
  "status": "complete",
  "self_corrections": 0,
  "verified_chunks": 3,
  "plan_steps": 3
}
```

**Streaming mode** (Server-Sent Events):

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "stream": true}'
```

---

### Kubernetes Deployment (Minikube)

**1. Start Minikube and enable addons:**

```bash
bash scripts/setup_minikube.sh
```

**2. Point Docker to Minikube's registry:**

```bash
eval $(minikube docker-env)
```

**3. Build all service images:**

```bash
bash scripts/build_images.sh
```

**4. Deploy with Helm:**

```bash
GEMINI_API_KEY=your_key GOOGLE_API_KEY=your_key bash scripts/deploy.sh
```

This runs:

```bash
helm upgrade --install orchestra ./helm/orchestra \
  --namespace axiom-mesh \
  --create-namespace \
  --values helm/orchestra/values.dev.yaml \
  --set secrets.geminiApiKey=...
```

**5. Port-forward all services:**

```bash
bash scripts/port_forward.sh
```

**6. Check deployment status:**

```bash
kubectl get all -n axiom-mesh
kubectl get networkpolicies -n axiom-mesh
```

---

## Ingesting Documents

Use the CLI tool to ingest PDFs before running research queries:

```bash
# Single PDF
python ingestion/ingest_cli.py --pdf ./papers/benchmark.pdf

# Entire directory
python ingestion/ingest_cli.py --pdf ./papers/ --vault-url http://localhost:8001

# Against K8s deployment (after port-forwarding)
python ingestion/ingest_cli.py --pdf ./papers/ --vault-url http://localhost:8001
```

You can also ingest programmatically via the MCP endpoint directly:

```bash
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tools/call",
    "params": {
      "name": "ingest_document",
      "arguments": {
        "text": "YOLOv11 achieves 7ms latency on A100 at FP16 precision.",
        "metadata": {"source": "manual", "topic": "benchmarks"}
      }
    }
  }'
```

---

## API Reference

### `POST /research`

Submit a research query.

**Request body:**

```json
{
  "query": "string",
  "stream": false
}
```

**Response (non-streaming):**

```json
{
  "answer": "string | null",
  "status": "complete | failed",
  "self_corrections": 0,
  "verified_chunks": 3,
  "plan_steps": 4
}
```

**Response (streaming, `stream: true`):** Server-Sent Events stream of LangGraph node events.

### `GET /health`

```json
{"status": "ok", "service": "orchestrator"}
```

### `GET /metrics`

Prometheus metrics endpoint. Tracked metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `orchestra_research_requests_total` | Counter | Total requests received |
| `orchestra_self_corrections` | Histogram | Corrections per request |
| `orchestra_verified_chunks` | Histogram | Verified chunks per request |

---

## Configuration

All configuration is via environment variables (loaded by pydantic-settings from `.env`).

### Orchestrator

| Variable | Default | Description |
|----------|---------|-------------|
| `VAULT_URL` | `http://localhost:8001` | Knowledge-Vault MCP base URL |
| `SANDBOX_URL` | `http://localhost:8002` | Executive-Sandbox MCP base URL |
| `LLM_PROVIDER` | `google` | `google` or `anthropic` |
| `LLM_MODEL` | `gemini-2.0-flash` | Model name for the selected provider |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `MAX_RETRIES` | `3` | Max self-correction attempts before failing |
| `MCP_TIMEOUT` | `30.0` | Per-MCP-call timeout in seconds |

### Knowledge-Vault

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint |
| `QDRANT_API_KEY` | — | Qdrant API key (optional) |
| `EMBEDDING_PROVIDER` | `google` | `google` or `sentence-transformers` |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model identifier |
| `GOOGLE_API_KEY` | — | Required if using Google embeddings |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |

### Executive-Sandbox

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CPU_SECONDS` | `10` | Hard CPU time limit per execution |
| `MAX_MEMORY_MB` | `128` | Virtual memory cap per execution |
| `TIMEOUT_SECONDS` | `15` | Wall-clock timeout per execution |

---

## Security Model

Orchestra AI follows a **defence-in-depth** approach across three layers:

### 1. MCP Tool Access (No Direct DB Access)

The LLM orchestrator never connects to Qdrant directly. All retrieval passes through the Knowledge-Vault MCP's typed tool schema. This means:

- Tool arguments are Pydantic-validated before reaching the database.
- Invalid queries return a JSON-RPC `-32602` error — no raw exceptions.
- The LLM cannot perform arbitrary database operations (delete, overwrite collections).

### 2. Sandbox Isolation (No Network, Resource-Limited)

The `execute_python` tool runs user-generated code with multiple layers of protection:

| Layer | Mechanism |
|-------|-----------|
| Process isolation | `multiprocessing.Process` with `fork` context — separate memory space |
| CPU limit | `resource.RLIMIT_CPU = 10s` — OOM-kills runaway loops |
| Memory limit | `resource.RLIMIT_AS = 128MB` — prevents memory exhaustion |
| Wall-clock timeout | Parent kills child after 15s regardless |
| Restricted builtins | `open`, `exec`, `eval`, `__import__`, `input`, `breakpoint` all removed |
| Non-root user | Container runs as UID 1000 with no Linux capabilities |

### 3. Kubernetes NetworkPolicy

The most critical control: **the sandbox pod cannot reach the internet**.

```yaml
# helm/orchestra/templates/network-policy/sandbox-egress-deny.yaml
spec:
  podSelector:
    matchLabels:
      app: executive-sandbox
  policyTypes: ["Egress"]
  egress: []   # Empty = deny ALL outbound traffic
```

This prevents:
- Generated code calling external APIs or exfiltrating data
- The sandbox bypassing the MCP schema to reach Qdrant or the LLM directly

---

## Running Tests

Each service has its own test suite. Install dependencies first:

```bash
pip install pytest pytest-asyncio respx
```

**Orchestrator tests** (graph state machine):

```bash
cd orchestrator
pytest tests/ -v
```

Tests cover:
- Happy path: planner → executor → critic PASS → synthesizer
- Correction loop: critic FAILs 2× then PASSes, verifying `self_correction_count == 2`
- Max retry exhaustion: critic always FAILs, verifying terminal state

**Knowledge-Vault tests** (pure functions):

```bash
cd knowledge-vault
pytest tests/ -v
```

Tests cover:
- `rrf_combine`: items appearing in both lists rank higher; correct score ordering
- `chunk_text`: overlap behavior, metadata propagation, chunk ID uniqueness

**Executive-Sandbox tests** (subprocess execution):

```bash
cd executive-sandbox
pytest tests/ -v
```

Tests cover:
- Successful execution + stdout capture
- Syntax errors and runtime errors
- Timeout kill (infinite loop)
- Math/statistics computation

---

## KPIs & Observability

Three primary KPIs are tracked per request and exposed via Prometheus:

| KPI | Formula | Where |
|-----|---------|-------|
| **Retrieval Precision** | `verified_chunks / total_retrieved` | Critic node |
| **Execution Success Rate** | `successful sandbox calls / total` | Executor node |
| **Self-Correction Rate** | corrections / steps | `OrchestraState.self_correction_count` |

Access metrics at `http://localhost:8000/metrics` (after deployment).

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph 0.2+, LangChain Core |
| LLM | Google Gemini 2.0 Flash / Anthropic Claude (configurable) |
| MCP Server | FastMCP 0.4+ |
| MCP Transport | JSON-RPC 2.0 over HTTP (httpx async client) |
| Vector DB | Qdrant 1.9 (dense + sparse vectors) |
| Embeddings | Google text-embedding-004 (768d) or sentence-transformers |
| PDF Processing | pypdf + tiktoken (cl100k_base) |
| Web Framework | FastAPI + uvicorn |
| Containerization | Docker (python:3.11-slim base) |
| Orchestration | Kubernetes (Minikube / K3s) + Helm 3 |
| Autoscaling | Kubernetes HPA (sandbox: 1→5 replicas on CPU) |
| Observability | Prometheus (prometheus-client) |
| Testing | pytest + pytest-asyncio + respx |
