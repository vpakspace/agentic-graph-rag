# v6 Design: Provenance + Typed API Contract

**Date**: 2026-02-17
**Status**: Approved
**Inspired by**: "Semantic Companion Layer" by Fanghua (Joshua) Yu (Feb 2026)

## Overview

Add full pipeline provenance (structured trace of every step) and a typed API
contract (FastAPI REST + MCP server) to Agentic Graph RAG. Streamlit becomes a
thin HTTP client instead of calling Python directly.

**Goal**: Formalize the semantic pipeline into a governed contract with
explainability, as described in the SCL architecture pattern.

## Architecture Decision

**Approach**: Monolith API — FastAPI + MCP in one process, shared resources.

```
┌─────────────────┐     HTTP      ┌─────────────────────────────┐
│  Streamlit UI   │──────────────▶│  FastAPI (:8507)             │
│  (:8506)        │  httpx        │  ├── /api/v1/query           │
│  thin client    │               │  ├── /api/v1/search          │
└─────────────────┘               │  ├── /api/v1/trace/{id}      │
                                  │  ├── /api/v1/health          │
┌─────────────────┐     MCP       │  ├── /api/v1/graph/stats     │
│  Claude Code    │──────────────▶│  └── MCP (SSE)               │
│  / AI Agents    │  JSON-RPC     │      ├── resolve_intent      │
└─────────────────┘               │      ├── search_graph        │
                                  │      └── explain_trace       │
                                  │                               │
                                  │  PipelineService (contract)   │
                                  │  ├── .query()                 │
                                  │  ├── .search()                │
                                  │  ├── .health()                │
                                  │  └── .get_trace()             │
                                  │                               │
                                  │  ┌─── Neo4j Driver Pool ───┐ │
                                  │  └─── OpenAI Client ───────┘ │
                                  └─────────────────────────────┘
```

**Key principle** (from SCL): Clients never talk to engines directly. The
contract is stable; engines are interchangeable implementation details.

## 1. Provenance Models

New Pydantic models in `rag_core/models.py`:

```python
class ToolStep(BaseModel):
    """One tool execution step in the pipeline."""
    tool_name: str                        # "vector_search", "cypher_traverse"
    results_count: int = 0
    relevance_score: float = 0.0          # from reflector (1-5 scale)
    duration_ms: int = 0
    query_used: str = ""                  # may be rephrased

class EscalationStep(BaseModel):
    """Tool-to-tool escalation record."""
    from_tool: str
    to_tool: str
    reason: str                           # "relevance 1.6 < threshold 2.0"
    rephrased_query: str = ""

class RouterStep(BaseModel):
    """Router classification result."""
    method: str                           # "pattern", "llm", "mangle"
    decision: RouterDecision
    duration_ms: int = 0
    rules_fired: list[str] = Field(default_factory=list)

class GeneratorStep(BaseModel):
    """Answer generation metadata."""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    confidence: float = 0.0
    completeness_check: bool | None = None

class PipelineTrace(BaseModel):
    """Full pipeline provenance artifact."""
    trace_id: str                         # "tr_" + UUID hex[:12]
    timestamp: str                        # ISO 8601
    query: str
    router_step: RouterStep | None = None
    tool_steps: list[ToolStep] = Field(default_factory=list)
    escalation_steps: list[EscalationStep] = Field(default_factory=list)
    generator_step: GeneratorStep | None = None
    total_duration_ms: int = 0
```

`QAResult` gains a new field: `trace: PipelineTrace | None = None`.

## 2. PipelineService (Internal Contract)

New file: `agentic_graph_rag/service.py`

```python
class PipelineService:
    """Typed contract for Agentic Graph RAG pipeline.

    All clients (FastAPI, MCP, Streamlit) go through this service.
    The service owns resource lifecycle (driver, client).
    """

    def __init__(self, driver, openai_client, reasoning=None):
        self._driver = driver
        self._client = openai_client
        self._reasoning = reasoning
        self._trace_cache: dict[str, PipelineTrace] = {}  # in-memory, bounded

    def query(self, text, mode="agent_pattern", lang="ru") -> QAResult:
        """Full pipeline: route → retrieve → generate.
        Returns QAResult with embedded PipelineTrace."""

    def search(self, text, tool="vector_search") -> tuple[list[SearchResult], PipelineTrace]:
        """Retrieval only (no generation)."""

    def health(self) -> dict:
        """Neo4j connectivity + graph node/edge counts."""

    def get_trace(self, trace_id: str) -> PipelineTrace | None:
        """Retrieve trace from in-memory cache."""
```

### Provenance collection

`retrieval_agent.self_correction_loop()` is refactored to accept and populate
a `PipelineTrace` object. Each tool call records a `ToolStep` with timing.
Each escalation records an `EscalationStep`. The reflector score is captured
in the `ToolStep.relevance_score`. Logging continues alongside structured trace.

### Trace cache

In-memory dict, bounded to 100 entries (LRU eviction). Traces are ephemeral —
this is sufficient for debugging and the Agent Trace tab.

## 3. FastAPI REST API

New directory: `api/`

```
api/
├── __init__.py
├── app.py          # FastAPI app + lifespan (init driver/client/service)
├── routes.py       # Endpoint handlers
├── mcp_server.py   # FastMCP tools
├── deps.py         # Dependency injection (get_service)
└── schemas.py      # Request/response models (if needed beyond rag_core)
```

### Endpoints

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/api/v1/query` | POST | `{text, mode?, lang?}` | `QAResult` (with trace) |
| `/api/v1/search` | POST | `{text, tool?}` | `{results: SearchResult[], trace}` |
| `/api/v1/trace/{trace_id}` | GET | — | `PipelineTrace` |
| `/api/v1/health` | GET | — | `{status, neo4j, graph_stats}` |
| `/api/v1/graph/stats` | GET | — | `{nodes, edges, phrase_nodes, passage_nodes}` |
| `/docs` | GET | — | Swagger UI (auto-generated) |

### Port

**8507** — next after Streamlit 8506.

### Lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create driver, client, service
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(...))
    client = OpenAI(api_key=cfg.openai.api_key)
    app.state.service = PipelineService(driver, client, reasoning)
    yield
    # Shutdown: close driver
    driver.close()
```

### Example request/response

```json
// POST /api/v1/query
{
  "text": "Какие компоненты системы KET-RAG?",
  "mode": "agent_mangle",
  "lang": "ru"
}

// Response 200
{
  "answer": "Система KET-RAG включает...",
  "confidence": 0.82,
  "sources": [
    {"chunk": {"id": "c_01", "content": "..."}, "score": 0.91, "rank": 1, "source": "vector"}
  ],
  "trace": {
    "trace_id": "tr_a1b2c3d4e5f6",
    "timestamp": "2026-02-17T14:30:00Z",
    "query": "Какие компоненты системы KET-RAG?",
    "router_step": {
      "method": "mangle",
      "decision": {
        "query_type": "global",
        "confidence": 0.7,
        "reasoning": "Mangle rule matched → comprehensive_search.",
        "suggested_tool": "comprehensive_search"
      },
      "duration_ms": 12,
      "rules_fired": ["routing.mg:global_query"]
    },
    "tool_steps": [
      {
        "tool_name": "comprehensive_search",
        "results_count": 12,
        "relevance_score": 3.2,
        "duration_ms": 850,
        "query_used": "Какие компоненты системы KET-RAG?"
      }
    ],
    "escalation_steps": [],
    "generator_step": {
      "model": "gpt-4o-mini",
      "prompt_tokens": 2100,
      "completion_tokens": 350,
      "confidence": 0.82,
      "completeness_check": true
    },
    "total_duration_ms": 1340
  }
}
```

## 4. MCP Server

3 tools via FastMCP, mounted on the same ASGI app:

| Tool | Description | Parameters |
|------|-------------|------------|
| `resolve_intent` | Full pipeline: query → answer + provenance | `query: str, mode?: str` |
| `search_graph` | Retrieval only (no generation) | `query: str, tool?: str` |
| `explain_trace` | Get provenance by trace_id | `trace_id: str` |

MCP transport: SSE (Server-Sent Events) over HTTP, compatible with Claude Code
`claude mcp add` configuration.

### Claude Code integration

```bash
claude mcp add agentic-graph-rag -s user -- \
  npx -y @anthropic-ai/mcp-proxy http://localhost:8507/mcp/sse
```

Or direct config in `~/.claude.json`:
```json
{
  "mcpServers": {
    "agentic-graph-rag": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-proxy", "http://localhost:8507/mcp/sse"]
    }
  }
}
```

## 5. Streamlit Changes

### Search & Q&A tab

Replace direct Python calls with HTTP:

```python
# Before (v5)
from agentic_graph_rag.agent.retrieval_agent import run
qa = run(query, driver, openai_client, ...)

# After (v6)
import httpx
resp = httpx.post("http://localhost:8507/api/v1/query", json={
    "text": query, "mode": mode, "lang": lang,
}, timeout=60.0)
qa_data = resp.json()
```

### Agent Trace tab

Renders `PipelineTrace` as a visual timeline:
- Router decision badge (method + query_type + confidence)
- Tool steps as cards (tool_name, results_count, relevance_score, duration)
- Escalation arrows between cards (reason, rephrased_query)
- Generator summary (model, tokens, confidence, completeness)
- Total duration bar

### Tabs NOT changed

- **Ingest** — stays as direct Python (heavy, with progress bar, file I/O)
- **Graph Explorer** — stays as direct Cypher queries
- **Benchmark** — stays as direct Python (batch execution)
- **Reasoning** — stays as direct Mangle testing
- **Settings** — stays as is

## 6. What Changes in Existing Code

### `retrieval_agent.py`

`self_correction_loop()` signature gains `trace: PipelineTrace` parameter.
Each tool call wraps execution with timing and appends `ToolStep`. Each
escalation appends `EscalationStep`. The function returns
`(results, retries, trace)` instead of `(results, retries)`.

`run()` creates `PipelineTrace` at the start, passes it through the pipeline,
and attaches it to `QAResult.trace`.

### `router.py`

`classify_query()` returns `RouterDecision` (unchanged), but `PipelineService`
wraps the call to measure timing and capture `rules_fired` from Mangle.

### `generator.py`

`generate_answer()` returns token usage from OpenAI response
(`response.usage.prompt_tokens`, `response.usage.completion_tokens`).

### `models.py`

Add 5 new model classes (ToolStep, EscalationStep, RouterStep, GeneratorStep,
PipelineTrace). Add `trace` field to `QAResult`.

## 7. Testing Strategy

| Category | What | Count (est.) |
|----------|------|-------------|
| Unit | PipelineTrace serialization/deserialization | ~10 |
| Unit | PipelineService with mock driver/client | ~15 |
| Unit | API request/response schemas | ~10 |
| Integration | API endpoints via TestClient | ~12 |
| Integration | MCP tools via FastMCP test client | ~5 |
| Regression | Benchmark v5 = 66/90 must hold | 1 run |
| **Total new** | | **~52** |

Combined with existing 398 tests → **~450 tests**.

## 8. Non-Goals (YAGNI)

- No persistent trace storage (database) — in-memory cache is sufficient
- No authentication on API — local development tool
- No rate limiting — single-user
- No GraphQL — REST + MCP is sufficient for our use case
- No Streamlit migration for Ingest/Benchmark/Reasoning tabs — only Search & Q&A
- No async rewrite of tools — they use synchronous Neo4j driver

## 9. Risks

| Risk | Mitigation |
|------|-----------|
| HTTP latency for Streamlit | Same machine, localhost — <1ms overhead |
| Breaking benchmark scores | Provenance is additive; retrieval logic unchanged |
| FastMCP + FastAPI mount issues | Tested pattern; fallback to separate process |
| Token overhead in trace | Trace is metadata only, not sent to LLM |

## 10. Dependencies

- `fastapi` + `uvicorn` — API server
- `fastmcp` — MCP server
- `httpx` — Streamlit HTTP client
- No new external services
