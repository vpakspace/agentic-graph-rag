"""FastAPI route handlers."""

from __future__ import annotations

from typing import Literal, get_args

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agentic_graph_rag.service import TOOL_NAMES
from api.deps import get_service

router = APIRouter(prefix="/api/v1")

VALID_MODES = Literal[
    "vector", "cypher", "hybrid",
    "agent_pattern", "agent_llm", "agent_mangle",
]
VALID_TOOLS = Literal[
    "vector_search", "cypher_traverse",
    "hybrid_search", "temporal_query", "comprehensive_search",
    "full_document_read",
]

# Runtime guard: routes must stay in sync with the service tool list.
assert set(get_args(VALID_TOOLS)) == set(TOOL_NAMES), (
    f"VALID_TOOLS {set(get_args(VALID_TOOLS))} != TOOL_NAMES {set(TOOL_NAMES)}"
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    mode: VALID_MODES = "agent_pattern"


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    tool: VALID_TOOLS = "vector_search"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    svc = get_service()
    return svc.health()


@router.post("/query")
def query(req: QueryRequest):
    svc = get_service()
    qa = svc.query(req.text, mode=req.mode)
    return qa.model_dump()


@router.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    svc = get_service()
    trace = svc.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.model_dump()


@router.post("/search")
def search(req: SearchRequest):
    svc = get_service()
    results = svc.search(req.text, tool=req.tool)
    return [r.model_dump() for r in results]


@router.get("/graph/stats")
def graph_stats():
    svc = get_service()
    return svc.graph_stats()


@router.get("/metrics")
def metrics():
    from api.middleware import get_metrics
    return get_metrics().snapshot()
