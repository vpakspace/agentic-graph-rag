"""FastAPI route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.deps import get_service

router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    text: str
    mode: str = "agent_pattern"
    lang: str = "ru"


class SearchRequest(BaseModel):
    text: str
    tool: str = "vector_search"


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
    qa = svc.query(req.text, mode=req.mode, lang=req.lang)
    return qa.model_dump()


@router.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    svc = get_service()
    trace = svc.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.model_dump()


@router.get("/graph/stats")
def graph_stats():
    svc = get_service()
    return svc.graph_stats()
