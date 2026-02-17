"""Tests for FastAPI routes."""
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_service():
    from rag_core.models import (
        Chunk,
        PipelineTrace,
        QAResult,
        QueryType,
        RouterDecision,
        RouterStep,
        SearchResult,
        ToolStep,
    )

    svc = MagicMock()
    trace = PipelineTrace(
        trace_id="tr_test123",
        timestamp="2026-02-17T00:00:00Z",
        query="test",
        router_step=RouterStep(
            method="pattern",
            decision=RouterDecision(
                query_type=QueryType.SIMPLE, confidence=0.5,
                reasoning="test", suggested_tool="vector_search",
            ),
        ),
        tool_steps=[ToolStep(tool_name="vector_search", results_count=3)],
        total_duration_ms=500,
    )
    qa = QAResult(
        answer="Test answer",
        query="test",
        sources=[SearchResult(chunk=Chunk(id="c1", content="text"), score=0.9, rank=1)],
        confidence=0.8,
        trace=trace,
    )
    svc.query.return_value = qa
    svc.health.return_value = {"status": "ok", "neo4j": "connected"}
    svc.get_trace.return_value = trace
    svc.graph_stats.return_value = {"nodes": 100, "edges": 200}
    return svc


@pytest.fixture
def client(mock_service):
    from fastapi.testclient import TestClient

    from api.app import create_app

    app = create_app(service=mock_service)
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query(client, mock_service):
    resp = client.post("/api/v1/query", json={"text": "test question"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer"
    assert data["trace"]["trace_id"] == "tr_test123"
    mock_service.query.assert_called_once()


def test_query_missing_text(client):
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422


def test_get_trace(client):
    resp = client.get("/api/v1/trace/tr_test123")
    assert resp.status_code == 200
    assert resp.json()["trace_id"] == "tr_test123"


def test_get_trace_not_found(client, mock_service):
    mock_service.get_trace.return_value = None
    resp = client.get("/api/v1/trace/tr_nonexistent")
    assert resp.status_code == 404


def test_graph_stats(client):
    resp = client.get("/api/v1/graph/stats")
    assert resp.status_code == 200
    assert resp.json()["nodes"] == 100
