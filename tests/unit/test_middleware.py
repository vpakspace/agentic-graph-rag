"""Tests for API middleware: rate limiting, request-id, metrics."""
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def client():
    """Create test client with middleware enabled."""
    from fastapi.testclient import TestClient

    from api.app import create_app

    svc = MagicMock()
    svc.health.return_value = {"status": "ok"}
    svc.graph_stats.return_value = {"nodes": 0, "edges": 0}

    app = create_app(service=svc)
    with TestClient(app) as c:
        yield c


def test_request_id_generated(client):
    resp = client.get("/api/v1/health")
    assert "X-Request-ID" in resp.headers
    # Should be a valid UUID4
    request_id = resp.headers["X-Request-ID"]
    assert len(request_id) == 36  # UUID format


def test_request_id_passthrough(client):
    custom_id = "my-custom-request-id"
    resp = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert resp.headers["X-Request-ID"] == custom_id


def test_metrics_endpoint(client):
    # Make a few requests first
    client.get("/api/v1/health")
    client.get("/api/v1/health")

    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] >= 2
    assert "avg_latency_ms" in data
    assert "endpoints" in data


def test_metrics_collector():
    from api.middleware import MetricsCollector

    mc = MetricsCollector()
    mc.record("/api/v1/health", 200, 10.5)
    mc.record("/api/v1/health", 200, 20.0)
    mc.record("/api/v1/query", 500, 100.0)

    snap = mc.snapshot()
    assert snap["total_requests"] == 3
    assert snap["total_errors"] == 1
    assert snap["max_latency_ms"] == 100.0
    assert "/api/v1/health" in snap["endpoints"]
    assert snap["endpoints"]["/api/v1/health"]["count"] == 2
    assert snap["endpoints"]["/api/v1/query"]["errors"] == 1


def test_rate_limit_allows_normal_traffic(client):
    # Health endpoint is exempt from rate limiting
    for _ in range(5):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
