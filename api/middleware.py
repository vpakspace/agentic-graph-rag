"""API middleware: rate limiting, request-id, metrics."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# ---------------------------------------------------------------------------
# X-Request-ID
# ---------------------------------------------------------------------------

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID to every request/response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Simple in-memory rate limiter (sliding window, per IP)
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter: max_requests per window_seconds."""

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health and metrics endpoints
        if request.url.path in ("/api/v1/health", "/api/v1/metrics"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self.window

        # Prune old entries
        hits = self._hits[client_ip]
        self._hits[client_ip] = [t for t in hits if t > cutoff]
        hits = self._hits[client_ip]

        if len(hits) >= self.max_requests:
            retry_after = int(hits[0] - cutoff) + 1
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)
        return await call_next(request)


# ---------------------------------------------------------------------------
# Metrics collector (lightweight, no external deps)
# ---------------------------------------------------------------------------

class MetricsCollector:
    """In-process metrics: request counts, latency, errors."""

    def __init__(self):
        self.request_count: int = 0
        self.error_count: int = 0
        self.latency_sum: float = 0.0
        self.latency_max: float = 0.0
        self._by_endpoint: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "errors": 0, "latency_sum": 0.0},
        )

    def record(self, path: str, status: int, duration: float) -> None:
        self.request_count += 1
        self.latency_sum += duration
        self.latency_max = max(self.latency_max, duration)
        ep = self._by_endpoint[path]
        ep["count"] += 1
        ep["latency_sum"] += duration
        if status >= 400:
            self.error_count += 1
            ep["errors"] += 1

    def snapshot(self) -> dict:
        avg = (self.latency_sum / self.request_count) if self.request_count else 0
        endpoints = {}
        for path, ep in self._by_endpoint.items():
            ep_avg = (ep["latency_sum"] / ep["count"]) if ep["count"] else 0
            endpoints[path] = {
                "count": ep["count"],
                "errors": ep["errors"],
                "avg_latency_ms": round(ep_avg, 1),
            }
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "avg_latency_ms": round(avg, 1),
            "max_latency_ms": round(self.latency_max, 1),
            "endpoints": endpoints,
        }


# Singleton metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect request count, latency, and error metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        _metrics.record(request.url.path, response.status_code, duration_ms)
        return response
