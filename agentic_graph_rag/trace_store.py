"""Trace storage backends: in-memory (default) and optional Redis.

Usage:
    store = create_trace_store()          # auto-selects based on REDIS_URL
    store.put(trace)
    trace = store.get("tr_abc123")
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict

from rag_core.models import PipelineTrace

logger = logging.getLogger(__name__)

_DEFAULT_MAX = 100
_REDIS_TTL = 3600  # 1 hour


class TraceStore(ABC):
    """Abstract trace storage interface."""

    @abstractmethod
    def put(self, trace: PipelineTrace) -> None: ...

    @abstractmethod
    def get(self, trace_id: str) -> PipelineTrace | None: ...


class InMemoryTraceStore(TraceStore):
    """Bounded in-memory LRU trace cache (default)."""

    def __init__(self, max_size: int = _DEFAULT_MAX):
        self._cache: OrderedDict[str, PipelineTrace] = OrderedDict()
        self._max = max_size

    def put(self, trace: PipelineTrace) -> None:
        self._cache[trace.trace_id] = trace
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def get(self, trace_id: str) -> PipelineTrace | None:
        return self._cache.get(trace_id)


class RedisTraceStore(TraceStore):
    """Redis-backed trace storage with TTL."""

    def __init__(self, url: str, ttl: int = _REDIS_TTL):
        import redis

        self._client = redis.from_url(url)
        self._ttl = ttl
        self._prefix = "agr:trace:"

    def put(self, trace: PipelineTrace) -> None:
        key = f"{self._prefix}{trace.trace_id}"
        self._client.setex(key, self._ttl, trace.model_dump_json())

    def get(self, trace_id: str) -> PipelineTrace | None:
        key = f"{self._prefix}{trace_id}"
        data = self._client.get(key)
        if data is None:
            return None
        return PipelineTrace.model_validate(json.loads(data))


def create_trace_store() -> TraceStore:
    """Factory: use Redis if REDIS_URL is set, otherwise in-memory."""
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            store = RedisTraceStore(redis_url)
            logger.info("Using Redis trace store at %s", redis_url)
            return store
        except Exception as e:
            logger.warning("Redis unavailable (%s), falling back to in-memory", e)
    return InMemoryTraceStore()
