"""Caching layer for Graph RAG retrieval results.

Provides LRU caches for subgraph traversal results and community summaries
to avoid redundant Neo4j queries and LLM calls.
"""

from __future__ import annotations

import hashlib
import logging
import struct
import time
from collections import OrderedDict
from typing import Any

from rag_core.models import GraphContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic cache key from embedding
# ---------------------------------------------------------------------------

def cache_key(query_embedding: list[float], precision: int = 4) -> str:
    """Generate a deterministic cache key from a query embedding.

    Rounds floats to ``precision`` decimals then hashes the byte representation
    for a compact, collision-resistant key.
    """
    rounded = [round(v, precision) for v in query_embedding]
    raw = struct.pack(f">{len(rounded)}f", *rounded)
    return hashlib.md5(raw).hexdigest()  # noqa: S324


# ---------------------------------------------------------------------------
# SubgraphCache — LRU cache for graph traversal results
# ---------------------------------------------------------------------------

class SubgraphCache:
    """LRU cache for GraphContext objects keyed by query embedding hash.

    Parameters
    ----------
    max_size : int
        Maximum number of entries (default 128).
    ttl : float
        Time-to-live in seconds (default 300 = 5 min).  Set to 0 to disable.
    """

    def __init__(self, max_size: int = 128, ttl: float = 300.0) -> None:
        self.max_size = max(1, max_size)
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple[GraphContext, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # -- public API ---------------------------------------------------------

    def get(self, key: str) -> GraphContext | None:
        """Return cached GraphContext or None on miss / expiry."""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        ctx, ts = entry
        if self.ttl > 0 and (time.monotonic() - ts) > self.ttl:
            del self._cache[key]
            self._misses += 1
            logger.debug("Cache expired: %s", key)
            return None

        self._cache.move_to_end(key)
        self._hits += 1
        return ctx

    def put(self, key: str, value: GraphContext) -> None:
        """Store a GraphContext in the cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = (value, time.monotonic())
        else:
            if len(self._cache) >= self.max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("Cache evicted: %s", evicted_key)
            self._cache[key] = (value, time.monotonic())

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry.  Returns True if it existed."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Drop all entries and reset counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    # -- stats --------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "ttl": self.ttl,
        }


# ---------------------------------------------------------------------------
# CommunityCache — cache for community / cluster summaries
# ---------------------------------------------------------------------------

class CommunityCache:
    """Simple dict cache for community summaries keyed by community id.

    Community summaries are expensive to generate (LLM call per community)
    but rarely change, so TTL is longer (default 1 hour).
    """

    def __init__(self, ttl: float = 3600.0) -> None:
        self.ttl = ttl
        self._cache: dict[str, tuple[str, float]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, community_id: str) -> str | None:
        entry = self._cache.get(community_id)
        if entry is None:
            self._misses += 1
            return None

        summary, ts = entry
        if self.ttl > 0 and (time.monotonic() - ts) > self.ttl:
            del self._cache[community_id]
            self._misses += 1
            return None

        self._hits += 1
        return summary

    def put(self, community_id: str, summary: str) -> None:
        self._cache[community_id] = (summary, time.monotonic())

    def invalidate(self, community_id: str) -> bool:
        if community_id in self._cache:
            del self._cache[community_id]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "ttl": self.ttl,
        }
