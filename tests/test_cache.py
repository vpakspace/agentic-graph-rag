"""Tests for agentic_graph_rag.optimization.cache."""

import time

from rag_core.models import GraphContext

from agentic_graph_rag.optimization.cache import (
    CommunityCache,
    SubgraphCache,
    cache_key,
)

# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic(self):
        emb = [0.1, 0.2, 0.3]
        assert cache_key(emb) == cache_key(emb)

    def test_different_embeddings(self):
        k1 = cache_key([0.1, 0.2])
        k2 = cache_key([0.3, 0.4])
        assert k1 != k2

    def test_precision_rounding(self):
        k1 = cache_key([0.12345], precision=3)
        k2 = cache_key([0.12349], precision=3)
        assert k1 == k2  # both round to 0.123

    def test_returns_hex_string(self):
        k = cache_key([1.0, 0.0])
        assert isinstance(k, str)
        assert len(k) == 32  # md5 hex digest

    def test_empty_embedding(self):
        k = cache_key([])
        assert isinstance(k, str)


# ---------------------------------------------------------------------------
# SubgraphCache
# ---------------------------------------------------------------------------

class TestSubgraphCache:
    def test_put_and_get(self):
        cache = SubgraphCache(max_size=10)
        ctx = GraphContext(passages=["Hello"])
        cache.put("k1", ctx)
        assert cache.get("k1") == ctx

    def test_miss(self):
        cache = SubgraphCache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = SubgraphCache(max_size=2)
        cache.put("a", GraphContext(passages=["A"]))
        cache.put("b", GraphContext(passages=["B"]))
        cache.put("c", GraphContext(passages=["C"]))
        # "a" should be evicted (LRU)
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_access_refreshes_lru(self):
        cache = SubgraphCache(max_size=2)
        cache.put("a", GraphContext(passages=["A"]))
        cache.put("b", GraphContext(passages=["B"]))
        # Access "a" to make it recently used
        cache.get("a")
        # Now add "c" — "b" should be evicted (least recently used)
        cache.put("c", GraphContext(passages=["C"]))
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_ttl_expiry(self):
        cache = SubgraphCache(max_size=10, ttl=0.1)
        cache.put("k1", GraphContext(passages=["A"]))
        assert cache.get("k1") is not None
        time.sleep(0.15)
        assert cache.get("k1") is None  # expired

    def test_ttl_zero_disables(self):
        cache = SubgraphCache(max_size=10, ttl=0)
        cache.put("k1", GraphContext(passages=["A"]))
        # With ttl=0, no expiry check
        assert cache.get("k1") is not None

    def test_invalidate(self):
        cache = SubgraphCache()
        cache.put("k1", GraphContext())
        assert cache.invalidate("k1") is True
        assert cache.get("k1") is None
        assert cache.invalidate("k1") is False

    def test_clear(self):
        cache = SubgraphCache()
        cache.put("k1", GraphContext())
        cache.put("k2", GraphContext())
        cache.get("k1")
        cache.clear()
        assert cache.size == 0
        assert cache.hit_rate == 0.0

    def test_stats(self):
        cache = SubgraphCache(max_size=10, ttl=300)
        cache.put("k1", GraphContext())
        cache.get("k1")  # hit
        cache.get("k2")  # miss

        s = cache.stats()
        assert s["size"] == 1
        assert s["max_size"] == 10
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
        assert s["ttl"] == 300

    def test_hit_rate_no_queries(self):
        cache = SubgraphCache()
        assert cache.hit_rate == 0.0

    def test_update_existing_key(self):
        cache = SubgraphCache()
        cache.put("k1", GraphContext(passages=["V1"]))
        cache.put("k1", GraphContext(passages=["V2"]))
        result = cache.get("k1")
        assert result is not None
        assert result.passages == ["V2"]
        assert cache.size == 1

    def test_max_size_minimum_one(self):
        cache = SubgraphCache(max_size=0)
        assert cache.max_size == 1


# ---------------------------------------------------------------------------
# CommunityCache
# ---------------------------------------------------------------------------

class TestCommunityCache:
    def test_put_and_get(self):
        cache = CommunityCache()
        cache.put("comm1", "Summary of community 1")
        assert cache.get("comm1") == "Summary of community 1"

    def test_miss(self):
        cache = CommunityCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = CommunityCache(ttl=0.1)
        cache.put("c1", "Summary")
        assert cache.get("c1") is not None
        time.sleep(0.15)
        assert cache.get("c1") is None

    def test_invalidate(self):
        cache = CommunityCache()
        cache.put("c1", "Summary")
        assert cache.invalidate("c1") is True
        assert cache.get("c1") is None
        assert cache.invalidate("c1") is False

    def test_clear(self):
        cache = CommunityCache()
        cache.put("c1", "A")
        cache.put("c2", "B")
        cache.clear()
        assert cache.size == 0

    def test_stats(self):
        cache = CommunityCache(ttl=3600)
        cache.put("c1", "A")
        cache.get("c1")  # hit
        cache.get("c2")  # miss

        s = cache.stats()
        assert s["size"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
        assert s["ttl"] == 3600

    def test_overwrite(self):
        cache = CommunityCache()
        cache.put("c1", "Old")
        cache.put("c1", "New")
        assert cache.get("c1") == "New"
        assert cache.size == 1
