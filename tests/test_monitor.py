"""Tests for agentic_graph_rag.optimization.monitor."""

import time

from agentic_graph_rag.optimization.monitor import QueryMonitor

# ---------------------------------------------------------------------------
# record_query + get_stats
# ---------------------------------------------------------------------------

class TestRecordQuery:
    def test_single_query(self):
        m = QueryMonitor()
        m.record_query("simple", "vector_search", 0.5)
        s = m.get_stats()
        assert s["total_queries"] == 1
        assert s["query_types"] == {"simple": 1}
        assert s["tools_used"] == {"vector_search": 1}
        assert s["avg_latency"] == 0.5
        assert s["avg_retries"] == 0.0

    def test_multiple_queries(self):
        m = QueryMonitor()
        m.record_query("simple", "vector_search", 0.3)
        m.record_query("relation", "cypher_traverse", 0.7, retries=1)
        m.record_query("simple", "vector_search", 0.5)

        s = m.get_stats()
        assert s["total_queries"] == 3
        assert s["query_types"] == {"simple": 2, "relation": 1}
        assert s["tools_used"] == {"vector_search": 2, "cypher_traverse": 1}
        assert s["avg_latency"] == round(0.5, 4)
        assert s["avg_retries"] == round(1 / 3, 2)

    def test_tool_avg_latency(self):
        m = QueryMonitor()
        m.record_query("simple", "vector_search", 0.2)
        m.record_query("simple", "vector_search", 0.4)
        m.record_query("relation", "cypher_traverse", 1.0)

        s = m.get_stats()
        assert s["tool_avg_latency"]["vector_search"] == 0.3
        assert s["tool_avg_latency"]["cypher_traverse"] == 1.0

    def test_retries_recorded(self):
        m = QueryMonitor()
        m.record_query("multi_hop", "hybrid_search", 1.0, retries=2)
        m.record_query("simple", "vector_search", 0.3, retries=0)

        s = m.get_stats()
        assert s["avg_retries"] == 1.0


# ---------------------------------------------------------------------------
# track context manager
# ---------------------------------------------------------------------------

class TestTrack:
    def test_auto_records(self):
        m = QueryMonitor()
        with m.track("simple", "vector_search"):
            time.sleep(0.01)

        s = m.get_stats()
        assert s["total_queries"] == 1
        assert s["query_types"] == {"simple": 1}
        assert s["avg_latency"] > 0

    def test_ctx_retries(self):
        m = QueryMonitor()
        with m.track("relation", "cypher_traverse") as ctx:
            ctx["retries"] = 2

        s = m.get_stats()
        assert s["avg_retries"] == 2.0

    def test_default_retries_zero(self):
        m = QueryMonitor()
        with m.track("simple", "vector_search"):
            pass

        s = m.get_stats()
        assert s["avg_retries"] == 0.0

    def test_records_even_on_exception(self):
        m = QueryMonitor()
        try:
            with m.track("simple", "vector_search"):
                raise ValueError("boom")
        except ValueError:
            pass

        assert m.get_stats()["total_queries"] == 1


# ---------------------------------------------------------------------------
# get_stats edge cases
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_empty_stats(self):
        m = QueryMonitor()
        s = m.get_stats()
        assert s["total_queries"] == 0
        assert s["avg_latency"] == 0.0
        assert s["avg_retries"] == 0.0
        assert s["query_types"] == {}
        assert s["tools_used"] == {}
        assert s["tool_avg_latency"] == {}


# ---------------------------------------------------------------------------
# suggest_pagerank_weights
# ---------------------------------------------------------------------------

class TestSuggestPageRankWeights:
    def test_no_data(self):
        m = QueryMonitor()
        result = m.suggest_pagerank_weights()
        assert result["suggestion"] == "Not enough data"
        assert result["adjustments"] == {}

    def test_relation_heavy(self):
        m = QueryMonitor()
        for _ in range(6):
            m.record_query("relation", "cypher_traverse", 0.5)
        for _ in range(4):
            m.record_query("simple", "vector_search", 0.3)

        result = m.suggest_pagerank_weights()
        assert "pagerank_damping" in result["adjustments"]

    def test_global_heavy(self):
        m = QueryMonitor()
        for _ in range(4):
            m.record_query("global", "community_search", 0.5)
        for _ in range(6):
            m.record_query("simple", "vector_search", 0.3)

        result = m.suggest_pagerank_weights()
        assert "skeleton_beta" in result["adjustments"]

    def test_simple_heavy(self):
        m = QueryMonitor()
        for _ in range(8):
            m.record_query("simple", "vector_search", 0.3)
        for _ in range(2):
            m.record_query("relation", "cypher_traverse", 0.5)

        result = m.suggest_pagerank_weights()
        assert "top_k_vector" in result["adjustments"]

    def test_temporal_heavy(self):
        m = QueryMonitor()
        for _ in range(3):
            m.record_query("temporal", "temporal_query", 0.8)
        for _ in range(7):
            m.record_query("simple", "vector_search", 0.3)

        result = m.suggest_pagerank_weights()
        assert "max_hops" in result["adjustments"]

    def test_high_retry_rate(self):
        m = QueryMonitor()
        for _ in range(4):
            m.record_query("simple", "vector_search", 0.3, retries=1)
        for _ in range(6):
            m.record_query("simple", "vector_search", 0.3, retries=0)

        result = m.suggest_pagerank_weights()
        assert "relevance_threshold" in result["adjustments"]

    def test_no_adjustments_balanced(self):
        m = QueryMonitor()
        m.record_query("simple", "vector_search", 0.3)
        m.record_query("relation", "cypher_traverse", 0.5)
        m.record_query("global", "community_search", 0.4)
        m.record_query("temporal", "temporal_query", 0.6)
        m.record_query("multi_hop", "hybrid_search", 0.8)

        result = m.suggest_pagerank_weights()
        assert result["adjustments"] == {}

    def test_query_distribution_present(self):
        m = QueryMonitor()
        m.record_query("simple", "vector_search", 0.3)
        m.record_query("relation", "cypher_traverse", 0.5)

        result = m.suggest_pagerank_weights()
        assert "query_distribution" in result
        assert result["query_distribution"]["simple"] == 0.5
        assert result["query_distribution"]["relation"] == 0.5


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_clears_everything(self):
        m = QueryMonitor()
        m.record_query("simple", "vector_search", 0.3)
        m.record_query("relation", "cypher_traverse", 0.5, retries=1)
        m.reset()

        s = m.get_stats()
        assert s["total_queries"] == 0
        assert s["avg_latency"] == 0.0
        assert s["query_types"] == {}

    def test_can_record_after_reset(self):
        m = QueryMonitor()
        m.record_query("simple", "vector_search", 0.3)
        m.reset()
        m.record_query("relation", "cypher_traverse", 0.7)

        s = m.get_stats()
        assert s["total_queries"] == 1
        assert s["query_types"] == {"relation": 1}
