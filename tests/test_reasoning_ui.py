"""Tests for ReasoningEngine UI-facing methods (from_sources, rule_sources, get_strata)."""
from __future__ import annotations

from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine


_ROUTING_SRC = """\
keyword(/relation, "связ").
keyword(/relation, "connect").
keyword(/temporal, "when").
match(Query, Cat) :- keyword(Cat, Word), query_contains(Query, Word).
tool_for(/relation, "cypher_traverse").
tool_for(/temporal, "temporal_query").
route_to(Tool, Query) :- match(Query, Cat), tool_for(Cat, Tool).
route_to("vector_search", Query) :- current_query(Query), !match(Query, X).
"""

_ACCESS_SRC = """\
permit(/viewer, /read, /public).
permit(/admin, /write, /public).
"""


class TestFromSources:
    def test_creates_engine(self):
        engine = ReasoningEngine.from_sources({"routing": _ROUTING_SRC})
        assert isinstance(engine, ReasoningEngine)

    def test_classify_query(self):
        engine = ReasoningEngine.from_sources({"routing": _ROUTING_SRC})
        result = engine.classify_query("show connections between nodes")
        assert result is not None
        assert result["tool"] == "cypher_traverse"

    def test_no_match_fallback(self):
        engine = ReasoningEngine.from_sources({"routing": _ROUTING_SRC})
        result = engine.classify_query("what is machine learning")
        assert result is not None
        assert result["tool"] == "vector_search"

    def test_no_routing_source(self):
        engine = ReasoningEngine.from_sources({"access": _ACCESS_SRC})
        result = engine.classify_query("any query")
        assert result is None


class TestRuleSources:
    def test_returns_copy(self):
        engine = ReasoningEngine.from_sources({"routing": _ROUTING_SRC, "access": _ACCESS_SRC})
        sources = engine.rule_sources
        assert "routing" in sources
        assert "access" in sources
        # Must be a copy
        sources["extra"] = "test"
        assert "extra" not in engine.rule_sources


class TestGetStrata:
    def test_routing_strata(self):
        engine = ReasoningEngine.from_sources({"routing": _ROUTING_SRC})
        strata = engine.get_strata("routing")
        assert len(strata) >= 1
        # route_to uses negation of match → must be in later stratum
        all_preds = [p for s in strata for p in s]
        assert "route_to" in all_preds

    def test_missing_source(self):
        engine = ReasoningEngine.from_sources({})
        assert engine.get_strata("nonexistent") == []

    def test_simple_no_negation(self):
        src = 'parent(/a, /b). ancestor(X, Y) :- parent(X, Y).'
        engine = ReasoningEngine.from_sources({"simple": src})
        strata = engine.get_strata("simple")
        assert len(strata) >= 1
