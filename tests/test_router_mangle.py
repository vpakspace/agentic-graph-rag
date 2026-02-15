"""Tests for Mangle integration in query router."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from rag_core.models import QueryType, RouterDecision

from agentic_graph_rag.agent.router import classify_query
from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

_RULES_DIR = str(Path(__file__).parent.parent / "agentic_graph_rag" / "reasoning" / "rules")


class TestMangleRouterIntegration:
    def test_mangle_router_used_when_available(self):
        """When ReasoningEngine is provided, Mangle rules drive classification."""
        engine = ReasoningEngine(_RULES_DIR)
        d = classify_query("What is the relationship between A and B?", reasoning=engine)
        assert d.query_type == QueryType.RELATION
        assert d.suggested_tool == "cypher_traverse"
        assert "mangle" in d.reasoning.lower() or "rule" in d.reasoning.lower()

    def test_fallback_to_patterns(self):
        """When Mangle returns no match, fall back to pattern matching."""
        engine = ReasoningEngine(_RULES_DIR)
        # Query with no Mangle keyword match but with a pattern match
        d = classify_query("Покажи историю изменений за 2024-01", reasoning=engine)
        # "истори" matches temporal keyword in routing.mg
        assert d.query_type == QueryType.TEMPORAL

    def test_no_mangle_backward_compat(self):
        """reasoning=None → existing pattern-based behavior unchanged."""
        d = classify_query("What is quantum computing?")
        assert d.query_type == QueryType.SIMPLE
        assert d.suggested_tool == "vector_search"

        d2 = classify_query("What is quantum computing?", reasoning=None)
        assert d2.query_type == d.query_type
        assert d2.suggested_tool == d.suggested_tool
