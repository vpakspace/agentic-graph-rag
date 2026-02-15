"""Tests for declarative routing rules (routing.mg)."""
from __future__ import annotations

from pathlib import Path

from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

_RULES_DIR = str(Path(__file__).parent.parent / "agentic_graph_rag" / "reasoning" / "rules")


class TestRoutingRules:
    def test_relation_query(self):
        """Query with relation keyword routes to cypher_traverse."""
        engine = ReasoningEngine(_RULES_DIR)
        result = engine.classify_query("What is the relationship between A and B?")
        assert result is not None
        assert result["tool"] == "cypher_traverse"

    def test_multi_hop_query(self):
        """Query with multi-hop keyword routes to cypher_traverse."""
        engine = ReasoningEngine(_RULES_DIR)
        result = engine.classify_query("Show the path through the graph")
        assert result is not None
        assert result["tool"] == "cypher_traverse"

    def test_global_query(self):
        """Query asking for all/overview routes to full_document_read."""
        engine = ReasoningEngine(_RULES_DIR)
        result = engine.classify_query("Give me an overview of all topics")
        assert result is not None
        assert result["tool"] == "full_document_read"

    def test_temporal_query(self):
        """Query about time/dates routes to temporal_query."""
        engine = ReasoningEngine(_RULES_DIR)
        result = engine.classify_query("When did this change happen? Show timeline")
        assert result is not None
        assert result["tool"] == "temporal_query"

    def test_simple_default(self):
        """Query with no special keywords defaults to vector_search."""
        engine = ReasoningEngine(_RULES_DIR)
        result = engine.classify_query("What is quantum computing?")
        assert result is not None
        assert result["tool"] == "vector_search"

    def test_multiple_keywords(self):
        """Query with keywords from multiple categories matches one."""
        engine = ReasoningEngine(_RULES_DIR)
        result = engine.classify_query("Show the connection path between nodes")
        assert result is not None
        # Has both relation ("connection") and multi_hop ("path") keywords
        assert result["tool"] == "cypher_traverse"

    def test_confidence_returned(self):
        """Result always contains the query string."""
        engine = ReasoningEngine(_RULES_DIR)
        result = engine.classify_query("связь между сущностями")
        assert result is not None
        assert "query" in result
        assert result["query"] == "связь между сущностями"

    def test_bilingual_ru(self):
        """Russian keywords work: связь → relation → cypher_traverse."""
        engine = ReasoningEngine(_RULES_DIR)
        result = engine.classify_query("покажи связь между узлами")
        assert result is not None
        assert result["tool"] == "cypher_traverse"

    def test_bilingual_temporal_ru(self):
        """Russian temporal keyword: когда → temporal → temporal_query."""
        engine = ReasoningEngine(_RULES_DIR)
        result = engine.classify_query("когда это произошло")
        assert result is not None
        assert result["tool"] == "temporal_query"
