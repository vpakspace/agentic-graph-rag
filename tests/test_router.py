"""Tests for agentic_graph_rag.agent.router."""

from unittest.mock import MagicMock, patch

from rag_core.models import QueryType, RouterDecision

from agentic_graph_rag.agent.router import (
    classify_query,
    classify_query_by_llm,
    classify_query_by_patterns,
)

# ---------------------------------------------------------------------------
# classify_query_by_patterns
# ---------------------------------------------------------------------------

class TestClassifyByPatterns:
    def test_simple_query(self):
        d = classify_query_by_patterns("What is Python?")
        assert d.query_type == QueryType.SIMPLE
        assert d.suggested_tool == "vector_search"

    def test_relation_query_en(self):
        d = classify_query_by_patterns("What is the relationship between A and B?")
        assert d.query_type == QueryType.RELATION
        assert d.suggested_tool == "cypher_traverse"

    def test_relation_query_ru(self):
        d = classify_query_by_patterns("Какая связь между Python и ML?")
        assert d.query_type == QueryType.RELATION

    def test_multi_hop_query(self):
        d = classify_query_by_patterns("Compare Python and Java through their ecosystems")
        assert d.query_type == QueryType.MULTI_HOP
        assert d.suggested_tool == "cypher_traverse"

    def test_multi_hop_chain(self):
        d = classify_query_by_patterns("Покажи цепочку зависимостей через модули")
        assert d.query_type == QueryType.MULTI_HOP

    def test_global_query(self):
        d = classify_query_by_patterns("Show all entities in the system")
        assert d.query_type == QueryType.GLOBAL
        assert d.suggested_tool == "comprehensive_search"

    def test_global_query_ru(self):
        d = classify_query_by_patterns("Покажи все сущности")
        assert d.query_type == QueryType.GLOBAL

    def test_temporal_query(self):
        d = classify_query_by_patterns("When was Python created?")
        assert d.query_type == QueryType.TEMPORAL
        assert d.suggested_tool == "temporal_query"

    def test_temporal_date_pattern(self):
        d = classify_query_by_patterns("What happened in 2024-01?")
        assert d.query_type == QueryType.TEMPORAL

    def test_temporal_query_ru(self):
        d = classify_query_by_patterns("Когда был создан этот документ?")
        assert d.query_type == QueryType.TEMPORAL

    def test_confidence_increases_with_matches(self):
        d1 = classify_query_by_patterns("relationship")
        d2 = classify_query_by_patterns("relationship between connected entities")
        assert d2.confidence >= d1.confidence

    def test_confidence_capped(self):
        d = classify_query_by_patterns(
            "связь отношения соединяет between link related connected"
        )
        assert d.confidence <= 0.95

    def test_returns_router_decision(self):
        d = classify_query_by_patterns("test query")
        assert isinstance(d, RouterDecision)
        assert d.reasoning != ""

    def test_empty_query(self):
        d = classify_query_by_patterns("")
        assert d.query_type == QueryType.SIMPLE


# ---------------------------------------------------------------------------
# classify_query_by_llm
# ---------------------------------------------------------------------------

class TestClassifyByLLM:
    def test_llm_classification(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "relation"
        client.chat.completions.create.return_value = resp

        d = classify_query_by_llm("How are A and B related?", openai_client=client)
        assert d.query_type == QueryType.RELATION
        assert d.confidence == 0.85

    def test_llm_returns_multi_hop(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "multi_hop"
        client.chat.completions.create.return_value = resp

        d = classify_query_by_llm("test", openai_client=client)
        assert d.query_type == QueryType.MULTI_HOP

    def test_llm_unknown_type_defaults_simple(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "unknown_type"
        client.chat.completions.create.return_value = resp

        d = classify_query_by_llm("test", openai_client=client)
        assert d.query_type == QueryType.SIMPLE

    def test_llm_error_fallback_patterns(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")

        d = classify_query_by_llm("Show all items", openai_client=client)
        # Falls back to patterns → global
        assert d.query_type == QueryType.GLOBAL

    @patch("agentic_graph_rag.agent.router.get_settings")
    def test_creates_client_when_none(self, mock_settings):
        cfg = MagicMock()
        cfg.openai.api_key = "test-key"
        cfg.openai.base_url = ""
        cfg.openai.llm_model_mini = "gpt-4o-mini"
        mock_settings.return_value = cfg

        with patch("agentic_graph_rag.agent.router.make_openai_client") as mock_make:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "simple"
            mock_client.chat.completions.create.return_value = resp
            mock_make.return_value = mock_client

            d = classify_query_by_llm("test query")
            mock_make.assert_called_once_with(cfg)
            assert d.query_type == QueryType.SIMPLE


# ---------------------------------------------------------------------------
# classify_query (main entry)
# ---------------------------------------------------------------------------

class TestClassifyQuery:
    def test_default_uses_patterns(self):
        d = classify_query("What is Python?")
        assert isinstance(d, RouterDecision)
        assert d.query_type == QueryType.SIMPLE

    def test_use_llm_flag(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "temporal"
        client.chat.completions.create.return_value = resp

        d = classify_query("When?", use_llm=True, openai_client=client)
        assert d.query_type == QueryType.TEMPORAL
        assert d.confidence == 0.85
