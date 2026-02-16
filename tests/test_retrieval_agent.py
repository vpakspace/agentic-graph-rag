"""Tests for agentic_graph_rag.agent.retrieval_agent."""

from unittest.mock import MagicMock, patch

from rag_core.models import (
    Chunk,
    QAResult,
    QueryType,
    RouterDecision,
    SearchResult,
)

from agentic_graph_rag.agent.retrieval_agent import (
    _ESCALATION_CHAIN,
    _TOOL_REGISTRY,
    _get_next_tool,
    run,
    select_tool,
    self_correction_loop,
)


def _make_results(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(id=f"c{i}", content=f"Content {i}"),
            score=0.9 - i * 0.1,
            rank=i + 1,
        )
        for i in range(n)
    ]


def _make_decision(
    query_type: QueryType = QueryType.SIMPLE,
    tool: str = "vector_search",
) -> RouterDecision:
    return RouterDecision(
        query_type=query_type,
        confidence=0.8,
        reasoning="test",
        suggested_tool=tool,
    )


def _mock_tool(results=None):
    """Create a mock tool function."""
    mock = MagicMock()
    mock.return_value = results if results is not None else _make_results(3)
    return mock


# ---------------------------------------------------------------------------
# select_tool
# ---------------------------------------------------------------------------

class TestSelectTool:
    def test_valid_tool(self):
        d = _make_decision(tool="cypher_traverse")
        assert select_tool(d) == "cypher_traverse"

    def test_unknown_tool_defaults(self):
        d = _make_decision(tool="nonexistent_tool")
        assert select_tool(d) == "vector_search"

    def test_all_known_tools(self):
        for tool in ["vector_search", "cypher_traverse", "community_search",
                      "hybrid_search", "temporal_query", "full_document_read"]:
            d = _make_decision(tool=tool)
            assert select_tool(d) == tool


# ---------------------------------------------------------------------------
# _get_next_tool
# ---------------------------------------------------------------------------

class TestGetNextTool:
    def test_escalates_from_vector(self):
        nxt = _get_next_tool("vector_search", {"vector_search"})
        assert nxt == "cypher_traverse"

    def test_escalates_from_cypher(self):
        nxt = _get_next_tool("cypher_traverse", {"cypher_traverse"})
        assert nxt == "hybrid_search"

    def test_skips_tried(self):
        nxt = _get_next_tool("vector_search", {"vector_search", "cypher_traverse"})
        assert nxt == "hybrid_search"

    def test_no_more_tools(self):
        nxt = _get_next_tool("full_document_read", {"full_document_read"})
        assert nxt is None

    def test_all_tried(self):
        all_tools = {"vector_search", "cypher_traverse", "hybrid_search", "comprehensive_search", "full_document_read"}
        nxt = _get_next_tool("vector_search", all_tools)
        assert nxt is None

    def test_unknown_current(self):
        nxt = _get_next_tool("temporal_query", {"temporal_query"})
        assert nxt == "vector_search"


# ---------------------------------------------------------------------------
# self_correction_loop
# ---------------------------------------------------------------------------

class TestSelfCorrectionLoop:
    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_relevance")
    def test_no_retry_when_relevant(self, mock_eval):
        results = _make_results(3)
        mock_tool = _mock_tool(results)
        mock_eval.return_value = 4.0

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {"vector_search": mock_tool}):
            out, retries = self_correction_loop(
                "test", driver, client, decision,
                max_retries=2, relevance_threshold=3.0,
            )
        assert retries == 0
        assert out == results
        mock_tool.assert_called_once()

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_relevance")
    def test_escalates_on_low_relevance(self, mock_eval):
        mock_vs = _mock_tool(_make_results(2))
        mock_ct = _mock_tool(_make_results(3))
        mock_eval.side_effect = [1.0, 4.0]

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {
            "vector_search": mock_vs,
            "cypher_traverse": mock_ct,
        }):
            out, retries = self_correction_loop(
                "test", driver, client, decision,
                max_retries=2, relevance_threshold=3.0,
            )
        assert retries == 1
        assert mock_vs.call_count == 1
        assert mock_ct.call_count == 1

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_relevance")
    def test_empty_results_triggers_escalation(self, mock_eval):
        mock_vs = _mock_tool([])
        mock_ct = _mock_tool(_make_results(2))
        mock_eval.return_value = 4.0

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {
            "vector_search": mock_vs,
            "cypher_traverse": mock_ct,
        }):
            out, retries = self_correction_loop(
                "test", driver, client, decision,
                max_retries=2, relevance_threshold=3.0,
            )
        assert retries == 1

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_relevance")
    def test_max_retries_exhausted(self, mock_eval):
        mock_vs = _mock_tool(_make_results(1))
        mock_ct = _mock_tool(_make_results(1))
        mock_hs = _mock_tool(_make_results(1))
        mock_fdr = _mock_tool(_make_results(1))
        mock_eval.return_value = 1.0

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {
            "vector_search": mock_vs,
            "cypher_traverse": mock_ct,
            "hybrid_search": mock_hs,
            "full_document_read": mock_fdr,
        }):
            out, retries = self_correction_loop(
                "test", driver, client, decision,
                max_retries=2, relevance_threshold=3.0,
            )
        assert retries == 2

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_relevance")
    def test_uses_settings_defaults(self, mock_eval):
        mock_tool = _mock_tool(_make_results(2))
        mock_eval.return_value = 4.0

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {"vector_search": mock_tool}):
            with patch("agentic_graph_rag.agent.retrieval_agent.get_settings") as mock_cfg:
                cfg = MagicMock()
                cfg.agent.max_retries = 1
                cfg.agent.relevance_threshold = 2.0
                mock_cfg.return_value = cfg

                out, retries = self_correction_loop(
                    "test", driver, client, decision,
                )
        assert retries == 0


# ---------------------------------------------------------------------------
# run (full pipeline)
# ---------------------------------------------------------------------------

class TestRun:
    @patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
    @patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
    @patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
    def test_full_pipeline(self, mock_classify, mock_loop, mock_gen):
        decision = _make_decision()
        mock_classify.return_value = decision
        results = _make_results(3)
        mock_loop.return_value = (results, 0)
        mock_gen.return_value = QAResult(
            answer="Test answer", sources=results, confidence=0.8, query="test",
        )

        driver = MagicMock()
        client = MagicMock()

        qa = run("test query", driver, openai_client=client)

        assert isinstance(qa, QAResult)
        assert qa.answer == "Test answer"
        assert qa.retries == 0
        assert qa.router_decision == decision
        mock_classify.assert_called_once()
        mock_loop.assert_called_once()
        mock_gen.assert_called_once()

    @patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
    @patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
    @patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
    def test_uses_llm_router(self, mock_classify, mock_loop, mock_gen):
        mock_classify.return_value = _make_decision()
        mock_loop.return_value = (_make_results(1), 0)
        mock_gen.return_value = QAResult(answer="A", query="q")

        driver = MagicMock()
        client = MagicMock()

        run("q", driver, openai_client=client, use_llm_router=True)
        mock_classify.assert_called_once_with("q", use_llm=True, openai_client=client, reasoning=None)

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_completeness", return_value=True)
    @patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
    @patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
    @patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
    @patch("agentic_graph_rag.agent.retrieval_agent.get_settings")
    def test_creates_client_when_none(self, mock_settings, mock_classify, mock_loop, mock_gen, _mock_compl):
        cfg = MagicMock()
        cfg.openai.api_key = "key"
        mock_settings.return_value = cfg
        mock_classify.return_value = _make_decision()
        mock_loop.return_value = (_make_results(1), 0)
        mock_gen.return_value = QAResult(answer="A", query="q")

        driver = MagicMock()

        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            run("q", driver)
            mock_cls.assert_called_once_with(api_key="key")


# ---------------------------------------------------------------------------
# Escalation chain structure
# ---------------------------------------------------------------------------

class TestEscalationChain:
    def test_comprehensive_search_in_chain(self):
        assert "comprehensive_search" in _ESCALATION_CHAIN
        # comprehensive_search should be after hybrid_search and before full_document_read
        idx_cs = _ESCALATION_CHAIN.index("comprehensive_search")
        idx_hs = _ESCALATION_CHAIN.index("hybrid_search")
        idx_fdr = _ESCALATION_CHAIN.index("full_document_read")
        assert idx_cs > idx_hs
        assert idx_cs < idx_fdr

    def test_comprehensive_search_in_registry(self):
        assert "comprehensive_search" in _TOOL_REGISTRY


# ---------------------------------------------------------------------------
# self_correction_loop with retry query
# ---------------------------------------------------------------------------

class TestRetryQuery:
    @patch("agentic_graph_rag.agent.retrieval_agent.generate_retry_query")
    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_relevance")
    def test_rephrases_query_before_escalation(self, mock_eval, mock_retry):
        mock_vs = _mock_tool(_make_results(2))
        mock_ct = _mock_tool(_make_results(3))
        mock_eval.side_effect = [1.0, 4.0]  # low then high
        mock_retry.return_value = "rephrased query"

        driver = MagicMock()
        client = MagicMock()
        decision = _make_decision()

        with patch.dict(_TOOL_REGISTRY, {
            "vector_search": mock_vs,
            "cypher_traverse": mock_ct,
        }):
            out, retries = self_correction_loop(
                "test", driver, client, decision,
                max_retries=2, relevance_threshold=3.0,
            )

        # generate_retry_query should have been called once (before escalation)
        mock_retry.assert_called_once()
        # cypher_traverse should have been called with the rephrased query
        mock_ct.assert_called_once_with("rephrased query", driver, client)


# ---------------------------------------------------------------------------
# Completeness check in run()
# ---------------------------------------------------------------------------

class TestCompletenessCheck:
    @patch("agentic_graph_rag.agent.retrieval_agent.comprehensive_search")
    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_completeness")
    @patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
    @patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
    @patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
    def test_completeness_retry_for_global(self, mock_classify, mock_loop, mock_gen, mock_compl, mock_cs):
        decision = _make_decision(query_type=QueryType.GLOBAL, tool="comprehensive_search")
        mock_classify.return_value = decision
        results = _make_results(3)
        mock_loop.return_value = (results, 0)

        # First generate returns incomplete, second returns complete
        qa_incomplete = QAResult(answer="Partial answer", sources=results, confidence=0.7, query="list all")
        qa_complete = QAResult(answer="Full answer: A, B, C, D", sources=results, confidence=0.9, query="list all")
        mock_gen.side_effect = [qa_incomplete, qa_complete]

        mock_compl.return_value = False  # answer is incomplete
        mock_cs.return_value = _make_results(5)

        driver = MagicMock()
        client = MagicMock()

        qa = run("list all items", driver, openai_client=client)

        mock_compl.assert_called_once()
        mock_cs.assert_called_once()
        # generate_answer called twice: initial + after completeness retry
        assert mock_gen.call_count == 2
        assert qa.retries == 1

    @patch("agentic_graph_rag.agent.retrieval_agent.evaluate_completeness")
    @patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
    @patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
    @patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
    def test_no_completeness_check_for_simple(self, mock_classify, mock_loop, mock_gen, mock_compl):
        decision = _make_decision(query_type=QueryType.SIMPLE, tool="vector_search")
        mock_classify.return_value = decision
        mock_loop.return_value = (_make_results(3), 0)
        mock_gen.return_value = QAResult(answer="Answer", query="q")

        driver = MagicMock()
        client = MagicMock()

        run("what is X?", driver, openai_client=client)
        # evaluate_completeness should NOT be called for SIMPLE queries
        mock_compl.assert_not_called()
