"""Tests for MCP server tools."""
from unittest.mock import MagicMock

from rag_core.models import (
    Chunk,
    PipelineTrace,
    QAResult,
    SearchResult,
)


def _make_qa():
    trace = PipelineTrace(
        trace_id="tr_mcp",
        timestamp="2026-02-17T00:00:00Z",
        query="test",
    )
    return QAResult(answer="answer", query="test", confidence=0.8, trace=trace,
                    sources=[SearchResult(chunk=Chunk(id="c1", content="t"), score=0.9, rank=1)])


def test_resolve_intent_tool():
    from api.mcp_server import create_mcp_tools

    svc = MagicMock()
    svc.query.return_value = _make_qa()

    tools = create_mcp_tools(svc)
    result = tools["resolve_intent"]("test query", "agent_pattern")
    assert result["answer"] == "answer"
    assert result["trace"]["trace_id"] == "tr_mcp"


def test_search_graph_dispatches_tool():
    from api.mcp_server import create_mcp_tools

    svc = MagicMock()
    svc.search.return_value = [
        SearchResult(chunk=Chunk(id="c1", content="found"), score=0.9, rank=1),
    ]

    tools = create_mcp_tools(svc)
    result = tools["search_graph"]("test query", "cypher_traverse")
    svc.search.assert_called_once_with("test query", tool="cypher_traverse")
    assert len(result["results"]) == 1


def test_search_graph_unknown_tool():
    from api.mcp_server import create_mcp_tools

    svc = MagicMock()
    svc.search.side_effect = ValueError("Unknown tool: bad_tool")

    tools = create_mcp_tools(svc)
    result = tools["search_graph"]("test", "bad_tool")
    assert "error" in result


def test_explain_trace_tool():
    from api.mcp_server import create_mcp_tools

    trace = PipelineTrace(trace_id="tr_exp", timestamp="T", query="q")
    svc = MagicMock()
    svc.get_trace.return_value = trace

    tools = create_mcp_tools(svc)
    result = tools["explain_trace"]("tr_exp")
    assert result["trace_id"] == "tr_exp"


def test_explain_trace_not_found():
    from api.mcp_server import create_mcp_tools

    svc = MagicMock()
    svc.get_trace.return_value = None

    tools = create_mcp_tools(svc)
    result = tools["explain_trace"]("nonexistent")
    assert "error" in result
