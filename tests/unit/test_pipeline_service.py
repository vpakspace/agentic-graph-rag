"""Tests for PipelineService."""
from unittest.mock import MagicMock, patch

from rag_core.models import Chunk, QAResult, SearchResult


def _mock_qa():
    return QAResult(answer="test answer", query="q", sources=[
        SearchResult(chunk=Chunk(id="c1", content="text"), score=0.9, rank=1),
    ], confidence=0.8)


@patch("agentic_graph_rag.service.agent_run")
def test_service_query_returns_qa_with_trace(mock_run):
    mock_run.return_value = _mock_qa()
    mock_run.return_value.trace = MagicMock(trace_id="tr_abc")

    from agentic_graph_rag.service import PipelineService

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())
    qa = svc.query("test question")
    assert qa.answer == "test answer"
    mock_run.assert_called_once()


@patch("agentic_graph_rag.service.agent_run")
def test_service_caches_trace(mock_run):
    from rag_core.models import PipelineTrace
    qa = _mock_qa()
    qa.trace = PipelineTrace(trace_id="tr_cached", timestamp="T", query="q")
    mock_run.return_value = qa

    from agentic_graph_rag.service import PipelineService

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())
    svc.query("test")
    assert svc.get_trace("tr_cached") is not None
    assert svc.get_trace("nonexistent") is None


def test_service_health():
    from agentic_graph_rag.service import PipelineService

    driver = MagicMock()
    session = MagicMock()
    session.run.return_value.single.return_value = [1]
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    svc = PipelineService(driver=driver, openai_client=MagicMock())
    health = svc.health()
    assert health["status"] == "ok"


def test_service_trace_cache_bounded():
    from agentic_graph_rag.service import PipelineService

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())
    from rag_core.models import PipelineTrace

    # Fill cache beyond limit
    for i in range(105):
        trace = PipelineTrace(trace_id=f"tr_{i:04d}", timestamp="T", query="q")
        svc._cache_trace(trace)

    # Oldest should be evicted (cache max 100)
    assert svc.get_trace("tr_0000") is None
    assert svc.get_trace("tr_0104") is not None
