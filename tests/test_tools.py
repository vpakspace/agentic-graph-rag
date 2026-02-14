"""Tests for agentic_graph_rag.agent.tools."""

from unittest.mock import MagicMock, patch

from rag_core.models import Chunk, GraphContext, SearchResult

from agentic_graph_rag.agent.tools import (
    _embed_query,
    _graph_context_to_results,
    _rrf_merge,
    community_search,
    cypher_traverse,
    full_document_read,
    hybrid_search,
    temporal_query,
    vector_search,
)


def _mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


def _mock_openai_client(embedding=None):
    client = MagicMock()
    resp = MagicMock()
    resp.data = [MagicMock()]
    resp.data[0].embedding = embedding or [1.0, 0.0]
    client.embeddings.create.return_value = resp
    return client


def _make_results(n: int, source: str = "vector") -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(id=f"c{i}", content=f"Content {i}"),
            score=1.0 / (i + 1),
            rank=i + 1,
            source=source,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _embed_query
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def test_returns_embedding(self):
        client = _mock_openai_client([0.5, 0.5])
        emb = _embed_query("test", client)
        assert emb == [0.5, 0.5]
        client.embeddings.create.assert_called_once()


# ---------------------------------------------------------------------------
# _graph_context_to_results
# ---------------------------------------------------------------------------

class TestGraphContextToResults:
    def test_empty_context(self):
        ctx = GraphContext()
        results = _graph_context_to_results(ctx, "test")
        assert results == []

    def test_converts_passages(self):
        ctx = GraphContext(
            passages=["Text A", "Text B"],
            source_ids=["c1", "c2"],
        )
        results = _graph_context_to_results(ctx, "graph")
        assert len(results) == 2
        assert results[0].chunk.content == "Text A"
        assert results[0].chunk.id == "c1"
        assert results[0].source == "graph"
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_handles_missing_source_ids(self):
        ctx = GraphContext(
            passages=["Text A", "Text B"],
            source_ids=["c1"],
        )
        results = _graph_context_to_results(ctx, "test")
        assert len(results) == 2
        assert results[0].chunk.id == "c1"
        assert results[1].chunk.id == ""  # no source_id for index 1


# ---------------------------------------------------------------------------
# _rrf_merge
# ---------------------------------------------------------------------------

class TestRRFMerge:
    def test_empty_lists(self):
        merged = _rrf_merge([], [])
        assert merged == []

    def test_single_list(self):
        results = _make_results(3)
        merged = _rrf_merge(results, [], top_k=5)
        assert len(merged) == 3

    def test_merges_two_lists(self):
        a = _make_results(3, source="vector")
        b = _make_results(3, source="graph")
        merged = _rrf_merge(a, b, top_k=5)
        # 3 unique IDs (c0, c1, c2) with combined scores
        assert len(merged) == 3
        assert merged[0].source == "hybrid"

    def test_deduplicates(self):
        a = [SearchResult(chunk=Chunk(id="c1", content="A"), score=0.9, rank=1, source="v")]
        b = [SearchResult(chunk=Chunk(id="c1", content="A"), score=0.8, rank=1, source="g")]
        merged = _rrf_merge(a, b, top_k=5)
        assert len(merged) == 1  # deduplicated by id

    def test_respects_top_k(self):
        a = _make_results(10)
        b = _make_results(10)
        merged = _rrf_merge(a, b, top_k=3)
        assert len(merged) == 3

    def test_ranks_assigned(self):
        merged = _rrf_merge(_make_results(3), _make_results(3), top_k=5)
        for i, r in enumerate(merged, start=1):
            assert r.rank == i


# ---------------------------------------------------------------------------
# vector_search
# ---------------------------------------------------------------------------

class TestVectorSearch:
    def test_returns_results(self):
        driver = _mock_driver()
        client = _mock_openai_client()
        ctx = GraphContext(passages=["Result"], source_ids=["c1"])

        with patch(
            "agentic_graph_rag.retrieval.vector_cypher.search",
            return_value=ctx,
        ):
            results = vector_search("test", driver, client, top_k=5)

        assert len(results) == 1
        assert results[0].chunk.content == "Result"
        client.embeddings.create.assert_called_once()


# ---------------------------------------------------------------------------
# full_document_read
# ---------------------------------------------------------------------------

class TestFullDocumentRead:
    def test_reads_passages(self):
        driver = _mock_driver()
        session = driver.session().__enter__()
        client = _mock_openai_client()

        rec1 = MagicMock()
        rec1.__getitem__ = lambda self, key: {"id": "p1", "text": "Text one", "chunk_id": "c1"}[key]
        rec2 = MagicMock()
        rec2.__getitem__ = lambda self, key: {"id": "p2", "text": "Text two", "chunk_id": "c2"}[key]

        result_mock = MagicMock()
        result_mock.__iter__ = MagicMock(return_value=iter([rec1, rec2]))
        session.run.return_value = result_mock

        results = full_document_read("overview", driver, client, top_k=5)
        assert len(results) == 2
        assert results[0].chunk.content == "Text one"
        assert results[0].source == "full_read"
        assert results[1].chunk.content == "Text two"

    def test_empty_passages(self):
        driver = _mock_driver()
        session = driver.session().__enter__()
        client = _mock_openai_client()

        result_mock = MagicMock()
        result_mock.__iter__ = MagicMock(return_value=iter([]))
        session.run.return_value = result_mock

        results = full_document_read("overview", driver, client, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# community_search / temporal_query (wrappers)
# ---------------------------------------------------------------------------

class TestWrapperTools:
    @patch("agentic_graph_rag.agent.tools.vector_search")
    def test_community_search_fallback(self, mock_vs):
        mock_vs.return_value = _make_results(2)
        driver = _mock_driver()
        client = _mock_openai_client()

        results = community_search("test", driver, client)
        mock_vs.assert_called_once_with("test", driver, client)
        assert len(results) == 2

    @patch("agentic_graph_rag.agent.tools.cypher_traverse")
    def test_temporal_query_uses_deep_traverse(self, mock_ct):
        mock_ct.return_value = _make_results(3)
        driver = _mock_driver()
        client = _mock_openai_client()

        results = temporal_query("when", driver, client)
        mock_ct.assert_called_once_with("when", driver, client, max_hops=3)
        assert len(results) == 3
