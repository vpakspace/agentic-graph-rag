"""Tests for agentic_graph_rag.agent.tools."""

from unittest.mock import MagicMock, patch

from rag_core.models import Chunk, GraphContext, SearchResult

from agentic_graph_rag.agent.tools import (
    _cosine_similarity,
    _embed_query,
    _generate_sub_queries,
    _graph_context_to_results,
    _rrf_merge,
    community_search,
    comprehensive_search,
    full_document_read,
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

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestFullDocumentRead:
    def test_reads_and_ranks_passages(self):
        driver = _mock_driver()
        session = driver.session().__enter__()
        # query embedding = [1.0, 0.0]
        client = _mock_openai_client([1.0, 0.0])

        rec1 = MagicMock()
        rec1.__getitem__ = lambda self, key: {
            "id": "p1", "text": "Text one", "chunk_id": "c1",
            "embedding": [0.5, 0.5],
        }[key]
        rec2 = MagicMock()
        rec2.__getitem__ = lambda self, key: {
            "id": "p2", "text": "Text two", "chunk_id": "c2",
            "embedding": [1.0, 0.0],  # identical to query → highest similarity
        }[key]

        result_mock = MagicMock()
        result_mock.__iter__ = MagicMock(return_value=iter([rec1, rec2]))
        session.run.return_value = result_mock

        results = full_document_read("overview", driver, client, top_k=5)
        assert len(results) == 2
        # rec2 has higher similarity, should be first
        assert results[0].chunk.content == "Text two"
        assert results[0].source == "full_read"
        assert results[0].score > results[1].score
        assert results[1].chunk.content == "Text one"

    def test_passages_without_embedding(self):
        driver = _mock_driver()
        session = driver.session().__enter__()
        client = _mock_openai_client([1.0, 0.0])

        rec1 = MagicMock()
        rec1.__getitem__ = lambda self, key: {
            "id": "p1", "text": "No emb", "chunk_id": "c1",
            "embedding": None,
        }[key]

        result_mock = MagicMock()
        result_mock.__iter__ = MagicMock(return_value=iter([rec1]))
        session.run.return_value = result_mock

        results = full_document_read("overview", driver, client, top_k=5)
        assert len(results) == 1
        assert results[0].score == 0.0

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

    def test_temporal_query_boosts_temporal_passages(self):
        driver = _mock_driver()
        session = driver.session().__enter__()
        client = _mock_openai_client([1.0, 0.0])

        rec_temporal = MagicMock()
        rec_temporal.__getitem__ = lambda self, key: {
            "id": "p1", "text": "Компания основана в 2015 году",
            "chunk_id": "c1", "embedding": [0.5, 0.5],
        }[key]
        rec_regular = MagicMock()
        rec_regular.__getitem__ = lambda self, key: {
            "id": "p2", "text": "Описание продукта и характеристики",
            "chunk_id": "c2", "embedding": [0.6, 0.4],
        }[key]

        result_mock = MagicMock()
        result_mock.__iter__ = MagicMock(return_value=iter([rec_temporal, rec_regular]))
        session.run.return_value = result_mock

        results = temporal_query("когда основана компания", driver, client)
        assert len(results) == 2
        assert results[0].source == "temporal"
        # Temporal passage should be boosted above regular
        assert results[0].chunk.content == "Компания основана в 2015 году"

    def test_temporal_query_empty_falls_back(self):
        driver = _mock_driver()
        session = driver.session().__enter__()
        client = _mock_openai_client()

        result_mock = MagicMock()
        result_mock.__iter__ = MagicMock(return_value=iter([]))
        session.run.return_value = result_mock

        with patch("agentic_graph_rag.agent.tools.vector_search") as mock_vs:
            mock_vs.return_value = _make_results(2)
            results = temporal_query("when", driver, client)
            mock_vs.assert_called_once()
            assert len(results) == 2


# ---------------------------------------------------------------------------
# _generate_sub_queries
# ---------------------------------------------------------------------------

class TestGenerateSubQueries:
    def test_generates_sub_queries(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "sub query 1\nsub query 2\nsub query 3"
        client.chat.completions.create.return_value = resp

        subs = _generate_sub_queries("list all features", client, "gpt-4o-mini", n=3)
        assert len(subs) == 3
        assert subs[0] == "sub query 1"

    def test_handles_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("fail")
        subs = _generate_sub_queries("test", client, "gpt-4o-mini")
        assert subs == ["test"]

    def test_handles_empty_response(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = ""
        client.chat.completions.create.return_value = resp

        subs = _generate_sub_queries("test", client, "gpt-4o-mini")
        assert subs == ["test"]

    def test_limits_to_n(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "q1\nq2\nq3\nq4\nq5"
        client.chat.completions.create.return_value = resp

        subs = _generate_sub_queries("test", client, "gpt-4o-mini", n=2)
        assert len(subs) == 2


# ---------------------------------------------------------------------------
# comprehensive_search
# ---------------------------------------------------------------------------

class TestComprehensiveSearch:
    @patch("agentic_graph_rag.agent.tools.vector_search")
    @patch("agentic_graph_rag.agent.tools._generate_sub_queries")
    def test_merges_sub_query_results(self, mock_gen, mock_vs):
        mock_gen.return_value = ["sub1", "sub2", "sub3"]
        # Return different results for each call
        mock_vs.side_effect = [
            _make_results(3, source="v"),  # sub1
            _make_results(3, source="v"),  # sub2
            _make_results(3, source="v"),  # sub3
            _make_results(3, source="v"),  # original query
        ]

        driver = _mock_driver()
        client = _mock_openai_client()

        results = comprehensive_search("list all features", driver, client, top_k=10)
        assert len(results) > 0
        assert mock_vs.call_count == 4  # 3 sub-queries + 1 original

    @patch("agentic_graph_rag.agent.tools.vector_search")
    @patch("agentic_graph_rag.agent.tools._generate_sub_queries")
    def test_falls_back_on_empty_sub_queries(self, mock_gen, mock_vs):
        mock_gen.return_value = []
        mock_vs.return_value = _make_results(5, source="v")

        driver = _mock_driver()
        client = _mock_openai_client()

        results = comprehensive_search("test", driver, client)
        assert len(results) == 5
