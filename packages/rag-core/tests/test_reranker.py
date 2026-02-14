"""Tests for rag_core.reranker."""

from unittest.mock import MagicMock, patch

from rag_core.models import Chunk, SearchResult
from rag_core.reranker import rerank, rerank_cosine


def _make_result(embedding: list[float], score: float = 0.5, rank: int = 1) -> SearchResult:
    chunk = Chunk(content="test", embedding=embedding)
    return SearchResult(chunk=chunk, score=score, rank=rank)


class TestRerankCosine:
    def test_empty_results(self):
        assert rerank_cosine([1.0, 0.0], [], top_k=5) == []

    def test_no_embeddings(self):
        chunk = Chunk(content="no emb")
        results = [SearchResult(chunk=chunk, score=0.5, rank=1)]
        reranked = rerank_cosine([1.0, 0.0], results, top_k=5)
        assert len(reranked) == 1
        assert reranked[0].chunk.content == "no emb"

    def test_ranks_by_similarity(self):
        # r1 is more similar to query [1, 0] than r2
        r1 = _make_result([1.0, 0.0], rank=2)
        r2 = _make_result([0.0, 1.0], rank=1)

        reranked = rerank_cosine([1.0, 0.0], [r2, r1], top_k=2)
        assert len(reranked) == 2
        assert reranked[0].score > reranked[1].score
        assert reranked[0].rank == 1
        assert reranked[1].rank == 2

    def test_respects_top_k(self):
        results = [_make_result([1.0, 0.0]) for _ in range(5)]
        reranked = rerank_cosine([1.0, 0.0], results, top_k=2)
        assert len(reranked) == 2

    def test_zero_norm_query(self):
        results = [_make_result([1.0, 0.0])]
        reranked = rerank_cosine([0.0, 0.0], results, top_k=5)
        assert len(reranked) == 1

    def test_zero_norm_chunk(self):
        r = _make_result([0.0, 0.0])
        reranked = rerank_cosine([1.0, 0.0], [r], top_k=5)
        assert len(reranked) == 1
        assert reranked[0].score == 0.0

    def test_cosine_similarity_values(self):
        # Identical vectors → similarity ≈ 1.0
        r = _make_result([1.0, 0.0])
        reranked = rerank_cosine([1.0, 0.0], [r], top_k=1)
        assert abs(reranked[0].score - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        r = _make_result([0.0, 1.0])
        reranked = rerank_cosine([1.0, 0.0], [r], top_k=1)
        assert abs(reranked[0].score) < 1e-6

    def test_mixed_embeddings(self):
        """Results with and without embeddings — only embedded ones get reranked."""
        r1 = _make_result([1.0, 0.0], rank=1)
        r2 = SearchResult(chunk=Chunk(content="no emb"), score=0.9, rank=2)

        reranked = rerank_cosine([1.0, 0.0], [r1, r2], top_k=5)
        # Only r1 has embedding, r2 filtered out
        assert len(reranked) == 1


class TestRerank:
    @patch("rag_core.reranker.get_settings")
    def test_uses_settings_top_k(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.top_k_final = 2
        mock_settings.return_value = cfg

        results = [_make_result([1.0, 0.0]) for _ in range(5)]
        reranked = rerank([1.0, 0.0], results)
        assert len(reranked) == 2

    def test_explicit_top_k_overrides(self):
        results = [_make_result([1.0, 0.0]) for _ in range(5)]
        reranked = rerank([1.0, 0.0], results, top_k=3)
        assert len(reranked) == 3
