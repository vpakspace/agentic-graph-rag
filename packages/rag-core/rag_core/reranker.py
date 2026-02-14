"""Result re-ranking for improved retrieval quality.

From RAG 2.0 — cosine similarity re-ranking of search results.
"""

from __future__ import annotations

import logging

import numpy as np

from rag_core.config import get_settings
from rag_core.models import SearchResult

logger = logging.getLogger(__name__)


def rerank_cosine(
    query_embedding: list[float],
    results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Re-rank results by cosine similarity between query and chunk embeddings."""
    if not results:
        return []

    valid_results = [r for r in results if r.chunk.embedding]
    if not valid_results:
        logger.warning("No results with embeddings to rerank")
        return results[:top_k]

    query_vec = np.array(query_embedding)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        logger.warning("Query embedding has zero norm, returning original results")
        return results[:top_k]

    scored_results = []
    for result in valid_results:
        chunk_vec = np.array(result.chunk.embedding)
        chunk_norm = np.linalg.norm(chunk_vec)

        if chunk_norm == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm))

        scored_results.append(
            SearchResult(chunk=result.chunk, score=similarity, rank=result.rank)
        )

    scored_results.sort(key=lambda r: r.score, reverse=True)

    reranked = []
    for i, result in enumerate(scored_results[:top_k]):
        reranked.append(SearchResult(chunk=result.chunk, score=result.score, rank=i + 1))

    logger.debug("Reranked %d results to top %d", len(scored_results), len(reranked))
    return reranked


def rerank(
    query_embedding: list[float],
    results: list[SearchResult],
    top_k: int | None = None,
) -> list[SearchResult]:
    """Re-rank search results using cosine similarity.

    Args:
        query_embedding: Query embedding vector.
        results: Search results to re-rank.
        top_k: Number of top results to return.
    """
    if top_k is None:
        top_k = get_settings().retrieval.top_k_final

    return rerank_cosine(query_embedding, results, top_k)
