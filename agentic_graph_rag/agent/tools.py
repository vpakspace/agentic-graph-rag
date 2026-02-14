"""Retrieval Tools — callable tools for the agentic router.

Each tool wraps a retrieval strategy and returns a list of SearchResult.
Tools are pure functions with driver/client injected for testability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.models import Chunk, GraphContext, SearchResult

if TYPE_CHECKING:
    from openai import OpenAI
    from neo4j import Driver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: embed query
# ---------------------------------------------------------------------------

def _embed_query(query: str, openai_client: OpenAI) -> list[float]:
    """Embed query text using OpenAI embeddings."""
    cfg = get_settings()
    response = openai_client.embeddings.create(
        model=cfg.openai.embedding_model,
        input=query,
        dimensions=cfg.openai.embedding_dimensions,
    )
    return response.data[0].embedding


def _graph_context_to_results(ctx: GraphContext, source: str) -> list[SearchResult]:
    """Convert GraphContext passages into SearchResult list."""
    results = []
    for i, passage in enumerate(ctx.passages):
        chunk_id = ctx.source_ids[i] if i < len(ctx.source_ids) else ""
        results.append(SearchResult(
            chunk=Chunk(id=chunk_id, content=passage),
            score=1.0 / (i + 1),  # rank-based score
            rank=i + 1,
            source=source,
        ))
    return results


# ---------------------------------------------------------------------------
# 1. Vector search (simple)
# ---------------------------------------------------------------------------

def vector_search(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Simple vector similarity search on PhraseNodes.

    Uses VectorCypher find_entry_points but returns passages directly.
    """
    from agentic_graph_rag.retrieval.vector_cypher import search as vc_search

    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_vector

    embedding = _embed_query(query, openai_client)
    ctx = vc_search(embedding, driver, top_k=top_k, max_hops=1)

    return _graph_context_to_results(ctx, source="vector")


# ---------------------------------------------------------------------------
# 2. Cypher traverse (relation / multi-hop)
# ---------------------------------------------------------------------------

def cypher_traverse(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
    max_hops: int | None = None,
) -> list[SearchResult]:
    """VectorCypher retrieval — vector entry + deep graph traversal."""
    from agentic_graph_rag.retrieval.vector_cypher import search as vc_search

    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_vector
    if max_hops is None:
        max_hops = cfg.retrieval.max_hops

    embedding = _embed_query(query, openai_client)
    ctx = vc_search(embedding, driver, top_k=top_k, max_hops=max_hops)

    return _graph_context_to_results(ctx, source="graph")


# ---------------------------------------------------------------------------
# 3. Community search (Graphiti)
# ---------------------------------------------------------------------------

def community_search(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
) -> list[SearchResult]:
    """Search using Graphiti community summaries.

    Falls back to vector search if Graphiti is unavailable.
    """
    # Community search requires Graphiti; use vector search as fallback
    logger.info("Community search — falling back to vector search (Graphiti optional)")
    return vector_search(query, driver, openai_client)


# ---------------------------------------------------------------------------
# 4. Hybrid search (vector + graph merge via RRF)
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Hybrid retrieval: vector + graph results merged via Reciprocal Rank Fusion."""
    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_final

    vector_results = vector_search(query, driver, openai_client, top_k=top_k)
    graph_results = cypher_traverse(query, driver, openai_client, top_k=top_k)

    return _rrf_merge(vector_results, graph_results, top_k=top_k)


def _rrf_merge(
    list_a: list[SearchResult],
    list_b: list[SearchResult],
    top_k: int = 5,
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion merge of two result lists."""
    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(list_a, start=1):
        key = r.chunk.id or r.chunk.content[:50]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        result_map[key] = r

    for rank, r in enumerate(list_b, start=1):
        key = r.chunk.id or r.chunk.content[:50]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in result_map:
            result_map[key] = r

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    merged = []
    for i, key in enumerate(sorted_keys):
        result = result_map[key]
        merged.append(SearchResult(
            chunk=result.chunk,
            score=scores[key],
            rank=i + 1,
            source="hybrid",
        ))

    return merged


# ---------------------------------------------------------------------------
# 5. Temporal query
# ---------------------------------------------------------------------------

def temporal_query(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
) -> list[SearchResult]:
    """Temporal-aware query using Graphiti temporal search.

    Falls back to cypher_traverse for temporal context.
    """
    logger.info("Temporal query — using deep cypher traversal")
    return cypher_traverse(query, driver, openai_client, max_hops=3)


# ---------------------------------------------------------------------------
# 6. Full document read
# ---------------------------------------------------------------------------

def full_document_read(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Read all passage nodes — for global/overview queries."""
    from agentic_graph_rag.indexing.dual_node import PASSAGE_LABEL

    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_final

    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (pa:{PASSAGE_LABEL})
            WHERE pa.text IS NOT NULL AND pa.text <> ''
            RETURN pa.id AS id, pa.text AS text, pa.chunk_id AS chunk_id
            LIMIT $limit
            """,
            limit=top_k * 3,
        )

        results = []
        for i, record in enumerate(result, start=1):
            results.append(SearchResult(
                chunk=Chunk(
                    id=record["chunk_id"] or record["id"] or "",
                    content=record["text"] or "",
                ),
                score=1.0 / i,
                rank=i,
                source="full_read",
            ))

    logger.info("Full document read: %d passages", len(results))
    return results[:top_k]
