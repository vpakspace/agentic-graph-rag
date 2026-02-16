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
    from neo4j import Driver
    from openai import OpenAI

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

    cfg_r = cfg.retrieval
    vector_results = vector_search(query, driver, openai_client, top_k=cfg_r.top_k_vector)
    graph_results = cypher_traverse(query, driver, openai_client, top_k=cfg_r.top_k_vector)

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

_TEMPORAL_RE = None


def _get_temporal_re():
    """Lazy-compiled regex for temporal keywords."""
    global _TEMPORAL_RE  # noqa: PLW0603
    if _TEMPORAL_RE is None:
        import re
        _TEMPORAL_RE = re.compile(
            r'\b('
            r'\d{4}'                                      # years: 2020, 1995
            r'|первый|первая|первое|первые'               # "first" (RU)
            r'|история|исторический|эволюция|развитие'    # history/evolution (RU)
            r'|first|history|evolution|timeline|founded'   # temporal (EN)
            r'|начало|основан|создан|появи'               # origin (RU)
            r')\b',
            re.IGNORECASE,
        )
    return _TEMPORAL_RE


def temporal_query(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
) -> list[SearchResult]:
    """Temporal-aware query: filters passages by temporal keywords, ranks by similarity."""
    from agentic_graph_rag.indexing.dual_node import PASSAGE_LABEL

    cfg = get_settings()
    top_k = cfg.retrieval.top_k_final
    query_emb = _embed_query(query, openai_client)
    temporal_re = _get_temporal_re()

    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (pa:{PASSAGE_LABEL})
            WHERE pa.text IS NOT NULL AND pa.text <> ''
            RETURN pa.id AS id, pa.text AS text, pa.chunk_id AS chunk_id,
                   pa.embedding AS embedding
            """,
        )

        scored: list[tuple[float, dict]] = []
        for record in result:
            text = record["text"] or ""
            emb = record["embedding"]
            if emb:
                sim = _cosine_similarity(query_emb, list(emb))
            else:
                sim = 0.0
            # Temporal boost: +0.15 for passages containing temporal markers
            if temporal_re.search(text):
                sim += 0.15
            scored.append((sim, {
                "id": record["id"],
                "text": text,
                "chunk_id": record["chunk_id"],
            }))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        logger.info("Temporal query — no passages, falling back to vector search")
        return vector_search(query, driver, openai_client)

    top = scored[:top_k]
    results = []
    for rank, (sim, rec) in enumerate(top, start=1):
        results.append(SearchResult(
            chunk=Chunk(
                id=rec["chunk_id"] or rec["id"] or "",
                content=rec["text"] or "",
            ),
            score=sim,
            rank=rank,
            source="temporal",
        ))

    logger.info("Temporal query: %d passages (temporal-boosted)", len(results))
    return results


# ---------------------------------------------------------------------------
# 6. Full document read
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def full_document_read(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Read passage nodes ranked by cosine similarity to query."""
    from agentic_graph_rag.indexing.dual_node import PASSAGE_LABEL

    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_final

    query_emb = _embed_query(query, openai_client)

    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (pa:{PASSAGE_LABEL})
            WHERE pa.text IS NOT NULL AND pa.text <> ''
            RETURN pa.id AS id, pa.text AS text, pa.chunk_id AS chunk_id,
                   pa.embedding AS embedding
            """,
        )

        scored: list[tuple[float, dict]] = []
        for record in result:
            emb = record["embedding"]
            if emb:
                sim = _cosine_similarity(query_emb, list(emb))
            else:
                sim = 0.0
            scored.append((sim, {
                "id": record["id"],
                "text": record["text"],
                "chunk_id": record["chunk_id"],
            }))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    results = []
    for rank, (sim, rec) in enumerate(top, start=1):
        results.append(SearchResult(
            chunk=Chunk(
                id=rec["chunk_id"] or rec["id"] or "",
                content=rec["text"] or "",
            ),
            score=sim,
            rank=rank,
            source="full_read",
        ))

    logger.info("Full document read: %d passages (ranked by similarity)", len(results))
    return results
