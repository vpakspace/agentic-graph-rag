"""Dual-node graph structure for HippoRAG 2.

Creates and manages PhraseNode (entity-level) and PassageNode (full-text)
nodes in Neo4j, linked via MENTIONED_IN relationships.

Also provides Personalized PageRank (PPR) for query-focused retrieval.

Reference: HippoRAG 2 (ICML 2025) — F1 +7.1 on MuSiQue, 12x fewer tokens.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import networkx as nx
from rag_core.config import get_settings
from rag_core.models import Chunk, Entity, PassageNode, PhraseNode, Relationship

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

logger = logging.getLogger(__name__)

PHRASE_LABEL = "PhraseNode"
PASSAGE_LABEL = "PassageNode"


# ---------------------------------------------------------------------------
# 1. Create PhraseNodes in Neo4j
# ---------------------------------------------------------------------------

def create_phrase_nodes(
    entities: list[Entity],
    driver: Driver,
    pagerank_scores: dict[str, float] | None = None,
) -> list[PhraseNode]:
    """Create PhraseNode nodes in Neo4j from extracted entities.

    Args:
        entities: Entities to create as graph nodes.
        driver: Neo4j driver.
        pagerank_scores: Optional mapping entity_id → pagerank score.

    Returns list of created PhraseNode objects.
    """
    if not entities:
        return []

    phrase_nodes: list[PhraseNode] = []
    scores = pagerank_scores or {}

    with driver.session() as session:
        for entity in entities:
            eid = entity.id or hashlib.md5(entity.name.lower().encode()).hexdigest()[:8]
            pr_score = scores.get(eid, 0.0)

            session.run(
                f"""
                MERGE (p:{PHRASE_LABEL} {{id: $id}})
                SET p.name = $name,
                    p.entity_type = $entity_type,
                    p.description = $description,
                    p.pagerank_score = $pagerank_score
                """,
                id=eid,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                pagerank_score=pr_score,
            )

            phrase_nodes.append(PhraseNode(
                id=eid,
                name=entity.name,
                entity_type=entity.entity_type,
                pagerank_score=pr_score,
            ))

    logger.info("Created %d PhraseNodes in Neo4j", len(phrase_nodes))
    return phrase_nodes


# ---------------------------------------------------------------------------
# 2. Create PassageNodes in Neo4j
# ---------------------------------------------------------------------------

def create_passage_nodes(
    chunks: list[Chunk],
    driver: Driver,
) -> list[PassageNode]:
    """Create PassageNode nodes in Neo4j from text chunks.

    Each passage stores full text + embedding for later retrieval.
    """
    if not chunks:
        return []

    passage_nodes: list[PassageNode] = []

    with driver.session() as session:
        for chunk in chunks:
            pid = chunk.id or hashlib.md5(chunk.content.encode()).hexdigest()[:8]

            session.run(
                f"""
                MERGE (p:{PASSAGE_LABEL} {{id: $id}})
                SET p.text = $text,
                    p.chunk_id = $chunk_id,
                    p.embedding = $embedding
                """,
                id=pid,
                text=chunk.enriched_content,
                chunk_id=chunk.id,
                embedding=chunk.embedding,
            )

            passage_nodes.append(PassageNode(
                id=pid,
                text=chunk.enriched_content,
                chunk_id=chunk.id,
                embedding=chunk.embedding,
            ))

    logger.info("Created %d PassageNodes in Neo4j", len(passage_nodes))
    return passage_nodes


# ---------------------------------------------------------------------------
# 3. Link PhraseNode → PassageNode via MENTIONED_IN
# ---------------------------------------------------------------------------

def link_phrase_to_passage(
    phrase_id: str,
    passage_id: str,
    driver: Driver,
) -> None:
    """Create MENTIONED_IN relationship between PhraseNode and PassageNode."""
    with driver.session() as session:
        session.run(
            f"""
            MATCH (ph:{PHRASE_LABEL} {{id: $phrase_id}})
            MATCH (pa:{PASSAGE_LABEL} {{id: $passage_id}})
            MERGE (ph)-[:MENTIONED_IN]->(pa)
            """,
            phrase_id=phrase_id,
            passage_id=passage_id,
        )


def link_entities_to_passages(
    entities: list[Entity],
    chunks: list[Chunk],
    driver: Driver,
) -> int:
    """Link all entities to chunks where they're mentioned.

    Uses case-insensitive substring matching.
    Returns number of links created.
    """
    if not entities or not chunks:
        return 0

    count = 0
    for entity in entities:
        name_lower = entity.name.lower()
        if len(name_lower) < 2:
            continue
        eid = entity.id or hashlib.md5(name_lower.encode()).hexdigest()[:8]

        for chunk in chunks:
            if name_lower in chunk.enriched_content.lower():
                pid = chunk.id or hashlib.md5(chunk.content.encode()).hexdigest()[:8]
                link_phrase_to_passage(eid, pid, driver)
                count += 1

    logger.info("Created %d MENTIONED_IN links", count)
    return count


# ---------------------------------------------------------------------------
# 4. Create inter-PhraseNode relationships (RELATED_TO)
# ---------------------------------------------------------------------------

RELATED_TO_LABEL = "RELATED_TO"


def create_phrase_relationships(
    relationships: list[Relationship],
    driver: Driver,
) -> int:
    """Create RELATED_TO edges between PhraseNodes from extracted relationships.

    Matches source/target by case-insensitive PhraseNode name.
    Returns number of edges created.
    """
    if not relationships:
        return 0

    count = 0
    with driver.session() as session:
        for rel in relationships:
            src = rel.source.strip()
            tgt = rel.target.strip()
            if not src or not tgt or src.lower() == tgt.lower():
                continue

            result = session.run(
                f"""
                MATCH (a:{PHRASE_LABEL})
                WHERE toLower(a.name) = toLower($src)
                MATCH (b:{PHRASE_LABEL})
                WHERE toLower(b.name) = toLower($tgt)
                MERGE (a)-[r:{RELATED_TO_LABEL}]->(b)
                SET r.relation_type = $rel_type
                RETURN count(r) AS cnt
                """,
                src=src,
                tgt=tgt,
                rel_type=rel.relation_type,
            )
            rec = result.single()
            if rec and rec["cnt"] > 0:
                count += 1

    logger.info("Created %d RELATED_TO edges between PhraseNodes", count)
    return count


# ---------------------------------------------------------------------------
# 5. Personalized PageRank (PPR)
# ---------------------------------------------------------------------------

def compute_ppr(
    graph: nx.Graph,
    query_nodes: list[int | str],
    alpha: float | None = None,
) -> dict[int | str, float]:
    """Compute Personalized PageRank from query starting nodes.

    Args:
        graph: NetworkX graph (can be directed or undirected).
        query_nodes: Starting node IDs for personalization.
        alpha: Restart probability (default from settings).

    Returns mapping node → PPR score.
    """
    if alpha is None:
        alpha = get_settings().retrieval.ppr_alpha

    if graph.number_of_nodes() == 0 or not query_nodes:
        return {}

    # Build personalization vector: uniform over query nodes
    personalization: dict[int | str, float] = {}
    valid_query = [n for n in query_nodes if n in graph]
    if not valid_query:
        return {}

    weight = 1.0 / len(valid_query)
    for node in graph.nodes():
        personalization[node] = weight if node in valid_query else 0.0

    scores: dict[int | str, float] = nx.pagerank(
        graph,
        alpha=alpha,
        personalization=personalization,
        weight="weight",
    )

    logger.debug("PPR computed: %d nodes, %d query nodes", len(scores), len(valid_query))
    return scores


# ---------------------------------------------------------------------------
# 5. Build dual-node graph from entities + chunks
# ---------------------------------------------------------------------------

def build_dual_graph(
    entities: list[Entity],
    chunks: list[Chunk],
    driver: Driver,
    pagerank_scores: dict[str, float] | None = None,
    relationships: list[Relationship] | None = None,
) -> tuple[list[PhraseNode], list[PassageNode], int]:
    """Build complete dual-node graph in Neo4j.

    1. Create PhraseNodes from entities
    2. Create PassageNodes from chunks
    3. Link entities to passages via MENTIONED_IN
    4. Create inter-PhraseNode edges from relationships (RELATED_TO)

    Returns (phrase_nodes, passage_nodes, link_count).
    """
    phrase_nodes = create_phrase_nodes(entities, driver, pagerank_scores)
    passage_nodes = create_passage_nodes(chunks, driver)
    link_count = link_entities_to_passages(entities, chunks, driver)

    # Create inter-PhraseNode edges from extracted relationships
    rel_count = create_phrase_relationships(relationships or [], driver)

    logger.info(
        "Dual graph built: %d phrases, %d passages, %d MENTIONED_IN, %d RELATED_TO",
        len(phrase_nodes), len(passage_nodes), link_count, rel_count,
    )
    return phrase_nodes, passage_nodes, link_count


# ---------------------------------------------------------------------------
# 6. Embed PhraseNodes and create vector index
# ---------------------------------------------------------------------------

PHRASE_INDEX_NAME = "phrase_node_index"


def embed_phrase_nodes(
    phrase_nodes: list[PhraseNode],
    driver: Driver,
    openai_client: OpenAI | None = None,
) -> int:
    """Add embeddings to PhraseNodes by embedding their name + description.

    Returns count of nodes updated.
    """
    cfg = get_settings()
    if openai_client is None:
        from openai import OpenAI as _OpenAI

        openai_client = _OpenAI(api_key=cfg.openai.api_key)

    if not phrase_nodes:
        return 0

    # Batch embed all phrase texts
    texts = [
        f"{pn.name}: {pn.entity_type}" for pn in phrase_nodes
    ]

    response = openai_client.embeddings.create(
        model=cfg.openai.embedding_model,
        input=texts,
        dimensions=cfg.openai.embedding_dimensions,
    )

    with driver.session() as session:
        for i, pn in enumerate(phrase_nodes):
            emb = response.data[i].embedding
            session.run(
                f"""
                MATCH (p:{PHRASE_LABEL} {{id: $id}})
                SET p.embedding = $embedding
                """,
                id=pn.id,
                embedding=emb,
            )

    logger.info("Added embeddings to %d PhraseNodes", len(phrase_nodes))
    return len(phrase_nodes)


def init_phrase_index(driver: Driver) -> None:
    """Create vector index on PhraseNode embeddings if not exists."""
    cfg = get_settings()
    with driver.session() as session:
        session.run(
            f"""
            CREATE VECTOR INDEX {PHRASE_INDEX_NAME} IF NOT EXISTS
            FOR (n:{PHRASE_LABEL})
            ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
            dimensions=cfg.openai.embedding_dimensions,
        )
    logger.info("Phrase vector index '%s' initialized", PHRASE_INDEX_NAME)
