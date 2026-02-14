"""VectorCypher Retrieval — hybrid vector entry + Cypher graph traversal.

Uses Neo4j vector index to find entry-point PhraseNodes, then traverses
the graph via Cypher to collect related PhraseNodes and PassageNodes,
assembling a rich GraphContext for answer generation.

Pipeline: query_embedding → vector entry → graph traversal → context assembly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.models import Entity, GraphContext

from agentic_graph_rag.indexing.dual_node import PASSAGE_LABEL, PHRASE_LABEL

if TYPE_CHECKING:
    from neo4j import Driver

logger = logging.getLogger(__name__)

# Vector index on PhraseNode embeddings (created during indexing)
PHRASE_INDEX_NAME = "phrase_node_index"


# ---------------------------------------------------------------------------
# 1. Find entry points via vector search
# ---------------------------------------------------------------------------

def find_entry_points(
    query_embedding: list[float],
    driver: Driver,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """Find nearest PhraseNodes via Neo4j vector index.

    Returns list of dicts with keys: id, name, entity_type, score.
    """
    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_vector
    if threshold is None:
        threshold = cfg.retrieval.vector_threshold

    with driver.session() as session:
        result = session.run(
            f"""
            CALL db.index.vector.queryNodes(
                '{PHRASE_INDEX_NAME}', $top_k, $embedding
            )
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.id AS id,
                   node.name AS name,
                   node.entity_type AS entity_type,
                   node.pagerank_score AS pagerank_score,
                   score
            ORDER BY score DESC
            """,
            top_k=top_k,
            embedding=query_embedding,
            threshold=threshold,
        )

        entries = []
        for record in result:
            entries.append({
                "id": record["id"],
                "name": record["name"] or "",
                "entity_type": record["entity_type"] or "",
                "pagerank_score": record["pagerank_score"] or 0.0,
                "score": record["score"],
            })

    logger.info(
        "Found %d entry points (top_k=%d, threshold=%.2f)",
        len(entries), top_k, threshold,
    )
    return entries


# ---------------------------------------------------------------------------
# 2. Graph traversal from entry points
# ---------------------------------------------------------------------------

def traverse_graph(
    entry_ids: list[str],
    driver: Driver,
    max_hops: int | None = None,
) -> dict:
    """Traverse graph from entry PhraseNodes to collect related nodes.

    Uses variable-length Cypher paths (no APOC dependency).

    Returns dict with keys:
      - phrase_nodes: list of {id, name, entity_type}
      - passage_nodes: list of {id, text, chunk_id}
      - relationships: list of {source, relation, target}
    """
    if not entry_ids:
        return {"phrase_nodes": [], "passage_nodes": [], "relationships": []}

    cfg = get_settings()
    if max_hops is None:
        max_hops = cfg.retrieval.max_hops

    phrase_nodes: dict[str, dict] = {}
    passage_nodes: dict[str, dict] = {}
    relationships: list[dict[str, str]] = []

    with driver.session() as session:
        # Step 1: Traverse PhraseNode relationships up to max_hops
        result = session.run(
            f"""
            MATCH (start:{PHRASE_LABEL})
            WHERE start.id IN $entry_ids
            MATCH path = (start)-[r*1..{max_hops}]-(connected:{PHRASE_LABEL})
            UNWIND relationships(path) AS rel
            WITH start, connected, rel,
                 startNode(rel) AS src, endNode(rel) AS tgt
            RETURN DISTINCT
                connected.id AS connected_id,
                connected.name AS connected_name,
                connected.entity_type AS connected_type,
                src.id AS src_id, src.name AS src_name,
                type(rel) AS rel_type,
                tgt.id AS tgt_id, tgt.name AS tgt_name
            """,
            entry_ids=entry_ids,
        )

        for record in result:
            cid = record["connected_id"]
            if cid and cid not in phrase_nodes:
                phrase_nodes[cid] = {
                    "id": cid,
                    "name": record["connected_name"] or "",
                    "entity_type": record["connected_type"] or "",
                }

            relationships.append({
                "source": record["src_name"] or record["src_id"] or "",
                "relation": record["rel_type"] or "",
                "target": record["tgt_name"] or record["tgt_id"] or "",
            })

        # Also include entry nodes themselves
        for eid in entry_ids:
            if eid not in phrase_nodes:
                r = session.run(
                    f"""
                    MATCH (p:{PHRASE_LABEL} {{id: $id}})
                    RETURN p.id AS id, p.name AS name, p.entity_type AS entity_type
                    """,
                    id=eid,
                )
                rec = r.single()
                if rec:
                    phrase_nodes[eid] = {
                        "id": rec["id"],
                        "name": rec["name"] or "",
                        "entity_type": rec["entity_type"] or "",
                    }

        # Step 2: Collect PassageNodes linked to all discovered PhraseNodes
        all_phrase_ids = list(phrase_nodes.keys())
        if all_phrase_ids:
            result = session.run(
                f"""
                MATCH (ph:{PHRASE_LABEL})-[:MENTIONED_IN]->(pa:{PASSAGE_LABEL})
                WHERE ph.id IN $phrase_ids
                RETURN DISTINCT
                    pa.id AS id, pa.text AS text, pa.chunk_id AS chunk_id
                """,
                phrase_ids=all_phrase_ids,
            )

            for record in result:
                pid = record["id"]
                if pid and pid not in passage_nodes:
                    passage_nodes[pid] = {
                        "id": pid,
                        "text": record["text"] or "",
                        "chunk_id": record["chunk_id"] or "",
                    }

    logger.info(
        "Traversal from %d entries: %d phrases, %d passages, %d relationships",
        len(entry_ids), len(phrase_nodes), len(passage_nodes), len(relationships),
    )
    return {
        "phrase_nodes": list(phrase_nodes.values()),
        "passage_nodes": list(passage_nodes.values()),
        "relationships": relationships,
    }


# ---------------------------------------------------------------------------
# 3. Collect and assemble context
# ---------------------------------------------------------------------------

def collect_context(traversal_result: dict) -> GraphContext:
    """Assemble GraphContext from traversal results.

    Combines triplets (relationships) with passage texts and entity info.
    """
    triplets = []
    for rel in traversal_result.get("relationships", []):
        triplets.append({
            "source": rel.get("source", ""),
            "relation": rel.get("relation", ""),
            "target": rel.get("target", ""),
        })

    # Deduplicate triplets
    seen = set()
    unique_triplets = []
    for t in triplets:
        key = (t["source"], t["relation"], t["target"])
        if key not in seen:
            seen.add(key)
            unique_triplets.append(t)

    passages = [
        p["text"] for p in traversal_result.get("passage_nodes", [])
        if p.get("text")
    ]

    entities = [
        Entity(
            id=p.get("id", ""),
            name=p.get("name", ""),
            entity_type=p.get("entity_type", ""),
        )
        for p in traversal_result.get("phrase_nodes", [])
    ]

    source_ids = [
        p["chunk_id"] for p in traversal_result.get("passage_nodes", [])
        if p.get("chunk_id")
    ]

    return GraphContext(
        triplets=unique_triplets,
        passages=passages,
        entities=entities,
        source_ids=source_ids,
    )


# ---------------------------------------------------------------------------
# 4. Full VectorCypher search pipeline
# ---------------------------------------------------------------------------

def search(
    query_embedding: list[float],
    driver: Driver,
    top_k: int | None = None,
    max_hops: int | None = None,
    threshold: float | None = None,
) -> GraphContext:
    """Full VectorCypher retrieval pipeline.

    1. Find entry PhraseNodes via vector similarity
    2. Traverse graph from entries
    3. Collect and assemble context

    Returns GraphContext with triplets, passages, entities, source_ids.
    """
    # Step 1: Vector entry
    entries = find_entry_points(query_embedding, driver, top_k=top_k, threshold=threshold)

    if not entries:
        logger.warning("No entry points found for query")
        return GraphContext()

    entry_ids = [e["id"] for e in entries]

    # Step 2: Graph traversal
    traversal = traverse_graph(entry_ids, driver, max_hops=max_hops)

    # Step 3: Assemble context
    context = collect_context(traversal)

    logger.info(
        "VectorCypher search: %d entries → %d triplets, %d passages",
        len(entries), len(context.triplets), len(context.passages),
    )
    return context
