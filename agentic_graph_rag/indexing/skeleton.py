"""KET-RAG Skeleton Indexer.

PageRank-based selective entity extraction: build a KNN graph over chunk
embeddings, rank chunks by PageRank, extract entities (LLM) from the top-β
"skeletal" chunks only, and cheaply link peripheral chunks via keywords.

Reference: KET-RAG (KDD 2025) — 10x cost reduction, +32.4% generation quality.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from rag_core.config import get_settings
from rag_core.models import Chunk, Entity, Relationship

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. KNN graph construction
# ---------------------------------------------------------------------------

def build_knn_graph(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    k: int | None = None,
) -> nx.DiGraph:
    """Build a directed KNN graph over chunks using cosine similarity.

    Each chunk connects to its *k* nearest neighbours (by embedding similarity).
    """
    if k is None:
        k = get_settings().indexing.knn_k

    n = len(chunks)
    if n == 0:
        return nx.DiGraph()

    emb_matrix = np.array(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    normed = emb_matrix / norms

    # Cosine similarity matrix
    sim_matrix = normed @ normed.T

    graph = nx.DiGraph()
    for i in range(n):
        graph.add_node(i, chunk_id=chunks[i].id)

    effective_k = min(k, n - 1)
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0  # exclude self
        if effective_k > 0:
            top_indices = np.argsort(sims)[-effective_k:][::-1]
            for j_idx in top_indices:
                j = int(j_idx)
                graph.add_edge(i, j, weight=float(sims[j]))

    logger.info("Built KNN graph: %d nodes, %d edges (k=%d)", n, graph.number_of_edges(), effective_k)
    return graph


# ---------------------------------------------------------------------------
# 2. PageRank computation
# ---------------------------------------------------------------------------

def compute_pagerank(
    knn_graph: nx.DiGraph,
    damping: float | None = None,
) -> dict[int, float]:
    """Compute PageRank scores for chunk nodes.

    Returns mapping of node index → PageRank score.
    """
    if damping is None:
        damping = get_settings().indexing.pagerank_damping

    if knn_graph.number_of_nodes() == 0:
        return {}

    scores: dict[int, float] = nx.pagerank(knn_graph, alpha=damping, weight="weight")
    logger.debug("PageRank computed for %d nodes", len(scores))
    return scores


# ---------------------------------------------------------------------------
# 3. Skeletal chunk selection
# ---------------------------------------------------------------------------

def select_skeletal_chunks(
    chunks: list[Chunk],
    pagerank_scores: dict[int, float],
    beta: float | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Split chunks into skeletal (top-β) and peripheral.

    Returns (skeletal_chunks, peripheral_chunks).
    """
    if beta is None:
        beta = get_settings().indexing.skeleton_beta

    if not chunks or not pagerank_scores:
        return [], list(chunks)

    # Sort node indices by descending PageRank
    ranked = sorted(pagerank_scores.items(), key=lambda kv: kv[1], reverse=True)

    n_skeletal = max(1, int(len(chunks) * beta))

    skeletal_indices = {idx for idx, _ in ranked[:n_skeletal]}

    skeletal: list[Chunk] = []
    peripheral: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        if i in skeletal_indices:
            skeletal.append(chunk)
        else:
            peripheral.append(chunk)

    logger.info(
        "Selected %d skeletal + %d peripheral chunks (beta=%.2f)",
        len(skeletal), len(peripheral), beta,
    )
    return skeletal, peripheral


# ---------------------------------------------------------------------------
# 4. Full LLM entity extraction (for skeletal chunks only)
# ---------------------------------------------------------------------------

def extract_entities_full(
    skeletal_chunks: list[Chunk],
    openai_client: OpenAI | None = None,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from skeletal chunks using LLM.

    This is the expensive operation — applied only to top-β chunks.
    """
    cfg = get_settings()
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=cfg.openai.api_key)

    if not skeletal_chunks:
        return [], []

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    system_prompt = (
        "You are an entity extraction expert. "
        "From the given text, extract entities and relationships.\n\n"
        "Return in this exact format (one per line):\n"
        "ENTITY: <name> | <type> | <description>\n"
        "RELATIONSHIP: <source> | <relation> | <target>\n\n"
        "Extract all important concepts, people, organizations, technical terms."
    )

    for chunk in skeletal_chunks:
        try:
            response = openai_client.chat.completions.create(
                model=cfg.openai.llm_model_mini,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk.enriched_content[:2000]},
                ],
                temperature=0.0,
            )

            text = response.choices[0].message.content or ""
            entities, rels = _parse_extraction_response(text, chunk.id)
            all_entities.extend(entities)
            all_relationships.extend(rels)

        except Exception as e:
            logger.error("Entity extraction failed for chunk %s: %s", chunk.id, e)

    logger.info(
        "Extracted %d entities, %d relationships from %d skeletal chunks",
        len(all_entities), len(all_relationships), len(skeletal_chunks),
    )
    return all_entities, all_relationships


def _parse_extraction_response(
    text: str, source_chunk_id: str,
) -> tuple[list[Entity], list[Relationship]]:
    """Parse LLM extraction output into Entity and Relationship objects."""
    entities: list[Entity] = []
    relationships: list[Relationship] = []

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("ENTITY:"):
            parts = [p.strip() for p in line[len("ENTITY:"):].split("|")]
            if len(parts) >= 2:
                ent_id = hashlib.md5(parts[0].lower().encode()).hexdigest()[:8]
                entities.append(Entity(
                    id=ent_id,
                    name=parts[0],
                    entity_type=parts[1] if len(parts) > 1 else "",
                    description=parts[2] if len(parts) > 2 else "",
                    metadata={"source_chunk": source_chunk_id},
                ))
        elif line.startswith("RELATIONSHIP:"):
            parts = [p.strip() for p in line[len("RELATIONSHIP:"):].split("|")]
            if len(parts) >= 3:
                rel_id = hashlib.md5(
                    f"{parts[0]}:{parts[1]}:{parts[2]}".lower().encode()
                ).hexdigest()[:8]
                relationships.append(Relationship(
                    id=rel_id,
                    source=parts[0],
                    target=parts[2],
                    relation_type=parts[1],
                    metadata={"source_chunk": source_chunk_id},
                ))

    return entities, relationships


# ---------------------------------------------------------------------------
# 5. Keyword-based peripheral linking (cheap, no LLM)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "and", "or", "but",
    "if", "then", "than", "that", "this", "it", "its", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "into", "about",
    "not", "no", "so", "up", "out", "all", "also", "very", "just", "how",
})


def link_peripheral_keywords(
    peripheral_chunks: list[Chunk],
    existing_entities: list[Entity],
) -> list[Relationship]:
    """Link peripheral chunks to existing entities via keyword matching.

    No LLM calls — uses simple case-insensitive string matching.
    Returns MENTIONED_IN-style relationships (entity → chunk).
    """
    if not peripheral_chunks or not existing_entities:
        return []

    entity_names_lower = {e.name.lower(): e for e in existing_entities}
    relationships: list[Relationship] = []

    for chunk in peripheral_chunks:
        text_lower = chunk.enriched_content.lower()
        for name_lower, entity in entity_names_lower.items():
            if len(name_lower) < 2:
                continue
            if name_lower in text_lower:
                rel_id = hashlib.md5(
                    f"{entity.id}:mentioned_in:{chunk.id}".encode()
                ).hexdigest()[:8]
                relationships.append(Relationship(
                    id=rel_id,
                    source=entity.name,
                    target=chunk.id,
                    relation_type="MENTIONED_IN",
                    metadata={"method": "keyword"},
                ))

    logger.info(
        "Linked %d peripheral mentions across %d chunks",
        len(relationships), len(peripheral_chunks),
    )
    return relationships


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract keywords from text (simple tokenizer, no LLM)."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS]
    # Frequency-based selection
    freq: dict[str, int] = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:max_keywords]]


# ---------------------------------------------------------------------------
# 6. Orchestrator
# ---------------------------------------------------------------------------

def build_skeleton_index(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    openai_client: OpenAI | None = None,
) -> tuple[list[Entity], list[Relationship], list[Chunk], list[Chunk]]:
    """Full KET-RAG skeleton indexing pipeline.

    1. Build KNN graph from embeddings
    2. Compute PageRank to rank chunks
    3. Select top-β as skeletal chunks
    4. Extract entities (LLM) from skeletal chunks
    5. Link peripheral chunks via keywords

    Returns (entities, relationships, skeletal_chunks, peripheral_chunks).
    """
    if not chunks or not embeddings:
        return [], [], [], []

    # Step 1-2: KNN graph + PageRank
    knn_graph = build_knn_graph(chunks, embeddings)
    pagerank_scores = compute_pagerank(knn_graph)

    # Step 3: Select skeletal chunks
    skeletal, peripheral = select_skeletal_chunks(chunks, pagerank_scores)

    # Step 4: LLM entity extraction on skeletal chunks
    entities, relationships = extract_entities_full(skeletal, openai_client)

    # Step 5: Keyword linking for peripheral chunks
    peripheral_rels = link_peripheral_keywords(peripheral, entities)
    relationships.extend(peripheral_rels)

    logger.info(
        "Skeleton index built: %d entities, %d relationships "
        "(%d skeletal + %d peripheral chunks)",
        len(entities), len(relationships), len(skeletal), len(peripheral),
    )
    return entities, relationships, skeletal, peripheral
