"""Knowledge Graph client wrapping Graphiti and direct Neo4j Cypher.

From rag-temporal — Graphiti episode ingestion, fact search,
temporal Cypher queries with COALESCE for Graphiti property mismatch.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from rag_core.config import get_settings

logger = logging.getLogger(__name__)

# Lucene special characters that must be sanitized
_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')

# Default episode limits
DEFAULT_MAX_EPISODE_CHARS = 8_000
DEFAULT_EPISODE_OVERLAP = 200
DEFAULT_SEARCH_RESULTS = 10
DEFAULT_FACT_SCORE = 0.8


def sanitize_lucene(text: str) -> str:
    """Remove Lucene special characters (replace with spaces)."""
    return _LUCENE_SPECIAL.sub(" ", text)


def split_episodes(
    text: str,
    max_chars: int = DEFAULT_MAX_EPISODE_CHARS,
    overlap: int = DEFAULT_EPISODE_OVERLAP,
) -> list[str]:
    """Split text into episodes if it exceeds max_chars.

    Splits on paragraph boundaries when possible.
    """
    if len(text) <= max_chars:
        return [text]

    episodes: list[str] = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                episodes.append(current)
                if overlap > 0:
                    current = current[-overlap:] + "\n\n" + para
                else:
                    current = para
            else:
                while len(para) > max_chars:
                    episodes.append(para[:max_chars])
                    para = para[max_chars - overlap:] if overlap else para[max_chars:]
                current = para

    if current:
        episodes.append(current)

    return episodes


@dataclass
class KGFact:
    """A fact from the knowledge graph."""

    content: str
    source: str = "kg"
    valid_at: str = ""
    entity_name: str = ""
    score: float = DEFAULT_FACT_SCORE
    metadata: dict = field(default_factory=dict)


class KGClient:
    """Knowledge Graph client using Graphiti + direct Cypher."""

    def __init__(self) -> None:
        self._graphiti: Any = None
        self._driver: Any = None

    async def connect(self) -> None:
        """Initialize Graphiti client and Neo4j driver."""
        from graphiti_core import Graphiti
        from neo4j import GraphDatabase

        cfg = get_settings()
        self._driver = GraphDatabase.driver(
            cfg.neo4j.uri,
            auth=(cfg.neo4j.user, cfg.neo4j.password),
        )
        self._graphiti = Graphiti(
            cfg.neo4j.uri,
            cfg.neo4j.user,
            cfg.neo4j.password,
        )
        logger.info("KGClient connected to %s", cfg.neo4j.uri)

    async def close(self) -> None:
        """Close connections."""
        if self._graphiti:
            await self._graphiti.close()
        if self._driver:
            self._driver.close()
        logger.info("KGClient closed")

    async def ingest_text(self, text: str, source: str = "document") -> int:
        """Ingest text into knowledge graph via Graphiti.

        Splits large documents into episodes (>8K chars).
        Returns number of episodes ingested.
        """
        if not self._graphiti:
            raise RuntimeError("KGClient not connected. Call connect() first.")

        from graphiti_core.nodes import EpisodeType

        sanitized = sanitize_lucene(text)
        episodes = split_episodes(sanitized)

        for i, episode_text in enumerate(episodes):
            episode_name = f"{source}_part_{i + 1}" if len(episodes) > 1 else source
            try:
                await self._graphiti.add_episode(
                    name=episode_name,
                    episode_body=episode_text,
                    source_description=source,
                    episode_type=EpisodeType.text,
                )
                logger.info("Ingested episode %d/%d (%d chars)", i + 1, len(episodes), len(episode_text))
            except Exception as e:
                logger.error("Failed to ingest episode %d: %s", i + 1, e)

        return len(episodes)

    async def search(self, query: str, num_results: int = DEFAULT_SEARCH_RESULTS) -> list[KGFact]:
        """Search knowledge graph for facts related to query."""
        if not self._graphiti:
            raise RuntimeError("KGClient not connected. Call connect() first.")

        sanitized = sanitize_lucene(query)

        try:
            edges = await self._graphiti.search(sanitized, num_results=num_results)
            facts = []
            for edge in edges:
                fact_text = getattr(edge, "fact", "") or ""
                valid_at = str(getattr(edge, "valid_at", "")) if getattr(edge, "valid_at", None) else ""
                facts.append(KGFact(content=fact_text, valid_at=valid_at))
            logger.info("KG search returned %d facts for: %s", len(facts), query[:50])
            return facts
        except Exception as e:
            logger.error("KG search failed: %s", e)
            return []

    def temporal_query(
        self,
        date_from: str = "",
        date_to: str = "",
        entity_name: str = "",
    ) -> list[KGFact]:
        """Direct Cypher query for temporal facts.

        Uses COALESCE for Graphiti property mismatch (uuid vs id, labels vs entity_type).
        """
        if not self._driver:
            raise RuntimeError("KGClient not connected. Call connect() first.")

        where_parts: list[str] = []
        params: dict[str, Any] = {}

        if date_from:
            where_parts.append("r.valid_at >= $date_from")
            params["date_from"] = date_from
        if date_to:
            where_parts.append("r.valid_at <= $date_to")
            params["date_to"] = date_to
        if entity_name:
            where_parts.append("(COALESCE(e.name, e.id, e.uuid) CONTAINS $entity_name)")
            params["entity_name"] = entity_name

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        cypher = f"""
        MATCH (e)-[r]->(t)
        {where_clause}
        RETURN
            COALESCE(e.name, e.id, e.uuid) AS entity,
            COALESCE(e.entity_type, head(labels(e))) AS entity_type,
            type(r) AS rel_type,
            COALESCE(r.fact, r.name, '') AS fact,
            COALESCE(toString(r.valid_at), '') AS valid_at,
            COALESCE(t.name, t.id, t.uuid, '') AS target
        ORDER BY r.valid_at DESC
        LIMIT 50
        """

        facts: list[KGFact] = []
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            for record in result:
                fact_text = record["fact"] or f"{record['entity']} -> {record['target']}"
                facts.append(KGFact(
                    content=fact_text,
                    entity_name=record["entity"] or "",
                    valid_at=record["valid_at"] or "",
                    metadata={
                        "entity_type": record["entity_type"] or "",
                        "rel_type": record["rel_type"] or "",
                        "target": record["target"] or "",
                    },
                ))

        logger.info("Temporal query returned %d facts", len(facts))
        return facts

    def entity_count(self) -> int:
        """Count total entities in knowledge graph."""
        if not self._driver:
            return 0
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE n:Entity OR n:TemporalEvent OR any(l IN labels(n) WHERE l <> 'RagChunk') "
                "RETURN count(n) AS total"
            )
            record = result.single()
            return record["total"] if record else 0

    def get_entities(self, limit: int = 50) -> list[dict]:
        """Get entities from knowledge graph with COALESCE for Graphiti mismatch."""
        if not self._driver:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e)
                WHERE e:Entity OR any(l IN labels(e) WHERE l <> 'RagChunk' AND l <> 'Episodic')
                RETURN
                    COALESCE(e.id, e.uuid) AS id,
                    COALESCE(e.name, '') AS name,
                    COALESCE(e.entity_type, head(labels(e))) AS entity_type,
                    COALESCE(e.summary, '') AS summary
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(record) for record in result]

    def get_relationships(self, limit: int = 50) -> list[dict]:
        """Get relationships from knowledge graph."""
        if not self._driver:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s)-[r]->(t)
                WHERE NOT s:RagChunk AND NOT t:RagChunk
                RETURN
                    COALESCE(s.name, s.id, s.uuid) AS source,
                    type(r) AS rel_type,
                    COALESCE(r.fact, r.name, type(r)) AS fact,
                    COALESCE(t.name, t.id, t.uuid) AS target,
                    COALESCE(toString(r.valid_at), '') AS valid_at
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(record) for record in result]
