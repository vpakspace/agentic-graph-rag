"""Neo4j Vector Index store for RAG.

From RAG 2.0 — CRUD operations on Neo4j vector index with cosine similarity search.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.models import Chunk, SearchResult

if TYPE_CHECKING:
    from neo4j import Driver

logger = logging.getLogger(__name__)

INDEX_NAME = "rag_chunks_index"
NODE_LABEL = "RagChunk"
EMBEDDING_PROPERTY = "embedding"


class VectorStore:
    """Neo4j-backed vector store with cosine similarity search."""

    def __init__(self, driver: Driver | None = None) -> None:
        cfg = get_settings()
        if driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                cfg.neo4j.uri,
                auth=(cfg.neo4j.user, cfg.neo4j.password),
            )
        else:
            self._driver = driver

    def close(self) -> None:
        self._driver.close()

    def init_index(self) -> None:
        """Create vector index in Neo4j if it doesn't exist."""
        cfg = get_settings()
        with self._driver.session() as session:
            session.run(
                f"""
                CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
                FOR (n:{NODE_LABEL})
                ON (n.{EMBEDDING_PROPERTY})
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $dimensions,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                dimensions=cfg.openai.embedding_dimensions,
            )
        logger.info("Vector index '%s' initialized", INDEX_NAME)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Store chunks as Neo4j nodes with embeddings. Returns count added."""
        if not chunks:
            return 0

        with self._driver.session() as session:
            for chunk in chunks:
                chunk_id = chunk.id or hashlib.md5(chunk.content.encode()).hexdigest()
                session.run(
                    f"""
                    MERGE (c:{NODE_LABEL} {{id: $id}})
                    SET c.content = $content,
                        c.context = $context,
                        c.enriched_content = $enriched_content,
                        c.{EMBEDDING_PROPERTY} = $embedding,
                        c.metadata = $metadata_json
                    """,
                    id=chunk_id,
                    content=chunk.content,
                    context=chunk.context,
                    enriched_content=chunk.enriched_content,
                    embedding=chunk.embedding,
                    metadata_json=str(chunk.metadata),
                )

        logger.info("Added %d chunks to vector store", len(chunks))
        return len(chunks)

    def search(
        self, query_embedding: list[float], top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search vector index by cosine similarity."""
        if top_k is None:
            top_k = get_settings().retrieval.top_k_vector

        with self._driver.session() as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes(
                    '{INDEX_NAME}', $top_k, $embedding
                )
                YIELD node, score
                RETURN node.id AS id,
                       node.content AS content,
                       node.context AS context,
                       score
                ORDER BY score DESC
                """,
                top_k=top_k,
                embedding=query_embedding,
            )

            results = []
            for i, record in enumerate(result):
                chunk = Chunk(
                    id=record["id"] or "",
                    content=record["content"] or "",
                    context=record["context"] or "",
                )
                results.append(
                    SearchResult(chunk=chunk, score=record["score"], rank=i + 1)
                )

        return results

    def delete_all(self) -> int:
        """Delete all RagChunk nodes. Returns count deleted."""
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (c:{NODE_LABEL})
                WITH c, count(c) AS total
                DETACH DELETE c
                RETURN total
                """
            )
            record = result.single()
            count = record["total"] if record else 0

        logger.info("Deleted %d chunks from vector store", count)
        return count

    def count(self) -> int:
        """Return total number of chunks."""
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (c:{NODE_LABEL}) RETURN count(c) AS total"
            )
            record = result.single()
            return record["total"] if record else 0
