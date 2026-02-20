#!/usr/bin/env python3
"""Ingest documents into the Agentic Graph RAG pipeline.

Usage:
    python scripts/ingest.py <file_or_directory> [options]

Options:
    --skip-enrichment   Skip LLM contextual enrichment (faster, no OpenAI calls for enrichment)
    --skip-skeleton     Skip skeleton indexing (no entity extraction, just vector store)
    --use-gpu           Enable GPU acceleration for Docling document parsing

Examples:
    python scripts/ingest.py data/sample_graph_rag.txt
    python scripts/ingest.py data/sample_graph_rag.txt --skip-enrichment
    python scripts/ingest.py ~/documents/ --use-gpu
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pymangle"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ingest")


def ingest_file(
    file_path: str,
    *,
    skip_enrichment: bool = False,
    skip_skeleton: bool = False,
    use_gpu: bool = False,
) -> None:
    """Ingest a single file through the full pipeline."""
    from neo4j import GraphDatabase
    from rag_core.chunker import chunk_text
    from rag_core.config import get_settings, make_openai_client
    from rag_core.embedder import embed_chunks
    from rag_core.enricher import enrich_chunks
    from rag_core.loader import load_file
    from rag_core.vector_store import VectorStore

    cfg = get_settings()
    if not cfg.openai.api_key and not cfg.openai.base_url:
        logger.error("OPENAI_API_KEY is not set. Please configure it in .env")
        sys.exit(1)

    # 1. Load document
    logger.info("Loading: %s (GPU=%s)", file_path, use_gpu)
    text = load_file(file_path, use_gpu=use_gpu)
    logger.info("Loaded %d characters", len(text))

    if not text.strip():
        logger.warning("Document is empty, skipping: %s", file_path)
        return

    # 2. Chunk
    chunks = chunk_text(text)
    logger.info("Created %d chunks", len(chunks))

    # 3. Enrich (optional)
    if not skip_enrichment:
        logger.info("Enriching chunks with LLM context...")
        chunks = enrich_chunks(chunks)
        logger.info("Enrichment complete")
    else:
        logger.info("Skipping enrichment (--skip-enrichment)")

    # 4. Embed
    logger.info("Embedding %d chunks...", len(chunks))
    chunks = embed_chunks(chunks)
    logger.info("Embedding complete")

    # 5. Store in vector index
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    try:
        store = VectorStore(driver=driver)
        store.init_index()
        stored = store.store_chunks(chunks)
        logger.info("Stored %d chunks in Neo4j vector index", stored)

        # 6. Skeleton indexing (optional)
        if not skip_skeleton:
            from agentic_graph_rag.indexing.dual_node import build_dual_graph, embed_phrase_nodes
            from agentic_graph_rag.indexing.skeleton import build_skeleton_index

            openai_client = make_openai_client(cfg)
            embeddings = [c.embedding for c in chunks if c.embedding]

            logger.info("Building skeleton index...")
            entities, relationships, skeletal, peripheral = build_skeleton_index(
                chunks, embeddings, openai_client=openai_client,
            )
            logger.info(
                "Skeleton: %d entities, %d relationships (%d skeletal, %d peripheral)",
                len(entities), len(relationships), len(skeletal), len(peripheral),
            )

            if entities:
                logger.info("Building dual-node graph...")
                phrase_nodes, passage_nodes, link_count = build_dual_graph(
                    entities, chunks, driver, relationships=relationships,
                )
                logger.info(
                    "Dual graph: %d phrase nodes, %d passage nodes, %d links",
                    len(phrase_nodes), len(passage_nodes), link_count,
                )

                logger.info("Embedding phrase nodes...")
                updated = embed_phrase_nodes(phrase_nodes, driver, openai_client)
                logger.info("Updated %d phrase node embeddings", updated)
        else:
            logger.info("Skipping skeleton indexing (--skip-skeleton)")

    finally:
        driver.close()

    logger.info("Done: %s", file_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into Agentic Graph RAG",
    )
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--skip-enrichment", action="store_true", help="Skip LLM enrichment")
    parser.add_argument("--skip-skeleton", action="store_true", help="Skip skeleton indexing")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU for Docling")
    args = parser.parse_args()

    target = os.path.abspath(args.path)
    if not os.path.exists(target):
        logger.error("Path does not exist: %s", target)
        sys.exit(1)

    files: list[str] = []
    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            if os.path.isfile(full) and not name.startswith("."):
                files.append(full)

    if not files:
        logger.error("No files found at: %s", target)
        sys.exit(1)

    logger.info("Ingesting %d file(s)...", len(files))
    for f in files:
        ingest_file(
            f,
            skip_enrichment=args.skip_enrichment,
            skip_skeleton=args.skip_skeleton,
            use_gpu=args.use_gpu,
        )

    logger.info("All done. %d file(s) ingested.", len(files))


if __name__ == "__main__":
    main()
