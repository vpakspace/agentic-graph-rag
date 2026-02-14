"""Batch embedding for chunks via OpenAI Embeddings API.

Extracted from RAG 2.0 enricher — dedicated module for embedding generation
using text-embedding-3-small (1536 dim by default).
"""

from __future__ import annotations

import logging

import openai

from rag_core.config import get_settings
from rag_core.models import Chunk

logger = logging.getLogger(__name__)


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Batch embed chunks using OpenAI Embeddings API.

    Uses enriched_content (context + content) if available.
    Sets chunk.embedding for each chunk.
    """
    if not chunks:
        return chunks

    cfg = get_settings()
    client = openai.OpenAI(api_key=cfg.openai.api_key)

    texts = [chunk.enriched_content for chunk in chunks]

    try:
        response = client.embeddings.create(
            model=cfg.openai.embedding_model,
            input=texts,
        )

        for i, chunk in enumerate(chunks):
            chunk.embedding = response.data[i].embedding

        logger.info("Embedded %d chunks (%s)", len(chunks), cfg.openai.embedding_model)

    except Exception as e:
        logger.error("Failed to embed chunks: %s", e)
        raise

    return chunks
