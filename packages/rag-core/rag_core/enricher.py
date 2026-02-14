"""Contextual enrichment for chunks via LLM.

From RAG 2.0 — generates per-chunk context explaining its role
within the document using OpenAI chat completions.
"""

from __future__ import annotations

import logging
import time

import openai

from rag_core.config import get_settings
from rag_core.models import Chunk

logger = logging.getLogger(__name__)


def enrich_chunks(
    chunks: list[Chunk], document_summary: str = "",
) -> list[Chunk]:
    """Enrich chunks with contextual information via LLM.

    If no document_summary provided, generates one from first few chunks.
    For each chunk, calls OpenAI to generate 1-2 sentence context.
    Sets chunk.context = LLM response.
    """
    if not chunks:
        return chunks

    cfg = get_settings()
    client = openai.OpenAI(api_key=cfg.openai.api_key)

    if not document_summary:
        document_summary = _generate_summary(chunks[:3], client, cfg.openai.llm_model)
        logger.info("Generated document summary: %s", document_summary[:100])

    enriched: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        try:
            context = _generate_context(
                chunk.content, document_summary, client, cfg.openai.llm_model,
            )
            chunk.context = context
            logger.debug("Enriched chunk %d/%d: %s", i + 1, len(chunks), context[:50])

            if i < len(chunks) - 1:
                time.sleep(0.1)

        except Exception as e:
            logger.warning("Failed to enrich chunk %d: %s", i, e)

        enriched.append(chunk)

    logger.info("Enriched %d chunks", len(enriched))
    return enriched


def _generate_summary(
    chunks: list[Chunk], client: openai.OpenAI, model: str,
) -> str:
    """Generate document summary from first few chunks."""
    combined = "\n\n".join(c.content for c in chunks)

    prompt = (
        f"Here are the first few sections of a document:\n\n"
        f"{combined[:2000]}\n\n"
        f"Write 2-3 sentences summarizing what this document is about."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content or "Unknown document"
    except Exception as e:
        logger.warning("Failed to generate summary: %s", e)
        return "Document"


def _generate_context(
    chunk_content: str, document_summary: str, client: openai.OpenAI, model: str,
) -> str:
    """Generate 1-2 sentence context for a chunk."""
    prompt = (
        f"Here's the document: {document_summary}\n\n"
        f"Here's a chunk from the document:\n\n"
        f"{chunk_content[:500]}\n\n"
        f"Write 1-2 sentences explaining the context of this chunk "
        f"within the document. Be specific and concise."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Failed to generate context: %s", e)
        return ""
