"""LLM answer generation from retrieved chunks.

From RAG 2.0 — generates answers using OpenAI chat completions
with context from retrieved search results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.models import QAResult, SearchResult

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


def generate_answer(
    query: str, results: list[SearchResult], openai_client: OpenAI | None = None,
) -> QAResult:
    """Generate answer from query and retrieved chunks using LLM."""
    cfg = get_settings()
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=cfg.openai.api_key)

    if not results:
        logger.warning("No results provided for answer generation")
        return QAResult(
            answer="I don't have enough context to answer this question.",
            sources=[],
            confidence=0.0,
            query=query,
        )

    context_chunks = []
    for i, result in enumerate(results, start=1):
        context_chunks.append(f"[Chunk {i}]\n{result.chunk.enriched_content}")
    context = "\n\n".join(context_chunks)

    system_prompt = (
        "You are a knowledgeable Q&A assistant. Synthesize information from ALL provided "
        "context chunks to give a comprehensive answer. Combine facts from different chunks "
        "when needed. If some details are missing, answer with what IS available rather than "
        "refusing. Cite chunk numbers used."
    )

    user_prompt = f"Query: {query}\n\nContext:\n{context}\n\nPlease provide an answer based on the above context."

    logger.info("Generating answer for query: %s", query)
    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=cfg.openai.llm_temperature,
        )

        answer_text = response.choices[0].message.content or ""
        logger.info("Generated answer: %s", answer_text[:100])

        avg_score = sum(r.score for r in results) / len(results)
        confidence = min(1.0, max(0.1, avg_score))

        return QAResult(
            answer=answer_text,
            sources=results,
            confidence=confidence,
            query=query,
        )

    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return QAResult(
            answer=f"Error generating answer: {e}",
            sources=results,
            confidence=0.0,
            query=query,
        )
