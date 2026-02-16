"""Self-reflective evaluation and retry query generation.

From RAG 2.0 — evaluates relevance of retrieved chunks (1-5 scale),
generates retry queries, and orchestrates the reflect-and-answer loop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.models import SearchResult

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


def evaluate_relevance(
    query: str, results: list[SearchResult], openai_client: OpenAI | None = None,
) -> float:
    """Evaluate how relevant retrieved chunks are to the query.

    Returns average relevance score (1-5 scale).
    """
    cfg = get_settings()
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=cfg.openai.api_key)

    if not results:
        logger.warning("No results to evaluate")
        return 0.0

    # Evaluate only top-5 results to keep prompt focused
    top_results = results[:5]
    context_chunks = []
    for i, result in enumerate(top_results, start=1):
        text = result.chunk.enriched_content or result.chunk.content
        context_chunks.append(f"[Chunk {i}]\n{text[:400]}")
    context = "\n\n".join(context_chunks)

    prompt = (
        f"You are evaluating search results for a RAG system.\n"
        f"Rate how relevant each chunk is to answering the query.\n"
        f"Scale: 1=irrelevant, 2=slightly relevant, 3=moderately relevant, "
        f"4=very relevant, 5=perfect match.\n"
        f"Return ONLY a comma-separated list of integer scores (one per chunk).\n\n"
        f"Query: {query}\n\nRetrieved chunks:\n{context}\n\nScores:"
    )

    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        scores_text = response.choices[0].message.content or ""
        scores: list[float] = []
        for s in scores_text.split(","):
            try:
                scores.append(float(s.strip()))
            except ValueError:
                scores.append(2.5)

        while len(scores) < len(results):
            scores.append(2.5)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info("Average relevance score: %.2f", avg_score)
        return avg_score

    except Exception as e:
        logger.error("Error evaluating relevance: %s", e)
        return 2.5


def generate_retry_query(
    query: str, results: list[SearchResult], openai_client: OpenAI | None = None,
) -> str:
    """Generate an improved query based on what was found (and what was missing)."""
    cfg = get_settings()
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=cfg.openai.api_key)

    if results:
        summary_parts = []
        for i, result in enumerate(results[:3], start=1):
            summary_parts.append(f"{i}. {result.chunk.enriched_content[:200]}...")
        summary = "\n".join(summary_parts)
    else:
        summary = "No relevant content found."

    prompt = (
        f"The search didn't find good results. Original query: {query}\n\n"
        f"Found content:\n{summary}\n\n"
        f"Generate a better search query that might find more relevant information."
    )

    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        retry_query = response.choices[0].message.content or query
        logger.info("Generated retry query: %s", retry_query)
        return retry_query

    except Exception as e:
        logger.error("Error generating retry query: %s", e)
        return query


def evaluate_completeness(
    query: str, answer: str, openai_client: OpenAI | None = None,
) -> bool:
    """Check whether the generated answer is complete for the query.

    Designed for GLOBAL/enumeration queries where partial answers are common.
    Returns True if the answer is considered complete.
    """
    cfg = get_settings()
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=cfg.openai.api_key)

    prompt = (
        "You are evaluating an answer for completeness.\n"
        f"Query: {query}\n\n"
        f"Answer: {answer}\n\n"
        "Is this answer COMPLETE? Does it cover ALL aspects of the query? "
        "Consider: does it list all items asked for? Does it acknowledge gaps?\n"
        "Respond with ONLY 'YES' or 'NO' followed by a brief explanation."
    )

    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model_mini,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip().upper()
        is_complete = text.startswith("YES")
        logger.info("Completeness check: %s", "complete" if is_complete else "incomplete")
        return is_complete

    except Exception as e:
        logger.error("Error evaluating completeness: %s", e)
        return True  # assume complete on error to avoid extra retries
