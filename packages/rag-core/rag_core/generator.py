"""LLM answer generation from retrieved chunks.

From RAG 2.0 — generates answers using OpenAI chat completions
with context from retrieved search results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings, make_openai_client
from rag_core.models import QAResult, SearchResult

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_ENUM_RE = None


def _is_enumeration_query(query: str) -> bool:
    """Detect enumeration/global queries that need comprehensive listing."""
    global _ENUM_RE  # noqa: PLW0603
    if _ENUM_RE is None:
        import re
        _ENUM_RE = re.compile(
            r'\b('
            r'все\b|всех\b|всё\b|перечисл|опиши все|резюмируй все|обзор\b'
            r'|list all|describe all|summarize all|overview|every\b'
            r'|все компоненты|все методы|все слои|все решения|семь\b|seven\b'
            r'|all components|all layers|all methods|all decisions'
            r')\b',
            re.IGNORECASE,
        )
    return bool(_ENUM_RE.search(query))


def generate_answer(
    query: str, results: list[SearchResult], openai_client: OpenAI | None = None,
) -> QAResult:
    """Generate answer from query and retrieved chunks using LLM."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

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

    # Detect enumeration/global queries for specialized prompt
    is_enumeration = _is_enumeration_query(query)

    if is_enumeration:
        system_prompt = (
            "You are an expert Q&A assistant specialized in comprehensive enumeration. "
            "Your task is to extract and list EVERY distinct item, component, decision, method, "
            "or concept mentioned across ALL provided context chunks.\n\n"
            "INSTRUCTIONS:\n"
            "1. Scan ALL chunks systematically — do not stop at the first few\n"
            "2. Create a NUMBERED LIST of every distinct item found\n"
            "3. For each item, provide a brief description (1-2 sentences)\n"
            "4. Combine information from multiple chunks about the same item\n"
            "5. Do NOT say 'the document does not list' — extract items even if "
            "they are discussed narratively rather than listed explicitly\n"
            "6. Answer in the same language as the query"
        )
    else:
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
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        logger.info("Generated answer: %s", answer_text[:100])

        avg_score = sum(r.score for r in results) / len(results)
        confidence = min(1.0, max(0.1, avg_score))

        return QAResult(
            answer=answer_text,
            sources=results,
            confidence=confidence,
            query=query,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return QAResult(
            answer=f"Error generating answer: {e}",
            sources=results,
            confidence=0.0,
            query=query,
        )
