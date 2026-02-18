"""Query expansion via LLM for improved retrieval.

From RAG 2.0 — expands short queries into detailed technical questions
and generates alternative query formulations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings, make_openai_client

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


def expand_query(query: str, openai_client: OpenAI | None = None) -> str:
    """Expand short query into detailed technical question using LLM."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    system_prompt = (
        "You are a query expansion expert. "
        "Expand the user's short query into a detailed, comprehensive technical question. "
        "Add relevant context, clarify ambiguous terms, and make implicit requirements explicit. "
        "Return only the expanded query without explanations."
    )

    response = openai_client.chat.completions.create(
        model=cfg.openai.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
    )

    expanded = response.choices[0].message.content or query
    logger.debug("Expanded query: %s -> %s", query, expanded)
    return expanded.strip()


def generate_multi_queries(
    query: str, n: int = 3, openai_client: OpenAI | None = None,
) -> list[str]:
    """Generate n alternative formulations of the query using LLM."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    system_prompt = (
        f"You are a query reformulation expert. "
        f"Generate {n} alternative formulations of the user's query. "
        f"Each formulation should capture the same intent but use different wording, "
        f"perspectives, or technical terminology. "
        f"Return exactly {n} queries, one per line, without numbering or explanations."
    )

    response = openai_client.chat.completions.create(
        model=cfg.openai.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content or ""
    queries = [q.strip() for q in content.strip().split("\n") if q.strip()]

    if len(queries) < n:
        queries.extend([query] * (n - len(queries)))
    elif len(queries) > n:
        queries = queries[:n]

    logger.debug("Generated %d multi-queries for: %s", len(queries), query)
    return queries
