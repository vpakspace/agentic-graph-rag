"""Agentic Query Router — classifies queries and selects retrieval tools.

Uses pattern matching for fast classification with optional LLM fallback.
Categories: simple, relation, multi_hop, global, temporal.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.models import QueryType, RouterDecision

if TYPE_CHECKING:
    from openai import OpenAI

    from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern-based classification rules
# ---------------------------------------------------------------------------

_RELATION_PATTERNS = [
    r"\bсвяз\w*\b", r"\bотношен\w*\b", r"\bсоедин\w*\b",
    r"\brelat\w*\b", r"\bconnect\w*\b", r"\blink\w*\b",
    r"\bbetween\b", r"\bмежду\b",
]

_MULTI_HOP_PATTERNS = [
    r"\bцепочк\w*\b", r"\bпуть\b", r"\bсравн\w*\b", r"\bчерез\b",
    r"\bchain\b", r"\bpath\b", r"\bcompar\w*\b", r"\bthrough\b",
    r"\bhow .+ affect\b", r"\bкак .+ влия\w*\b",
]

_GLOBAL_PATTERNS = [
    r"\bвсе\b", r"\bкажд\w*\b", r"\bобзор\b", r"\bсписок\b",
    r"\ball\b", r"\bevery\b", r"\boverview\b", r"\blist\b", r"\bsummar\w*\b",
    r"\bпокажи все\b", r"\bshow all\b",
]

_TEMPORAL_PATTERNS = [
    r"\bкогда\b", r"\bдата\b", r"\bвремя\b", r"\bисторi?\w*\b",
    r"\bwhen\b", r"\bdate\b", r"\btime\w*\b", r"\bhistor\w*\b",
    r"\bbefore\b", r"\bafter\b", r"\bдо\b", r"\bпосле\b",
    r"\b\d{4}[-/]\d{2}\b",  # date pattern YYYY-MM
]

# Tool mapping per query type
_TOOL_MAP: dict[QueryType, str] = {
    QueryType.SIMPLE: "vector_search",
    QueryType.RELATION: "cypher_traverse",
    QueryType.MULTI_HOP: "cypher_traverse",
    QueryType.GLOBAL: "full_document_read",
    QueryType.TEMPORAL: "temporal_query",
}


def _match_patterns(query: str, patterns: list[str]) -> int:
    """Count how many patterns match in the query."""
    count = 0
    for pat in patterns:
        if re.search(pat, query, re.IGNORECASE):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Pattern-based classification (fast, no LLM)
# ---------------------------------------------------------------------------

def classify_query_by_patterns(query: str) -> RouterDecision:
    """Classify query using regex pattern matching.

    Returns RouterDecision with confidence based on match count.
    """
    scores: dict[QueryType, int] = {
        QueryType.TEMPORAL: _match_patterns(query, _TEMPORAL_PATTERNS),
        QueryType.MULTI_HOP: _match_patterns(query, _MULTI_HOP_PATTERNS),
        QueryType.RELATION: _match_patterns(query, _RELATION_PATTERNS),
        QueryType.GLOBAL: _match_patterns(query, _GLOBAL_PATTERNS),
    }

    best_type = max(scores, key=lambda k: scores[k])
    best_count = scores[best_type]

    if best_count == 0:
        query_type = QueryType.SIMPLE
        confidence = 0.5
        reasoning = "No specific patterns matched; defaulting to simple vector search."
    else:
        query_type = best_type
        confidence = min(0.5 + best_count * 0.2, 0.95)
        reasoning = f"Matched {best_count} {query_type.value} pattern(s)."

    return RouterDecision(
        query_type=query_type,
        confidence=confidence,
        reasoning=reasoning,
        suggested_tool=_TOOL_MAP[query_type],
    )


# ---------------------------------------------------------------------------
# LLM-based classification (high-quality, slower)
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """Classify the following query into exactly ONE category:
- simple: factual lookup, single entity question
- relation: asks about relationships between entities
- multi_hop: requires traversing multiple connections or comparing entities
- global: asks about all/every/overview of something
- temporal: asks about time, dates, history, changes over time

Query: {query}

Respond with ONLY the category name (simple/relation/multi_hop/global/temporal):"""


def classify_query_by_llm(
    query: str, openai_client: OpenAI | None = None,
) -> RouterDecision:
    """Classify query using LLM for higher accuracy."""
    cfg = get_settings()
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=cfg.openai.api_key)

    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model_mini,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(query=query)}],
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "simple").strip().lower()

        type_map = {
            "simple": QueryType.SIMPLE,
            "relation": QueryType.RELATION,
            "multi_hop": QueryType.MULTI_HOP,
            "global": QueryType.GLOBAL,
            "temporal": QueryType.TEMPORAL,
        }
        query_type = type_map.get(raw, QueryType.SIMPLE)

        return RouterDecision(
            query_type=query_type,
            confidence=0.85,
            reasoning=f"LLM classified as '{raw}'.",
            suggested_tool=_TOOL_MAP[query_type],
        )

    except Exception as e:
        logger.error("LLM classification failed: %s — falling back to patterns", e)
        return classify_query_by_patterns(query)


# ---------------------------------------------------------------------------
# Mangle-based classification
# ---------------------------------------------------------------------------

_MANGLE_TOOL_TO_TYPE: dict[str, QueryType] = {
    "vector_search": QueryType.SIMPLE,
    "cypher_traverse": QueryType.RELATION,
    "full_document_read": QueryType.GLOBAL,
    "temporal_query": QueryType.TEMPORAL,
}


def _classify_by_mangle(query: str, reasoning: ReasoningEngine) -> RouterDecision | None:
    """Attempt classification via Mangle rules. Returns None if no match."""
    result = reasoning.classify_query(query)
    if result is None:
        return None

    tool = result["tool"]
    query_type = _MANGLE_TOOL_TO_TYPE.get(tool, QueryType.SIMPLE)

    return RouterDecision(
        query_type=query_type,
        confidence=0.7,
        reasoning=f"Mangle rule matched → {tool}.",
        suggested_tool=tool,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_query(
    query: str,
    use_llm: bool = False,
    openai_client: OpenAI | None = None,
    reasoning: ReasoningEngine | None = None,
) -> RouterDecision:
    """Classify query and suggest retrieval tool.

    When a ReasoningEngine is provided, Mangle rules are tried first.
    Falls back to patterns (or LLM) if Mangle produces no match.
    """
    if reasoning is not None:
        mangle_result = _classify_by_mangle(query, reasoning)
        if mangle_result is not None:
            return mangle_result
        logger.debug("Mangle produced no match for query, falling back to patterns")

    if use_llm:
        return classify_query_by_llm(query, openai_client=openai_client)
    return classify_query_by_patterns(query)
