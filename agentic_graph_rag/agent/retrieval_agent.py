"""Agentic Retrieval Agent — query routing + self-correction loop.

Main entry point for the Agentic Graph RAG system.
Routes queries to appropriate tools, evaluates results,
and retries with different strategies when quality is low.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.generator import generate_answer
from rag_core.models import QAResult, QueryType, RouterDecision, SearchResult
from rag_core.reflector import evaluate_completeness, evaluate_relevance, generate_retry_query

from agentic_graph_rag.agent.router import classify_query
from agentic_graph_rag.agent.tools import (
    community_search,
    comprehensive_search,
    cypher_traverse,
    full_document_read,
    hybrid_search,
    temporal_query,
    vector_search,
)

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

    from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

logger = logging.getLogger(__name__)

# Tool registry: query_type → tool function
_TOOL_REGISTRY = {
    "vector_search": vector_search,
    "cypher_traverse": cypher_traverse,
    "community_search": community_search,
    "hybrid_search": hybrid_search,
    "temporal_query": temporal_query,
    "comprehensive_search": comprehensive_search,
    "full_document_read": full_document_read,
}

# Escalation chain: if current tool fails, try next
_ESCALATION_CHAIN = [
    "vector_search",
    "cypher_traverse",
    "hybrid_search",
    "comprehensive_search",
    "full_document_read",
]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

def select_tool(decision: RouterDecision) -> str:
    """Select the retrieval tool name based on router decision."""
    tool_name = decision.suggested_tool
    if tool_name in _TOOL_REGISTRY:
        return tool_name
    logger.warning("Unknown tool '%s', defaulting to vector_search", tool_name)
    return "vector_search"


# ---------------------------------------------------------------------------
# Self-correction loop
# ---------------------------------------------------------------------------

def self_correction_loop(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    decision: RouterDecision,
    max_retries: int | None = None,
    relevance_threshold: float | None = None,
) -> tuple[list[SearchResult], int]:
    """Execute retrieval with self-correction.

    Tries the suggested tool first, evaluates relevance,
    and escalates to more powerful tools if quality is low.

    Returns (results, retries_used).
    """
    cfg = get_settings()
    if max_retries is None:
        max_retries = cfg.agent.max_retries
    if relevance_threshold is None:
        relevance_threshold = cfg.agent.relevance_threshold

    tool_name = select_tool(decision)
    tried_tools: set[str] = set()
    best_results: list[SearchResult] = []
    best_score: float = 0.0
    best_attempt: int = 0

    for attempt in range(max_retries + 1):
        # Execute tool
        tool_fn = _TOOL_REGISTRY[tool_name]
        results = tool_fn(query, driver, openai_client)
        tried_tools.add(tool_name)

        if not results:
            logger.warning("Tool '%s' returned no results (attempt %d)", tool_name, attempt + 1)
        else:
            # Evaluate relevance
            score = evaluate_relevance(query, results, openai_client=openai_client)
            logger.info(
                "Relevance score: %.2f (threshold: %.2f, tool: %s, attempt: %d)",
                score, relevance_threshold, tool_name, attempt + 1,
            )

            # Track best results across attempts
            if score > best_score:
                best_results = results
                best_score = score
                best_attempt = attempt

            if score >= relevance_threshold:
                return results, attempt

        # Escalate to next tool, rephrasing query first
        if attempt < max_retries:
            next_tool = _get_next_tool(tool_name, tried_tools)
            if next_tool:
                # Rephrase query before escalating for better coverage
                query = generate_retry_query(query, results, openai_client=openai_client)
                logger.info("Escalating from '%s' to '%s' (rephrased query)", tool_name, next_tool)
                tool_name = next_tool
            else:
                logger.info("No more tools to escalate to")
                break

    # Return best results found across all attempts
    if best_results:
        logger.info("Returning best results (score=%.2f from attempt %d)", best_score, best_attempt + 1)
        return best_results, max_retries
    return results, max_retries


def _get_next_tool(current: str, tried: set[str]) -> str | None:
    """Get next tool in escalation chain that hasn't been tried."""
    try:
        idx = _ESCALATION_CHAIN.index(current)
    except ValueError:
        idx = -1

    for tool in _ESCALATION_CHAIN[idx + 1:]:
        if tool not in tried:
            return tool
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    query: str,
    driver: Driver,
    openai_client: OpenAI | None = None,
    use_llm_router: bool = False,
    reasoning: ReasoningEngine | None = None,
) -> QAResult:
    """Run the agentic retrieval pipeline.

    1. Classify query → select tool
    2. Execute with self-correction loop
    3. Generate answer from results

    Returns QAResult with answer, sources, confidence, and metadata.
    """
    cfg = get_settings()
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=cfg.openai.api_key)

    # Step 1: Classify query
    decision = classify_query(query, use_llm=use_llm_router, openai_client=openai_client, reasoning=reasoning)
    logger.info(
        "Query classified: type=%s, tool=%s, confidence=%.2f",
        decision.query_type.value, decision.suggested_tool, decision.confidence,
    )

    # Step 2: Self-correction loop
    results, retries = self_correction_loop(
        query, driver, openai_client, decision,
    )

    # Step 3: Generate answer
    qa_result = generate_answer(query, results, openai_client=openai_client)

    # Step 4: Completeness check for GLOBAL queries (max 1 retry)
    if (
        decision.query_type == QueryType.GLOBAL
        and not evaluate_completeness(query, qa_result.answer, openai_client=openai_client)
    ):
        logger.info("Completeness check failed for GLOBAL query — retrying with comprehensive_search")
        extra_results = comprehensive_search(query, driver, openai_client)
        if extra_results:
            # Merge original + extra, deduplicate
            combined = results + [r for r in extra_results if r.chunk.id not in {sr.chunk.id for sr in results}]
            qa_result = generate_answer(query, combined, openai_client=openai_client)
            retries += 1

    # Enrich with metadata
    qa_result.retries = retries
    qa_result.router_decision = decision

    logger.info(
        "Agent result: %d sources, %d retries, confidence=%.2f",
        len(qa_result.sources), retries, qa_result.confidence,
    )
    return qa_result
