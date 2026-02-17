"""PipelineService — typed contract for Agentic Graph RAG pipeline.

All clients (FastAPI, MCP, Streamlit) use this service.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from rag_core.models import PipelineTrace, QAResult

from agentic_graph_rag.agent.retrieval_agent import run as agent_run

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

    from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

logger = logging.getLogger(__name__)

_TRACE_CACHE_MAX = 100


class PipelineService:
    """Typed contract for the Agentic Graph RAG pipeline."""

    def __init__(
        self,
        driver: Driver,
        openai_client: OpenAI,
        reasoning: ReasoningEngine | None = None,
    ):
        self._driver = driver
        self._client = openai_client
        self._reasoning = reasoning
        self._trace_cache: OrderedDict[str, PipelineTrace] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        mode: str = "agent_pattern",
        lang: str = "ru",
    ) -> QAResult:
        """Full pipeline: route -> retrieve -> generate -> trace."""
        use_llm = mode == "agent_llm"
        reasoning = self._reasoning if mode == "agent_mangle" else None

        qa = agent_run(
            text,
            self._driver,
            openai_client=self._client,
            use_llm_router=use_llm,
            reasoning=reasoning,
        )

        if qa.trace:
            self._cache_trace(qa.trace)

        return qa

    def get_trace(self, trace_id: str) -> PipelineTrace | None:
        """Retrieve trace from in-memory cache."""
        return self._trace_cache.get(trace_id)

    def health(self) -> dict:
        """Neo4j connectivity check."""
        try:
            with self._driver.session() as session:
                session.run("RETURN 1").single()
            return {"status": "ok", "neo4j": "connected"}
        except Exception as e:
            return {"status": "degraded", "neo4j": str(e)}

    def graph_stats(self) -> dict:
        """Node and edge counts."""
        try:
            with self._driver.session() as session:
                result = session.run(
                    "MATCH (n) RETURN count(n) AS nodes "
                    "UNION ALL "
                    "MATCH ()-[r]->() RETURN count(r) AS nodes"
                )
                counts = [r["nodes"] for r in result]
            return {"nodes": counts[0] if counts else 0, "edges": counts[1] if len(counts) > 1 else 0}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cache_trace(self, trace: PipelineTrace) -> None:
        """Add trace to bounded cache (LRU eviction)."""
        self._trace_cache[trace.trace_id] = trace
        while len(self._trace_cache) > _TRACE_CACHE_MAX:
            self._trace_cache.popitem(last=False)
