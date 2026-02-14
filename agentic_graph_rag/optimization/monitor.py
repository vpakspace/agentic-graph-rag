"""Query Monitor — tracks query types, latency, and tool usage.

Provides runtime analytics for the agentic retrieval pipeline
and suggestions for PageRank weight tuning.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)


class QueryMonitor:
    """Lightweight in-memory monitor for retrieval pipeline analytics.

    Thread-safety is *not* guaranteed — suitable for single-process
    Streamlit or notebook usage.
    """

    def __init__(self) -> None:
        self._query_types: Counter[str] = Counter()
        self._tools_used: Counter[str] = Counter()
        self._latencies: list[float] = []
        self._retries: list[int] = []
        self._tool_latencies: defaultdict[str, list[float]] = defaultdict(list)
        self._total_queries = 0

    # -- recording ----------------------------------------------------------

    def record_query(
        self,
        query_type: str,
        tool_name: str,
        latency: float,
        retries: int = 0,
    ) -> None:
        """Record a completed query."""
        self._query_types[query_type] += 1
        self._tools_used[tool_name] += 1
        self._latencies.append(latency)
        self._retries.append(retries)
        self._tool_latencies[tool_name].append(latency)
        self._total_queries += 1

    @contextmanager
    def track(self, query_type: str, tool_name: str) -> Generator[dict[str, Any], None, None]:
        """Context manager that auto-records latency.

        Usage::

            with monitor.track("simple", "vector_search") as ctx:
                results = do_search(...)
                ctx["retries"] = 1
        """
        ctx: dict[str, Any] = {"retries": 0}
        start = time.monotonic()
        try:
            yield ctx
        finally:
            elapsed = time.monotonic() - start
            self.record_query(
                query_type=query_type,
                tool_name=tool_name,
                latency=elapsed,
                retries=ctx.get("retries", 0),
            )

    # -- stats --------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return dashboard-friendly statistics dict."""
        avg_latency = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies
            else 0.0
        )
        avg_retries = (
            sum(self._retries) / len(self._retries)
            if self._retries
            else 0.0
        )

        tool_avg = {}
        for tool, lats in self._tool_latencies.items():
            tool_avg[tool] = sum(lats) / len(lats) if lats else 0.0

        return {
            "total_queries": self._total_queries,
            "query_types": dict(self._query_types),
            "tools_used": dict(self._tools_used),
            "avg_latency": round(avg_latency, 4),
            "avg_retries": round(avg_retries, 2),
            "tool_avg_latency": {k: round(v, 4) for k, v in tool_avg.items()},
        }

    # -- suggestions --------------------------------------------------------

    def suggest_pagerank_weights(self) -> dict[str, Any]:
        """Recommend PageRank weight adjustments based on query distribution.

        Idea: if relation / multi_hop queries dominate, increase
        damping factor; if global queries dominate, lower skeleton_beta.
        """
        total = self._total_queries
        if total == 0:
            return {"suggestion": "Not enough data", "adjustments": {}}

        type_pcts = {k: v / total for k, v in self._query_types.items()}

        adjustments: dict[str, str] = {}

        relation_pct = type_pcts.get("relation", 0) + type_pcts.get("multi_hop", 0)
        if relation_pct > 0.5:
            adjustments["pagerank_damping"] = "increase to 0.90 (more graph-aware)"
            adjustments["max_hops"] = "increase to 3 (deeper traversal)"

        if type_pcts.get("global", 0) > 0.3:
            adjustments["skeleton_beta"] = "decrease to 0.15 (more skeletal coverage)"

        if type_pcts.get("simple", 0) > 0.7:
            adjustments["top_k_vector"] = "increase to 15 (broader vector recall)"

        if type_pcts.get("temporal", 0) > 0.2:
            adjustments["max_hops"] = "increase to 3 (temporal chains are deeper)"

        # Check retry rate
        retry_rate = sum(1 for r in self._retries if r > 0) / total if total else 0
        if retry_rate > 0.3:
            adjustments["relevance_threshold"] = "decrease to 2.5 (too many retries)"

        return {
            "suggestion": "Adjustments based on query distribution",
            "query_distribution": {k: round(v, 2) for k, v in type_pcts.items()},
            "retry_rate": round(retry_rate, 2),
            "adjustments": adjustments,
        }

    # -- reset --------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded data."""
        self._query_types.clear()
        self._tools_used.clear()
        self._latencies.clear()
        self._retries.clear()
        self._tool_latencies.clear()
        self._total_queries = 0
