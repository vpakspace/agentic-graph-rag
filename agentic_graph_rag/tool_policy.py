"""Policy-based tool budgeting.

Controls escalation depth, per-tool timeouts, and query cost budget.
Configured via environment variables or Settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ToolPolicy:
    """Execution policy for retrieval tools."""

    max_escalation_depth: int = 3
    max_tool_timeout_ms: int = 30_000
    max_tools_per_query: int = 5
    tool_timeouts: dict[str, int] = field(default_factory=dict)

    def check_escalation(self, current_depth: int) -> bool:
        """Return True if further escalation is allowed."""
        return current_depth < self.max_escalation_depth

    def get_timeout(self, tool_name: str) -> int:
        """Get timeout in ms for a specific tool."""
        return self.tool_timeouts.get(tool_name, self.max_tool_timeout_ms)

    def check_budget(self, tools_used: int) -> bool:
        """Return True if more tool calls are within budget."""
        return tools_used < self.max_tools_per_query


_DEFAULT_TOOL_TIMEOUTS = {
    "vector_search": 10_000,
    "cypher_traverse": 15_000,
    "hybrid_search": 20_000,
    "temporal_query": 15_000,
    "comprehensive_search": 30_000,
    "full_document_read": 30_000,
}


def get_tool_policy() -> ToolPolicy:
    """Create ToolPolicy from environment variables."""
    return ToolPolicy(
        max_escalation_depth=int(os.environ.get("TOOL_MAX_ESCALATION_DEPTH", "3")),
        max_tool_timeout_ms=int(os.environ.get("TOOL_MAX_TIMEOUT_MS", "30000")),
        max_tools_per_query=int(os.environ.get("TOOL_MAX_PER_QUERY", "5")),
        tool_timeouts=_DEFAULT_TOOL_TIMEOUTS,
    )
