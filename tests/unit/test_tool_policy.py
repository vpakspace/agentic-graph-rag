"""Tests for tool policy budgeting."""
from agentic_graph_rag.tool_policy import ToolPolicy, get_tool_policy


def test_default_policy():
    policy = ToolPolicy()
    assert policy.max_escalation_depth == 3
    assert policy.max_tool_timeout_ms == 30_000
    assert policy.max_tools_per_query == 5


def test_escalation_allowed():
    policy = ToolPolicy(max_escalation_depth=2)
    assert policy.check_escalation(0) is True
    assert policy.check_escalation(1) is True
    assert policy.check_escalation(2) is False
    assert policy.check_escalation(3) is False


def test_budget_allowed():
    policy = ToolPolicy(max_tools_per_query=3)
    assert policy.check_budget(0) is True
    assert policy.check_budget(2) is True
    assert policy.check_budget(3) is False


def test_tool_timeout():
    policy = ToolPolicy(
        max_tool_timeout_ms=20_000,
        tool_timeouts={"vector_search": 5_000},
    )
    assert policy.get_timeout("vector_search") == 5_000
    assert policy.get_timeout("unknown_tool") == 20_000


def test_get_tool_policy_defaults():
    policy = get_tool_policy()
    assert policy.max_escalation_depth == 3
    assert "vector_search" in policy.tool_timeouts


def test_get_tool_policy_env_override(monkeypatch):
    monkeypatch.setenv("TOOL_MAX_ESCALATION_DEPTH", "5")
    monkeypatch.setenv("TOOL_MAX_PER_QUERY", "10")
    policy = get_tool_policy()
    assert policy.max_escalation_depth == 5
    assert policy.max_tools_per_query == 10
