"""Tests for agentic_graph_rag.generation.graph_verifier."""

from unittest.mock import MagicMock

from rag_core.models import GraphContext

from agentic_graph_rag.generation.graph_verifier import (
    check_contradictions,
    verify_via_traversal,
)


def _make_context(
    triplets: list[dict] | None = None,
    passages: list[str] | None = None,
) -> GraphContext:
    return GraphContext(
        triplets=triplets or [],
        passages=passages or [],
    )


# ---------------------------------------------------------------------------
# verify_via_traversal
# ---------------------------------------------------------------------------

class TestVerifyViaTraversal:
    def test_empty_claim(self):
        ctx = _make_context(triplets=[{"source": "A", "relation": "R", "target": "B"}])
        result = verify_via_traversal("", ctx)
        assert result["verified"] is False

    def test_empty_triplets(self):
        ctx = _make_context()
        result = verify_via_traversal("Python is great", ctx)
        assert result["verified"] is False
        assert result["confidence"] == 0.0

    def test_matching_source_and_target(self):
        ctx = _make_context(triplets=[
            {"source": "Python", "relation": "USED_FOR", "target": "Machine Learning"},
        ])
        result = verify_via_traversal("Python is used for Machine Learning", ctx)
        assert result["verified"] is True
        assert len(result["supporting_triplets"]) == 1
        assert result["confidence"] > 0

    def test_partial_match_source_and_relation(self):
        ctx = _make_context(triplets=[
            {"source": "Python", "relation": "CREATED_BY", "target": "Guido"},
        ])
        result = verify_via_traversal("Python was created by someone", ctx)
        assert result["verified"] is True

    def test_no_match(self):
        ctx = _make_context(triplets=[
            {"source": "Java", "relation": "USED_FOR", "target": "Enterprise"},
        ])
        result = verify_via_traversal("Python is popular", ctx)
        assert result["verified"] is False

    def test_multiple_supporting_triplets(self):
        ctx = _make_context(triplets=[
            {"source": "Python", "relation": "USED_FOR", "target": "ML"},
            {"source": "Python", "relation": "CREATED_BY", "target": "Guido"},
            {"source": "Python", "relation": "SUPPORTS", "target": "ML"},
        ])
        result = verify_via_traversal("Python supports ML", ctx)
        assert result["verified"] is True
        assert len(result["supporting_triplets"]) >= 2

    def test_confidence_capped_at_1(self):
        ctx = _make_context(triplets=[
            {"source": "A", "relation": "R", "target": "B"},
            {"source": "A", "relation": "S", "target": "B"},
            {"source": "A", "relation": "T", "target": "B"},
            {"source": "A", "relation": "U", "target": "B"},
        ])
        result = verify_via_traversal("A is related to B", ctx)
        assert result["confidence"] <= 1.0

    def test_case_insensitive(self):
        ctx = _make_context(triplets=[
            {"source": "PYTHON", "relation": "RUNS_ON", "target": "LINUX"},
        ])
        result = verify_via_traversal("python runs on linux", ctx)
        assert result["verified"] is True

    def test_underscore_in_relation(self):
        ctx = _make_context(triplets=[
            {"source": "ML", "relation": "PART_OF", "target": "AI"},
        ])
        result = verify_via_traversal("ML is part of AI research", ctx)
        assert result["verified"] is True


# ---------------------------------------------------------------------------
# check_contradictions
# ---------------------------------------------------------------------------

class TestCheckContradictions:
    def test_empty_facts(self):
        ctx = _make_context(triplets=[{"source": "A", "relation": "R", "target": "B"}])
        result = check_contradictions([], ctx)
        assert result == []

    def test_empty_context(self):
        ctx = _make_context()
        result = check_contradictions(["fact1"], ctx)
        assert result == []

    def test_no_contradictions(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "NONE"
        client.chat.completions.create.return_value = resp

        ctx = _make_context(
            triplets=[{"source": "A", "relation": "R", "target": "B"}],
            passages=["A is related to B"],
        )
        result = check_contradictions(["A relates to B"], ctx, openai_client=client)
        assert result == []

    def test_finds_contradictions(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = (
            "CONTRADICTION: Python was created in 2000 | Created in 1991 | high\n"
            "CONTRADICTION: ML is simple | Requires years of study | medium"
        )
        client.chat.completions.create.return_value = resp

        ctx = _make_context(
            triplets=[{"source": "Python", "relation": "CREATED", "target": "1991"}],
        )
        result = check_contradictions(
            ["Python was created in 2000", "ML is simple"],
            ctx,
            openai_client=client,
        )
        assert len(result) == 2
        assert result[0]["severity"] == "high"
        assert result[1]["severity"] == "medium"

    def test_partial_contradiction_format(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "CONTRADICTION: claim | evidence"
        client.chat.completions.create.return_value = resp

        ctx = _make_context(passages=["Some passage"])
        result = check_contradictions(["claim"], ctx, openai_client=client)
        assert len(result) == 1
        assert result[0]["severity"] == "medium"  # default

    def test_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")

        ctx = _make_context(passages=["Some passage"])
        result = check_contradictions(["fact"], ctx, openai_client=client)
        assert result == []

    def test_with_passages_only(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "NONE"
        client.chat.completions.create.return_value = resp

        ctx = _make_context(passages=["Python was created in 1991 by Guido"])
        result = check_contradictions(["Python is old"], ctx, openai_client=client)
        assert result == []
        # Verify prompt includes passages
        call_args = client.chat.completions.create.call_args
        msg_content = call_args[1]["messages"][0]["content"]
        assert "Python was created in 1991" in msg_content
