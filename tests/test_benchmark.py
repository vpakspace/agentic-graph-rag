"""Tests for benchmark.runner and benchmark.compare."""

from benchmark.compare import accuracy_by_type, compare_modes, compute_metrics
from benchmark.runner import MODES, load_questions

# ---------------------------------------------------------------------------
# load_questions
# ---------------------------------------------------------------------------

class TestLoadQuestions:
    def test_loads_all(self):
        qs = load_questions()
        assert len(qs) == 15

    def test_has_required_fields(self):
        qs = load_questions()
        for q in qs:
            assert "id" in q
            assert "question" in q
            assert "type" in q
            assert "keywords" in q

    def test_all_types_covered(self):
        qs = load_questions()
        types = {q["type"] for q in qs}
        assert types == {"simple", "relation", "multi_hop", "global", "temporal"}


# ---------------------------------------------------------------------------
# MODES
# ---------------------------------------------------------------------------

class TestModes:
    def test_five_modes(self):
        assert len(MODES) == 5
        assert set(MODES.keys()) == {
            "vector", "cypher", "hybrid", "agent_pattern", "agent_llm",
        }


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_empty(self):
        m = compute_metrics([])
        assert m["accuracy"] == 0.0
        assert m["total"] == 0

    def test_all_pass(self):
        results = [
            {"passed": True, "confidence": 0.8, "latency": 1.0, "retries": 0},
            {"passed": True, "confidence": 0.9, "latency": 2.0, "retries": 1},
        ]
        m = compute_metrics(results)
        assert m["accuracy"] == 1.0
        assert m["correct"] == 2
        assert m["avg_confidence"] == 0.85
        assert m["avg_latency"] == 1.5
        assert m["avg_retries"] == 0.5

    def test_partial(self):
        results = [
            {"passed": True, "confidence": 0.8, "latency": 1.0, "retries": 0},
            {"passed": False, "confidence": 0.3, "latency": 2.0, "retries": 2},
        ]
        m = compute_metrics(results)
        assert m["accuracy"] == 0.5
        assert m["correct"] == 1
        assert m["total"] == 2


# ---------------------------------------------------------------------------
# compare_modes
# ---------------------------------------------------------------------------

class TestCompareModes:
    def test_generates_rows(self):
        all_results = {
            "vector": [
                {"passed": True, "confidence": 0.8, "latency": 1.0, "retries": 0},
            ],
            "hybrid": [
                {"passed": False, "confidence": 0.3, "latency": 2.0, "retries": 1},
            ],
        }
        rows = compare_modes(all_results)
        assert len(rows) == 2
        assert rows[0]["Mode"] == "vector"
        assert rows[1]["Mode"] == "hybrid"
        assert "Accuracy" in rows[0]
        assert "Avg Confidence" in rows[0]

    def test_empty_results(self):
        rows = compare_modes({})
        assert rows == []


# ---------------------------------------------------------------------------
# accuracy_by_type
# ---------------------------------------------------------------------------

class TestAccuracyByType:
    def test_breakdown(self):
        all_results = {
            "vector": [
                {"type": "simple", "passed": True},
                {"type": "simple", "passed": False},
                {"type": "relation", "passed": True},
            ],
        }
        breakdown = accuracy_by_type(all_results)
        assert breakdown["vector"]["simple"] == 0.5
        assert breakdown["vector"]["relation"] == 1.0

    def test_empty(self):
        breakdown = accuracy_by_type({})
        assert breakdown == {}
