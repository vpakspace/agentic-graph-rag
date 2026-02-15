"""Tests for Mangle benchmark mode."""
from __future__ import annotations

from pathlib import Path

from benchmark.runner import MODES, load_questions


class TestBenchmarkMangleMode:
    def test_benchmark_mangle_mode_runs(self):
        """agent_mangle mode is registered in MODES."""
        assert "agent_mangle" in MODES
        assert callable(MODES["agent_mangle"])

    def test_benchmark_mangle_vs_pattern(self):
        """agent_mangle and agent_pattern are distinct modes with same questions."""
        assert "agent_mangle" in MODES
        assert "agent_pattern" in MODES
        assert MODES["agent_mangle"] is not MODES["agent_pattern"]

        # Both use the same question set
        questions = load_questions()
        assert len(questions) == 15

    def test_six_modes_total(self):
        """Adding agent_mangle brings total to 6 modes."""
        assert len(MODES) == 6
        assert set(MODES.keys()) == {
            "vector", "cypher", "hybrid",
            "agent_pattern", "agent_llm", "agent_mangle",
        }
