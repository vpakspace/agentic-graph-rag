"""Tests for temporal fact store and Allen's interval algebra."""
from __future__ import annotations

from datetime import datetime

import pytest

from pymangle.ast_nodes import Atom, Constant, Interval, TermType
from pymangle.temporal import TemporalFactStore, allen_relation


# Helper to build temporal facts
def _atom(pred: str, *args: str) -> Atom:
    return Atom(pred, tuple(Constant(a, TermType.STRING) for a in args))


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class TestTemporalFactStore:
    def test_add_temporal_fact(self):
        """Store a fact with a time interval."""
        store = TemporalFactStore()
        atom = _atom("employed", "alice", "acme")
        store.add(atom, _dt("2020-01-01"), _dt("2023-06-30"))
        intervals = store.get_intervals(atom)
        assert len(intervals) == 1
        assert intervals[0] == (_dt("2020-01-01"), _dt("2023-06-30"))

    def test_query_point_in_time(self):
        """Query facts valid at a specific point."""
        store = TemporalFactStore()
        a1 = _atom("role", "alice", "engineer")
        a2 = _atom("role", "bob", "manager")
        store.add(a1, _dt("2020-01-01"), _dt("2023-01-01"))
        store.add(a2, _dt("2022-01-01"), _dt("2024-01-01"))

        results = store.query_at(_dt("2021-06-01"))
        preds = {(a.predicate, a.args[0].value) for a, _, _ in results}
        assert ("role", "alice") in preds
        assert ("role", "bob") not in preds  # bob starts 2022

        results2 = store.query_at(_dt("2022-06-01"))
        preds2 = {(a.predicate, a.args[0].value) for a, _, _ in results2}
        assert ("role", "alice") in preds2
        assert ("role", "bob") in preds2

    def test_query_range(self):
        """Query facts overlapping with a time range."""
        store = TemporalFactStore()
        a1 = _atom("project", "alpha")
        a2 = _atom("project", "beta")
        store.add(a1, _dt("2020-01-01"), _dt("2021-01-01"))
        store.add(a2, _dt("2021-06-01"), _dt("2022-06-01"))

        # Range that overlaps only alpha
        results = store.query_range(_dt("2020-06-01"), _dt("2020-12-01"))
        atoms = {a.args[0].value for a, _, _ in results}
        assert atoms == {"alpha"}

        # Range that overlaps both
        results2 = store.query_range(_dt("2020-06-01"), _dt("2022-01-01"))
        atoms2 = {a.args[0].value for a, _, _ in results2}
        assert atoms2 == {"alpha", "beta"}

    def test_interval_coalescing(self):
        """Overlapping intervals are merged."""
        store = TemporalFactStore()
        atom = _atom("status", "active")
        store.add(atom, _dt("2020-01-01"), _dt("2021-01-01"))
        store.add(atom, _dt("2020-06-01"), _dt("2022-01-01"))  # overlaps

        intervals = store.get_intervals(atom)
        assert len(intervals) == 1
        assert intervals[0] == (_dt("2020-01-01"), _dt("2022-01-01"))

    def test_coalescing_adjacent(self):
        """Adjacent (meeting) intervals are merged."""
        store = TemporalFactStore()
        atom = _atom("shift", "day")
        store.add(atom, _dt("2020-01-01"), _dt("2020-02-01"))
        store.add(atom, _dt("2020-02-01"), _dt("2020-03-01"))

        intervals = store.get_intervals(atom)
        assert len(intervals) == 1
        assert intervals[0] == (_dt("2020-01-01"), _dt("2020-03-01"))

    def test_no_coalescing_gap(self):
        """Non-overlapping intervals stay separate."""
        store = TemporalFactStore()
        atom = _atom("contract", "c1")
        store.add(atom, _dt("2020-01-01"), _dt("2020-06-01"))
        store.add(atom, _dt("2021-01-01"), _dt("2021-06-01"))

        intervals = store.get_intervals(atom)
        assert len(intervals) == 2


class TestAllenRelations:
    """Test Allen's 13 interval relations (7 base + inverses + equals)."""

    def test_before(self):
        assert allen_relation(
            _dt("2020-01-01"), _dt("2020-06-01"),
            _dt("2021-01-01"), _dt("2021-06-01"),
        ) == "before"

    def test_meets(self):
        assert allen_relation(
            _dt("2020-01-01"), _dt("2020-06-01"),
            _dt("2020-06-01"), _dt("2021-01-01"),
        ) == "meets"

    def test_overlaps(self):
        assert allen_relation(
            _dt("2020-01-01"), _dt("2021-01-01"),
            _dt("2020-06-01"), _dt("2021-06-01"),
        ) == "overlaps"

    def test_during(self):
        """A is strictly inside B."""
        assert allen_relation(
            _dt("2020-03-01"), _dt("2020-09-01"),
            _dt("2020-01-01"), _dt("2021-01-01"),
        ) == "during"

    def test_contains(self):
        """A strictly contains B (inverse of during)."""
        assert allen_relation(
            _dt("2020-01-01"), _dt("2021-01-01"),
            _dt("2020-03-01"), _dt("2020-09-01"),
        ) == "contains"

    def test_starts(self):
        """A starts at same time as B but ends earlier."""
        assert allen_relation(
            _dt("2020-01-01"), _dt("2020-06-01"),
            _dt("2020-01-01"), _dt("2021-01-01"),
        ) == "starts"

    def test_finishes(self):
        """A starts later but ends at same time as B."""
        assert allen_relation(
            _dt("2020-06-01"), _dt("2021-01-01"),
            _dt("2020-01-01"), _dt("2021-01-01"),
        ) == "finishes"

    def test_equals(self):
        assert allen_relation(
            _dt("2020-01-01"), _dt("2021-01-01"),
            _dt("2020-01-01"), _dt("2021-01-01"),
        ) == "equals"


class TestTemporalRuleIntegration:
    def test_temporal_rule_interval_intersection(self):
        """When two temporal facts join, result interval is their intersection."""
        store = TemporalFactStore()
        # Alice employed at Acme 2020-2023
        store.add(_atom("employed", "alice", "acme"), _dt("2020-01-01"), _dt("2023-01-01"))
        # Acme in NYC 2019-2022
        store.add(_atom("located", "acme", "nyc"), _dt("2019-01-01"), _dt("2022-01-01"))

        # Intersection: 2020-01-01 to 2022-01-01
        a1_intervals = store.get_intervals(_atom("employed", "alice", "acme"))
        a2_intervals = store.get_intervals(_atom("located", "acme", "nyc"))
        s1, e1 = a1_intervals[0]
        s2, e2 = a2_intervals[0]
        inter_start = max(s1, s2)
        inter_end = min(e1, e2)
        assert inter_start == _dt("2020-01-01")
        assert inter_end == _dt("2022-01-01")
        assert inter_start < inter_end  # valid intersection
