"""Tests for external predicate protocol."""
from __future__ import annotations

from collections.abc import Iterator

from pymangle.ast_nodes import Atom, Constant, TermType
from pymangle.engine import eval_program
from pymangle.external import ExternalPredicateRegistry
from pymangle.parser import parse


def _const(val: str | int) -> Constant:
    if isinstance(val, int):
        return Constant(val, TermType.NUMBER)
    return Constant(val, TermType.STRING)


class MockDBPredicate:
    """Mock external predicate simulating a database lookup."""

    def __init__(self, rows: list[list[str | int]]) -> None:
        self._rows = rows

    def query(self, inputs: list[Constant], filters: list) -> Iterator[list[Constant]]:
        for row in self._rows:
            consts = [_const(v) for v in row]
            # Filter: if inputs are provided, match positionally
            match = True
            for i, inp in enumerate(inputs):
                if inp is not None and i < len(consts) and consts[i] != inp:
                    match = False
                    break
            if match:
                yield consts


class TestExternalPredicate:
    def test_basic_query(self):
        """Mock external returns facts via registry."""
        registry = ExternalPredicateRegistry()
        mock = MockDBPredicate([["alice", 30], ["bob", 25]])
        registry.register("db_person", mock)

        results = list(registry.query("db_person", [], []))
        assert len(results) == 2
        assert results[0] == [_const("alice"), _const(30)]

    def test_with_input(self):
        """Bound args passed as inputs filter results."""
        registry = ExternalPredicateRegistry()
        mock = MockDBPredicate([["alice", 30], ["bob", 25]])
        registry.register("db_person", mock)

        results = list(registry.query("db_person", [_const("alice")], []))
        assert len(results) == 1
        assert results[0][0] == _const("alice")

    def test_no_results(self):
        """Empty result handled gracefully."""
        registry = ExternalPredicateRegistry()
        mock = MockDBPredicate([])
        registry.register("db_empty", mock)

        results = list(registry.query("db_empty", [], []))
        assert results == []

    def test_filter_pushdown(self):
        """Filters are passed to callback."""
        captured_filters = []

        class FilterCapture:
            def query(self, inputs, filters):
                captured_filters.extend(filters)
                return iter([])

        registry = ExternalPredicateRegistry()
        registry.register("filtered", FilterCapture())
        list(registry.query("filtered", [], ["age > 20"]))
        assert captured_filters == ["age > 20"]

    def test_external_in_rule(self):
        """External predicate used in rule body via eval_program."""
        prog = parse("""
            local("x").
            combined(X, Y) :- local(X), ext_data(Y).
        """)
        registry = ExternalPredicateRegistry()
        mock = MockDBPredicate([["a"], ["b"]])
        registry.register("ext_data", mock)

        store = eval_program(prog, externals=registry)
        results = store.get_by_predicate("combined")
        pairs = {(a.args[0].value, a.args[1].value) for a in results}
        assert pairs == {("x", "a"), ("x", "b")}
