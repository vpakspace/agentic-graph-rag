"""Tests for indexed fact storage."""
from __future__ import annotations

from pymangle.ast_nodes import Atom, Constant, TermType, Variable
from pymangle.factstore import IndexedFactStore


def _c(val: str) -> Constant:
    return Constant(val, TermType.STRING)


def _atom(pred: str, *args: str) -> Atom:
    return Atom(pred, tuple(_c(a) for a in args))


class TestFactStore:
    def test_add_new_fact(self):
        store = IndexedFactStore()
        assert store.add(_atom("edge", "a", "b")) is True
        assert len(store) == 1

    def test_add_duplicate(self):
        store = IndexedFactStore()
        store.add(_atom("edge", "a", "b"))
        assert store.add(_atom("edge", "a", "b")) is False
        assert len(store) == 1

    def test_query_exact(self):
        store = IndexedFactStore()
        store.add(_atom("edge", "a", "b"))
        store.add(_atom("edge", "a", "c"))
        store.add(_atom("node", "x"))

        results = list(store.query(Atom("edge", (Variable("X"), Variable("Y")))))
        assert len(results) == 2

    def test_query_first_arg_index(self):
        store = IndexedFactStore()
        store.add(_atom("edge", "a", "b"))
        store.add(_atom("edge", "a", "c"))
        store.add(_atom("edge", "b", "d"))

        results = list(store.query(Atom("edge", (_c("a"), Variable("Y")))))
        assert len(results) == 2

    def test_query_no_match(self):
        store = IndexedFactStore()
        store.add(_atom("edge", "a", "b"))
        results = list(store.query(Atom("node", (Variable("X"),))))
        assert len(results) == 0

    def test_merge(self):
        store1 = IndexedFactStore()
        store1.add(_atom("edge", "a", "b"))

        store2 = IndexedFactStore()
        store2.add(_atom("edge", "b", "c"))
        store2.add(_atom("edge", "a", "b"))  # duplicate

        new_facts = store1.merge(store2)
        assert len(store1) == 2
        assert len(new_facts) == 1  # only "b","c" is new

    def test_get_by_predicate(self):
        store = IndexedFactStore()
        store.add(_atom("edge", "a", "b"))
        store.add(_atom("edge", "c", "d"))
        store.add(_atom("node", "x"))

        edges = list(store.get_by_predicate("edge"))
        assert len(edges) == 2

    def test_all_facts(self):
        store = IndexedFactStore()
        store.add(_atom("a", "1"))
        store.add(_atom("b", "2"))
        assert len(list(store.all_facts())) == 2
