"""Tests for aggregation pipeline (group_by + reducers)."""
from __future__ import annotations

from pymangle.engine import eval_program
from pymangle.parser import parse


class TestAggregation:
    def test_count(self):
        """fn:count() per group."""
        store = eval_program(parse("""
            item("g1", 10). item("g1", 20). item("g1", 30).
            item("g2", 5). item("g2", 15).
            total(G, N) :- item(G, _) |> do fn:group_by(G), let N = fn:count().
        """))
        results = {(a.args[0].value, a.args[1].value) for a in store.get_by_predicate("total")}
        assert results == {("g1", 3), ("g2", 2)}

    def test_sum(self):
        """fn:sum(V) per group."""
        store = eval_program(parse("""
            item("g1", 10). item("g1", 20).
            item("g2", 5).
            total(G, S) :- item(G, V) |> do fn:group_by(G), let S = fn:sum(V).
        """))
        results = {(a.args[0].value, a.args[1].value) for a in store.get_by_predicate("total")}
        assert results == {("g1", 30), ("g2", 5)}

    def test_collect(self):
        """fn:collect(V) gathers values into a list."""
        store = eval_program(parse("""
            tag("doc1", "ai"). tag("doc1", "ml").
            tag("doc2", "db").
            all_tags(D, L) :- tag(D, T) |> do fn:group_by(D), let L = fn:collect(T).
        """))
        results = store.get_by_predicate("all_tags")
        by_doc = {}
        for a in results:
            by_doc[a.args[0].value] = a.args[1]
        # doc1 should have a ListTerm with 2 elements
        assert len(by_doc["doc1"].elements) == 2
        assert len(by_doc["doc2"].elements) == 1

    def test_no_group_by(self):
        """Global aggregation (no group_by keys)."""
        store = eval_program(parse("""
            score(10). score(20). score(30).
            total(S) :- score(V) |> let S = fn:sum(V).
        """))
        results = store.get_by_predicate("total")
        assert len(results) == 1
        assert results[0].args[0].value == 60

    def test_multi_group_keys(self):
        """fn:group_by(A, B) with two keys."""
        store = eval_program(parse("""
            sale("us", "q1", 100). sale("us", "q1", 200).
            sale("us", "q2", 50).
            sale("eu", "q1", 300).
            total(C, Q, S) :- sale(C, Q, V) |> do fn:group_by(C, Q), let S = fn:sum(V).
        """))
        results = {(a.args[0].value, a.args[1].value, a.args[2].value) for a in store.get_by_predicate("total")}
        assert results == {("us", "q1", 300), ("us", "q2", 50), ("eu", "q1", 300)}
