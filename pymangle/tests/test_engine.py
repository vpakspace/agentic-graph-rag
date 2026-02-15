"""Tests for semi-naive bottom-up evaluation engine."""
from __future__ import annotations

import pytest

from pymangle.engine import FactLimitError, eval_program
from pymangle.parser import parse


class TestBasicEvaluation:
    def test_fact_only(self):
        """Facts are returned as-is."""
        store = eval_program(parse('edge("a", "b"). edge("b", "c").'))
        assert len(list(store.get_by_predicate("edge"))) == 2

    def test_single_rule(self):
        """Simple rule derives new facts."""
        store = eval_program(parse("""
            edge("a", "b"). edge("b", "c").
            path(X, Y) :- edge(X, Y).
        """))
        paths = store.get_by_predicate("path")
        assert len(paths) == 2

    def test_transitive_closure(self):
        """Recursive rule computes transitive closure."""
        store = eval_program(parse("""
            edge("a", "b"). edge("b", "c"). edge("c", "d").
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        """))
        results = store.get_by_predicate("reachable")
        # a→b, a→c, a→d, b→c, b→d, c→d = 6
        assert len(results) == 6

    def test_join(self):
        """Two-premise join."""
        store = eval_program(parse("""
            parent("alice", "bob").
            parent("bob", "carol").
            grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        """))
        results = store.get_by_predicate("grandparent")
        assert len(results) == 1


class TestDeltaOptimization:
    def test_no_redundant_derivation(self):
        """Semi-naive avoids re-deriving known facts."""
        store = eval_program(parse("""
            edge("a", "b"). edge("b", "c"). edge("c", "a").
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        """))
        # Cycle: should still terminate with 9 reachable facts (3x3)
        results = store.get_by_predicate("reachable")
        assert len(results) == 9


class TestFactLimit:
    def test_limit_prevents_infinite(self):
        """Fact limit stops runaway evaluation."""
        with pytest.raises(FactLimitError):
            eval_program(
                parse("""
                    num(0).
                    num(X) :- num(Y), X = fn:plus(Y, 1).
                """),
                fact_limit=100,
            )


class TestMultipleRules:
    def test_multiple_rules_same_head(self):
        store = eval_program(parse("""
            cat("whiskers"). dog("rex").
            pet(X) :- cat(X).
            pet(X) :- dog(X).
        """))
        pets = store.get_by_predicate("pet")
        assert len(pets) == 2
