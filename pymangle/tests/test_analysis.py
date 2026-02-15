"""Tests for stratification and dependency analysis."""
from __future__ import annotations

import pytest

from pymangle.analysis import StratificationError, stratify
from pymangle.engine import eval_program
from pymangle.parser import parse


class TestStratification:
    def test_no_negation_single_stratum(self):
        """All rules in one stratum when no negation."""
        prog = parse("""
            edge("a", "b").
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        """)
        strata = stratify(prog)
        assert len(strata) == 1

    def test_negation_two_strata(self):
        """Negated predicate must be in lower stratum."""
        prog = parse("""
            person("alice"). employee("bob").
            unemployed(X) :- person(X), !employee(X).
        """)
        strata = stratify(prog)
        assert len(strata) == 2

    def test_recursive_negation_rejected(self):
        """Recursive negation is invalid — should raise error."""
        prog = parse("""
            p(X) :- q(X), !p(X).
        """)
        with pytest.raises(StratificationError):
            stratify(prog)

    def test_aggregation_creates_stratum(self):
        """Rules with transforms are treated like negation for stratification."""
        prog = parse("""
            item("g1", 10). item("g1", 20).
            total(G, N) :- item(G, _) |> do fn:group_by(G), let N = fn:count().
            report(G, N) :- total(G, N).
        """)
        strata = stratify(prog)
        # item is stratum 0, total (with agg) is stratum 1, report depends on total is stratum 1 or 2
        assert len(strata) >= 2

    def test_complex_dependency_graph(self):
        """4+ predicates with chained negation produce 3 strata."""
        prog = parse("""
            a("x"). b("y").
            c(X) :- a(X).
            d(X) :- b(X), !c(X).
            e(X) :- a(X), !d(X).
        """)
        strata = stratify(prog)
        # c in stratum 0, d in stratum 1 (negates c), e in stratum 2 (negates d)
        assert len(strata) >= 3

    def test_independent_predicates(self):
        """Unrelated predicates can be in the same stratum."""
        prog = parse("""
            cat("whiskers"). dog("rex").
            pet_cat(X) :- cat(X).
            pet_dog(X) :- dog(X).
        """)
        strata = stratify(prog)
        assert len(strata) == 1

    def test_self_recursive_ok(self):
        """Positive self-recursion is allowed in same stratum."""
        prog = parse("""
            edge("a", "b"). edge("b", "c").
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        """)
        strata = stratify(prog)
        # reachable self-recurses positively — single stratum
        assert len(strata) == 1


class TestStratifiedEvaluation:
    def test_stratified_eval_order(self):
        """Facts computed in correct order with stratified eval."""
        store = eval_program(parse("""
            person("alice"). person("bob"). person("carol").
            employee("bob").
            unemployed(X) :- person(X), !employee(X).
        """))
        results = store.get_by_predicate("unemployed")
        names = {a.args[0].value for a in results}
        assert names == {"alice", "carol"}

    def test_stratified_negation_chain(self):
        """Multi-level negation evaluated correctly."""
        store = eval_program(parse("""
            a("x"). a("y"). a("z").
            b("y").
            c(X) :- a(X), !b(X).
            d(X) :- a(X), !c(X).
        """))
        c_vals = {a.args[0].value for a in store.get_by_predicate("c")}
        assert c_vals == {"x", "z"}
        d_vals = {a.args[0].value for a in store.get_by_predicate("d")}
        assert d_vals == {"y"}
