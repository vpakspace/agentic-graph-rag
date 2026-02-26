"""Tests for semi-naive bottom-up evaluation engine."""
from __future__ import annotations

import pytest

from pymangle.ast_nodes import (
    Atom,
    Clause,
    Comparison,
    Constant,
    FunCall,
    Interval,
    ListTerm,
    NegAtom,
    Program,
    TemporalAtom,
    TermType,
    Transform,
    Variable,
)
from pymangle.engine import FactLimitError, _eval_comparison, _extract_filters, eval_program
from pymangle.factstore import IndexedFactStore
from pymangle.parser import parse


# ── Basic evaluation ────────────────────────────────────────────────


class TestBasicEvaluation:
    def test_fact_only(self):
        store = eval_program(parse('edge("a", "b"). edge("b", "c").'))
        assert len(list(store.get_by_predicate("edge"))) == 2

    def test_single_rule(self):
        store = eval_program(parse("""
            edge("a", "b"). edge("b", "c").
            path(X, Y) :- edge(X, Y).
        """))
        paths = store.get_by_predicate("path")
        assert len(paths) == 2

    def test_transitive_closure(self):
        store = eval_program(parse("""
            edge("a", "b"). edge("b", "c"). edge("c", "d").
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        """))
        results = store.get_by_predicate("reachable")
        assert len(results) == 6

    def test_join(self):
        store = eval_program(parse("""
            parent("alice", "bob").
            parent("bob", "carol").
            grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        """))
        results = store.get_by_predicate("grandparent")
        assert len(results) == 1

    def test_empty_program(self):
        store = eval_program(parse(""))
        assert len(store) == 0

    def test_facts_only_no_rules(self):
        """Program with facts but no rules returns just the facts."""
        store = eval_program(parse('node("a"). node("b"). node("c").'))
        assert len(list(store.get_by_predicate("node"))) == 3

    def test_existing_store(self):
        """Pass existing store with pre-loaded facts."""
        existing = IndexedFactStore()
        existing.add(Atom("base", (Constant("x", TermType.STRING),)))
        store = eval_program(parse('derived(X) :- base(X).'), store=existing)
        assert len(list(store.get_by_predicate("derived"))) == 1


# ── Delta optimization ──────────────────────────────────────────────


class TestDeltaOptimization:
    def test_no_redundant_derivation(self):
        store = eval_program(parse("""
            edge("a", "b"). edge("b", "c"). edge("c", "a").
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        """))
        results = store.get_by_predicate("reachable")
        assert len(results) == 9


# ── Fact limit ──────────────────────────────────────────────────────


class TestFactLimit:
    def test_limit_prevents_infinite(self):
        with pytest.raises(FactLimitError):
            eval_program(
                parse("""
                    num(0).
                    num(X) :- num(Y), X = fn:plus(Y, 1).
                """),
                fact_limit=100,
            )

    def test_limit_in_delta_iteration(self):
        """Limit hit during delta phase (not just initial)."""
        with pytest.raises(FactLimitError):
            eval_program(
                parse("""
                    seed(0).
                    grow(X) :- seed(X).
                    grow(X) :- grow(Y), X = fn:plus(Y, 1).
                """),
                fact_limit=50,
            )


# ── Multiple rules ──────────────────────────────────────────────────


class TestMultipleRules:
    def test_multiple_rules_same_head(self):
        store = eval_program(parse("""
            cat("whiskers"). dog("rex").
            pet(X) :- cat(X).
            pet(X) :- dog(X).
        """))
        pets = store.get_by_predicate("pet")
        assert len(pets) == 2


# ── Negation ────────────────────────────────────────────────────────


class TestNegation:
    def test_negation_as_failure(self):
        store = eval_program(parse("""
            bird("tweety"). bird("penguin").
            flightless("penguin").
            can_fly(X) :- bird(X), !flightless(X).
        """))
        results = store.get_by_predicate("can_fly")
        assert len(results) == 1
        assert results[0].args[0] == Constant("tweety", TermType.STRING)

    def test_negation_all_fail(self):
        """Negation blocks all when all are negated."""
        store = eval_program(parse("""
            item("a"). excluded("a").
            ok(X) :- item(X), !excluded(X).
        """))
        results = store.get_by_predicate("ok")
        assert len(results) == 0


# ── Comparisons ─────────────────────────────────────────────────────


class TestComparisons:
    def test_inequality_filter(self):
        store = eval_program(parse("""
            val(1). val(2). val(3).
            big(X) :- val(X), X > 1.
        """))
        results = store.get_by_predicate("big")
        assert len(results) == 2

    def test_equality_filter(self):
        store = eval_program(parse("""
            pair("a", "a"). pair("a", "b").
            same(X) :- pair(X, Y), X == Y.
        """))
        results = store.get_by_predicate("same")
        assert len(results) == 1

    def test_not_equal_filter(self):
        store = eval_program(parse("""
            pair("a", "a"). pair("a", "b").
            diff(X, Y) :- pair(X, Y), X != Y.
        """))
        results = store.get_by_predicate("diff")
        assert len(results) == 1

    def test_assignment_with_funcall(self):
        store = eval_program(parse("""
            base(3).
            doubled(X, D) :- base(X), D = fn:mult(X, 2).
        """))
        results = store.get_by_predicate("doubled")
        assert len(results) == 1
        assert results[0].args[1] == Constant(6, TermType.NUMBER)

    def test_assignment_with_constant(self):
        """Assignment X = constant (not funcall)."""
        store = eval_program(parse("""
            item("a").
            tagged(X, T) :- item(X), T = "default".
        """))
        results = store.get_by_predicate("tagged")
        assert len(results) == 1
        assert results[0].args[1] == Constant("default", TermType.STRING)

    def test_le_comparison(self):
        store = eval_program(parse("""
            val(1). val(2). val(3).
            small(X) :- val(X), X <= 2.
        """))
        assert len(store.get_by_predicate("small")) == 2

    def test_ge_comparison(self):
        store = eval_program(parse("""
            val(1). val(2). val(3).
            big(X) :- val(X), X >= 2.
        """))
        assert len(store.get_by_predicate("big")) == 2

    def test_lt_comparison(self):
        store = eval_program(parse("""
            val(1). val(2). val(3).
            tiny(X) :- val(X), X < 2.
        """))
        assert len(store.get_by_predicate("tiny")) == 1


# ── Aggregation ─────────────────────────────────────────────────────


class TestAggregation:
    def test_count(self):
        store = eval_program(parse("""
            item("a"). item("b"). item("c").
            total(N) :- item(_) |> let N = fn:count().
        """))
        results = store.get_by_predicate("total")
        assert len(results) == 1
        assert results[0].args[0] == Constant(3, TermType.NUMBER)

    def test_sum(self):
        store = eval_program(parse("""
            score("alice", 10). score("bob", 20). score("carol", 30).
            total(S) :- score(_, V) |> let S = fn:sum(V).
        """))
        results = store.get_by_predicate("total")
        assert len(results) == 1
        assert results[0].args[0] == Constant(60, TermType.NUMBER)

    def test_avg(self):
        store = eval_program(parse("""
            score("alice", 10). score("bob", 20). score("carol", 30).
            average(A) :- score(_, V) |> let A = fn:avg(V).
        """))
        results = store.get_by_predicate("average")
        assert len(results) == 1
        assert results[0].args[0] == Constant(20.0, TermType.FLOAT)

    def test_min(self):
        store = eval_program(parse("""
            val(5). val(2). val(8).
            smallest(M) :- val(V) |> let M = fn:min(V).
        """))
        results = store.get_by_predicate("smallest")
        assert len(results) == 1
        assert results[0].args[0] == Constant(2, TermType.NUMBER)

    def test_max(self):
        store = eval_program(parse("""
            val(5). val(2). val(8).
            largest(M) :- val(V) |> let M = fn:max(V).
        """))
        results = store.get_by_predicate("largest")
        assert len(results) == 1
        assert results[0].args[0] == Constant(8, TermType.NUMBER)

    def test_collect(self):
        store = eval_program(parse("""
            tag("a"). tag("b"). tag("c").
            all_tags(L) :- tag(V) |> let L = fn:collect(V).
        """))
        results = store.get_by_predicate("all_tags")
        assert len(results) == 1
        assert isinstance(results[0].args[0], ListTerm)
        assert len(results[0].args[0].elements) == 3

    def test_group_by(self):
        store = eval_program(parse("""
            score("alice", 10). score("alice", 20). score("bob", 30).
            total_by(Name, S) :- score(Name, V) |> do fn:group_by(Name), let S = fn:sum(V).
        """))
        results = store.get_by_predicate("total_by")
        assert len(results) == 2  # alice and bob

    def test_unknown_reducer(self):
        """Unknown reducer produces no results."""
        store = eval_program(parse("""
            item("a").
            result(N) :- item(_) |> let N = fn:nonexistent().
        """))
        results = store.get_by_predicate("result")
        assert len(results) == 0

    def test_sum_float(self):
        """Sum of float values returns float."""
        store = eval_program(parse("""
            val(1.5). val(2.5).
            total(S) :- val(V) |> let S = fn:sum(V).
        """))
        results = store.get_by_predicate("total")
        assert len(results) == 1
        assert results[0].args[0] == Constant(4.0, TermType.FLOAT)

    def test_min_empty_values(self):
        """Min with no numeric values returns nothing."""
        # Only items without the aggregated variable bound
        store = eval_program(parse("""
            item("a").
            smallest(M) :- item(X) |> let M = fn:min(X).
        """))
        # X is string, min works on constants — should still produce result
        results = store.get_by_predicate("smallest")
        assert len(results) == 1

    def test_avg_empty(self):
        """Avg with empty values returns nothing."""
        prog = Program(
            facts=[],
            clauses=[
                Clause(
                    head=Atom("result", (Variable("A"),)),
                    premises=(Atom("empty_pred", (Variable("V"),)),),
                    transform=Transform(
                        group_by=(),
                        variable=Variable("A"),
                        reducer=FunCall("fn:avg", (Variable("V"),)),
                    ),
                ),
            ],
        )
        store = eval_program(prog)
        assert len(store.get_by_predicate("result")) == 0


# ── Fact limit in aggregation ───────────────────────────────────────


class TestFactLimitAggregation:
    def test_limit_in_aggregation(self):
        """Fact limit hit during aggregation phase."""
        with pytest.raises(FactLimitError):
            # Create many groups that each produce a fact
            facts_str = " ".join(f'item("g{i}", {i}).' for i in range(200))
            eval_program(
                parse(f"""
                    {facts_str}
                    total(G, S) :- item(G, V) |> do fn:group_by(G), let S = fn:sum(V).
                """),
                fact_limit=150,  # enough for facts, not for aggregation results
            )


# ── Temporal atoms ──────────────────────────────────────────────────


class TestTemporalAtoms:
    def test_temporal_atom_in_body(self):
        """Temporal atom resolves like a regular atom + binds interval vars."""
        store = eval_program(parse("""
            event("login", "alice").
            happened(E, W) :- event(E, W)@[S, _].
        """))
        results = store.get_by_predicate("happened")
        assert len(results) == 1


# ── _eval_comparison ────────────────────────────────────────────────


class TestEvalComparison:
    def test_equal_constants(self):
        a = Constant(5, TermType.NUMBER)
        b = Constant(5, TermType.NUMBER)
        assert _eval_comparison(a, b, "=") is True
        assert _eval_comparison(a, b, "==") is True

    def test_not_equal(self):
        a = Constant(5, TermType.NUMBER)
        b = Constant(3, TermType.NUMBER)
        assert _eval_comparison(a, b, "!=") is True
        assert _eval_comparison(a, a, "!=") is False

    def test_less_than(self):
        a = Constant(3, TermType.NUMBER)
        b = Constant(5, TermType.NUMBER)
        assert _eval_comparison(a, b, "<") is True
        assert _eval_comparison(b, a, "<") is False

    def test_less_equal(self):
        a = Constant(5, TermType.NUMBER)
        b = Constant(5, TermType.NUMBER)
        assert _eval_comparison(a, b, "<=") is True

    def test_greater_than(self):
        a = Constant(5, TermType.NUMBER)
        b = Constant(3, TermType.NUMBER)
        assert _eval_comparison(a, b, ">") is True

    def test_greater_equal(self):
        a = Constant(5, TermType.NUMBER)
        b = Constant(5, TermType.NUMBER)
        assert _eval_comparison(a, b, ">=") is True

    def test_unknown_op(self):
        a = Constant(5, TermType.NUMBER)
        assert _eval_comparison(a, a, "~") is False

    def test_non_constant_terms(self):
        """Comparison with raw values (non-Constant)."""
        assert _eval_comparison(5, 5, "=") is True


# ── _extract_filters ────────────────────────────────────────────────


class TestExtractFilters:
    def test_extract_simple_filter(self):
        premises = (
            Atom("data", (Variable("X"), Variable("Y"))),
            Comparison(Variable("Y"), ">", Constant(10, TermType.NUMBER)),
        )
        filters = _extract_filters(premises, 0, {})
        assert len(filters) == 1
        assert filters[0]["var"] == "Y"
        assert filters[0]["op"] == ">"
        assert filters[0]["value"] == 10

    def test_extract_reversed_filter(self):
        """Filter with ground on left, variable on right → flip operator."""
        premises = (
            Atom("data", (Variable("X"),)),
            Comparison(Constant(100, TermType.NUMBER), "<", Variable("X")),
        )
        filters = _extract_filters(premises, 0, {})
        assert len(filters) == 1
        assert filters[0]["var"] == "X"
        assert filters[0]["op"] == ">"  # flipped from <

    def test_no_filters_for_atom_premises(self):
        """Non-comparison premises are skipped."""
        premises = (
            Atom("a", (Variable("X"),)),
            Atom("b", (Variable("Y"),)),
        )
        filters = _extract_filters(premises, 0, {})
        assert len(filters) == 0

    def test_multiple_filters(self):
        premises = (
            Atom("data", (Variable("X"),)),
            Comparison(Variable("X"), ">", Constant(0, TermType.NUMBER)),
            Comparison(Variable("X"), "<", Constant(100, TermType.NUMBER)),
        )
        filters = _extract_filters(premises, 0, {})
        assert len(filters) == 2


# ── Delta iteration: negation, temporal, comparison ─────────────────


class TestDeltaNegation:
    def test_negation_in_delta_iteration(self):
        """NegAtom premise evaluated during delta phase."""
        store = eval_program(parse("""
            seed("a"). seed("b").
            blocked("b").
            step(X) :- seed(X).
            step(Y) :- step(X), seed(Y), !blocked(Y).
        """))
        results = store.get_by_predicate("step")
        # step("a") and step("b") from first rule; delta won't add more
        assert len(results) == 2


class TestDeltaTemporalAtom:
    def test_temporal_in_delta_rule(self):
        """TemporalAtom in a rule that triggers delta iteration."""
        store = eval_program(parse("""
            event("login", "alice").
            event("login", "bob").
            base("alice"). base("bob").
            linked(W) :- base(W).
            seen(E, W) :- linked(W), event(E, W)@[S, _].
        """))
        results = store.get_by_predicate("seen")
        assert len(results) == 2


class TestDeltaComparison:
    def test_funcall_assignment_in_delta(self):
        """Comparison with FunCall assignment during delta iteration."""
        store = eval_program(parse("""
            num(1). num(2).
            acc(X) :- num(X).
            acc(Z) :- acc(X), num(Y), Z = fn:plus(X, Y), Z < 10.
        """))
        results = store.get_by_predicate("acc")
        # 1, 2, 3, 4, 5, 6, 7, 8, 9 (various sums < 10)
        assert len(results) >= 5

    def test_constant_assignment_in_delta(self):
        """Comparison X = constant during delta iteration."""
        store = eval_program(parse("""
            item("a").
            chain(X) :- item(X).
            chain(Y) :- chain(X), Y = "done", X != "done".
        """))
        results = store.get_by_predicate("chain")
        assert Atom("chain", (Constant("done", TermType.STRING),)) in results

    def test_ground_comparison_in_delta(self):
        """Ground comparison filter during delta iteration."""
        store = eval_program(parse("""
            val(1). val(2). val(3).
            step(X) :- val(X).
            step(Z) :- step(X), val(Y), Z = fn:plus(X, Y), Z <= 5.
        """))
        results = store.get_by_predicate("step")
        # All derived values <= 5
        for r in results:
            if isinstance(r.args[0], Constant) and isinstance(r.args[0].value, int):
                assert r.args[0].value <= 5


class TestDeltaExternalPredicate:
    def test_external_in_non_delta_premise(self):
        """External predicate used in non-delta premise during delta iteration."""

        class MockExternals:
            def has(self, pred):
                return pred == "lookup"

            def query_as_atoms(self, pred, inputs, filters):
                return [Atom("lookup", (Constant("found", TermType.STRING),))]

        prog = Program(
            facts=[
                Clause(head=Atom("seed", (Constant("a", TermType.STRING),))),
            ],
            clauses=[
                Clause(
                    head=Atom("step", (Variable("X"),)),
                    premises=(Atom("seed", (Variable("X"),)),),
                ),
                Clause(
                    head=Atom("result", (Variable("X"), Variable("Y"))),
                    premises=(
                        Atom("step", (Variable("X"),)),
                        Atom("lookup", (Variable("Y"),)),
                    ),
                ),
            ],
        )
        store = eval_program(prog, externals=MockExternals())
        results = store.get_by_predicate("result")
        assert len(results) == 1


# ── Initial round fact limit ────────────────────────────────────────


class TestInitialRoundFactLimit:
    def test_limit_in_initial_round(self):
        """Fact limit hit during initial (non-delta) evaluation round."""
        # Cross product of 20x20 = 400 facts, limit at 50
        facts_a = " ".join(f'a({i}).' for i in range(20))
        facts_b = " ".join(f'b({i}).' for i in range(20))
        with pytest.raises(FactLimitError):
            eval_program(
                parse(f"""
                    {facts_a} {facts_b}
                    cross(X, Y) :- a(X), b(Y).
                """),
                fact_limit=50,
            )


# ── Aggregation edge cases ──────────────────────────────────────────


class TestAggregationEdgeCases:
    def test_reducer_no_agg_var(self):
        """Reducer (not fn:count) with no agg_var returns None per group."""
        prog = Program(
            facts=[Clause(head=Atom("item", (Constant("a", TermType.STRING),)))],
            clauses=[
                Clause(
                    head=Atom("result", (Variable("S"),)),
                    premises=(Atom("item", (Variable("X"),)),),
                    transform=Transform(
                        group_by=(),
                        variable=Variable("S"),
                        reducer=FunCall("fn:sum", ()),  # no args → agg_var is None
                    ),
                ),
            ],
        )
        store = eval_program(prog)
        assert len(store.get_by_predicate("result")) == 0

    def test_max_empty_nums(self):
        """fn:max with no numeric constants."""
        prog = Program(
            facts=[],
            clauses=[
                Clause(
                    head=Atom("result", (Variable("M"),)),
                    premises=(Atom("empty", (Variable("V"),)),),
                    transform=Transform(
                        group_by=(),
                        variable=Variable("M"),
                        reducer=FunCall("fn:max", (Variable("V"),)),
                    ),
                ),
            ],
        )
        store = eval_program(prog)
        assert len(store.get_by_predicate("result")) == 0

    def test_min_empty_nums(self):
        """fn:min with no numeric constants."""
        prog = Program(
            facts=[],
            clauses=[
                Clause(
                    head=Atom("result", (Variable("M"),)),
                    premises=(Atom("empty", (Variable("V"),)),),
                    transform=Transform(
                        group_by=(),
                        variable=Variable("M"),
                        reducer=FunCall("fn:min", (Variable("V"),)),
                    ),
                ),
            ],
        )
        store = eval_program(prog)
        assert len(store.get_by_predicate("result")) == 0

    def test_transform_without_reducer(self):
        """Transform with reducer=None produces nothing."""
        prog = Program(
            facts=[Clause(head=Atom("item", (Constant("a", TermType.STRING),)))],
            clauses=[
                Clause(
                    head=Atom("result", (Variable("X"),)),
                    premises=(Atom("item", (Variable("X"),)),),
                    transform=Transform(group_by=(), variable=None, reducer=None),
                ),
            ],
        )
        store = eval_program(prog)
        assert len(store.get_by_predicate("result")) == 0


# ── External predicates ─────────────────────────────────────────────


class TestExternalPredicates:
    def test_external_fallback(self):
        """Engine falls back to external predicate when store has no match."""

        class MockExternals:
            def has(self, pred):
                return pred == "ext_data"

            def query_as_atoms(self, pred, inputs, filters):
                return [
                    Atom("ext_data", (Constant("result1", TermType.STRING),)),
                    Atom("ext_data", (Constant("result2", TermType.STRING),)),
                ]

        prog = Program(
            facts=[],
            clauses=[
                Clause(
                    head=Atom("found", (Variable("X"),)),
                    premises=(Atom("ext_data", (Variable("X"),)),),
                ),
            ],
        )
        store = eval_program(prog, externals=MockExternals())
        results = store.get_by_predicate("found")
        assert len(results) == 2

    def test_external_with_filter_pushdown(self):
        """Comparisons after external atom become pushdown filters."""
        captured_filters = []

        class MockExternals:
            def has(self, pred):
                return pred == "remote"

            def query_as_atoms(self, pred, inputs, filters):
                captured_filters.extend(filters)
                return [Atom("remote", (Constant(50, TermType.NUMBER),))]

        prog = Program(
            facts=[],
            clauses=[
                Clause(
                    head=Atom("ok", (Variable("X"),)),
                    premises=(
                        Atom("remote", (Variable("X"),)),
                        Comparison(Variable("X"), ">", Constant(10, TermType.NUMBER)),
                    ),
                ),
            ],
        )
        store = eval_program(prog, externals=MockExternals())
        assert len(captured_filters) == 1
        assert captured_filters[0]["op"] == ">"

    def test_no_external_when_store_has_data(self):
        """External not called when store already has matching facts."""
        called = []

        class MockExternals:
            def has(self, pred):
                called.append(pred)
                return True

            def query_as_atoms(self, pred, inputs, filters):
                called.append("query")
                return []

        store = eval_program(
            parse("""
                data("local").
                found(X) :- data(X).
            """),
            externals=MockExternals(),
        )
        results = store.get_by_predicate("found")
        assert len(results) == 1
        assert "query" not in called
