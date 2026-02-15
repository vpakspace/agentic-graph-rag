"""Tests for Mangle parser (Lark → AST)."""
from __future__ import annotations

import pytest

from pymangle.ast_nodes import Atom, Clause, Constant, NegAtom, Program, TermType, Variable
from pymangle.parser import ParseError, parse


class TestParseFacts:
    def test_string_args(self):
        prog = parse('edge("a", "b").')
        assert len(prog.facts) == 1
        fact = prog.facts[0]
        assert fact.head.predicate == "edge"
        assert fact.head.args == (
            Constant("a", TermType.STRING),
            Constant("b", TermType.STRING),
        )

    def test_name_constant(self):
        prog = parse('tool_for(/simple, "vector_search").')
        assert prog.facts[0].head.args[0] == Constant("simple", TermType.NAME)

    def test_number(self):
        prog = parse("score(42).")
        assert prog.facts[0].head.args[0] == Constant(42, TermType.NUMBER)

    def test_float(self):
        prog = parse("weight(3.14).")
        assert prog.facts[0].head.args[0] == Constant(3.14, TermType.FLOAT)


class TestParseRules:
    def test_simple_rule(self):
        prog = parse("reachable(X, Y) :- edge(X, Y).")
        assert len(prog.clauses) == 1
        clause = prog.clauses[0]
        assert clause.head.predicate == "reachable"
        assert len(clause.premises) == 1
        assert isinstance(clause.premises[0], Atom)

    def test_multi_premise(self):
        prog = parse("reachable(X, Z) :- edge(X, Y), edge(Y, Z).")
        assert len(prog.clauses[0].premises) == 2

    def test_negation(self):
        prog = parse("unemployed(X) :- person(X), !employee(X).")
        clause = prog.clauses[0]
        assert isinstance(clause.premises[1], NegAtom)
        assert clause.premises[1].atom.predicate == "employee"

    def test_variables_shared(self):
        prog = parse("path(X, Z) :- edge(X, Y), edge(Y, Z).")
        clause = prog.clauses[0]
        # Y appears in both premises
        assert clause.premises[0].args[1] == Variable("Y")
        assert clause.premises[1].args[0] == Variable("Y")


class TestParseTransform:
    def test_count_transform(self):
        prog = parse("total(G, N) :- item(G, _) |> do fn:group_by(G), let N = fn:count().")
        clause = prog.clauses[0]
        assert clause.transform is not None
        assert clause.transform.reducer.name == "fn:count"

    def test_sum_transform(self):
        prog = parse("total(G, S) :- item(G, V) |> do fn:group_by(G), let S = fn:sum(V).")
        clause = prog.clauses[0]
        assert clause.transform.reducer.name == "fn:sum"


class TestParseDecl:
    def test_external(self):
        prog = parse("Decl phrase_node(X, Y, Z) external.")
        assert len(prog.decls) == 1
        assert prog.decls[0].predicate == "phrase_node"
        assert prog.decls[0].arity == 3

    def test_temporal_with_bounds(self):
        prog = parse("Decl link(X, Y) temporal bound [/name, /name].")
        decl = prog.decls[0]
        assert "temporal" in [d.value for d in decl.descriptors]
        assert len(decl.bounds) == 1


class TestParseErrors:
    def test_missing_dot(self):
        with pytest.raises(ParseError):
            parse("edge(a, b)")

    def test_unclosed_paren(self):
        with pytest.raises(ParseError):
            parse("edge(a, b.")


class TestParseProgram:
    def test_full_program(self):
        prog = parse("""
            % Routing rules
            keyword(/relation, "связь").
            keyword(/relation, "between").

            match(Q, Cat) :- query_contains(Q, W), keyword(Cat, W).

            route_to(Q, "vector_search", 0.3) :- !match(Q, _).
        """)
        assert len(prog.facts) == 2
        assert len(prog.clauses) == 2
