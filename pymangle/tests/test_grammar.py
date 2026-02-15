"""Tests for Lark grammar parsing."""
from __future__ import annotations

from pathlib import Path

from lark import Lark

GRAMMAR_PATH = Path(__file__).parent.parent / "pymangle" / "grammar.lark"


def _parser() -> Lark:
    return Lark(GRAMMAR_PATH.read_text(), start="start", parser="earley")


class TestGrammarFacts:
    def test_simple_fact(self):
        tree = _parser().parse('edge("a", "b").')
        assert tree.data == "start"

    def test_name_constant_fact(self):
        tree = _parser().parse('tool_for(/simple, "vector_search").')
        assert tree.data == "start"

    def test_numeric_fact(self):
        tree = _parser().parse("score(42, 3.14).")
        assert tree.data == "start"

    def test_temporal_fact(self):
        tree = _parser().parse('link("a", "b")@[2024-01-01, 2024-12-31].')
        assert tree.data == "start"


class TestGrammarRules:
    def test_simple_rule(self):
        tree = _parser().parse("reachable(X, Z) :- edge(X, Y), edge(Y, Z).")
        assert tree.data == "start"

    def test_negation(self):
        tree = _parser().parse("unemployed(X) :- person(X), !employee(X).")
        assert tree.data == "start"

    def test_comparison(self):
        tree = _parser().parse("adult(X) :- person(X, Age), Age >= 18.")
        assert tree.data == "start"

    def test_funcall_in_body(self):
        tree = _parser().parse("total(X, S) :- val(X, V), S = fn:plus(V, 1).")
        assert tree.data == "start"

    def test_transform(self):
        tree = _parser().parse(
            "count(G, N) :- item(G, _) |> do fn:group_by(G), let N = fn:count()."
        )
        assert tree.data == "start"

    def test_builtin_predicate(self):
        tree = _parser().parse('ok(X) :- val(X), :string:starts_with(X, "hello").')
        assert tree.data == "start"


class TestGrammarDecls:
    def test_external_decl(self):
        tree = _parser().parse("Decl phrase_node(X, Y, Z) external.")
        assert tree.data == "start"

    def test_temporal_decl(self):
        tree = _parser().parse("Decl link(X, Y) temporal bound [/name, /name].")
        assert tree.data == "start"


class TestGrammarProgram:
    def test_multi_statement(self):
        tree = _parser().parse("""
            % Comment line
            edge("a", "b").
            edge("b", "c").
            reachable(X, Y) :- edge(X, Y).
            reachable(X, Z) :- reachable(X, Y), edge(Y, Z).
        """)
        stmts = [c for c in tree.children if c is not None]
        assert len(stmts) == 4
