"""Tests for term unification."""
from __future__ import annotations

from pymangle.ast_nodes import Atom, Constant, TermType, Variable
from pymangle.unifier import apply_subst, unify


class TestUnify:
    def test_var_to_const(self):
        subst = unify(Variable("X"), Constant("a", TermType.STRING), {})
        assert subst == {"X": Constant("a", TermType.STRING)}

    def test_const_to_const_same(self):
        a = Constant("a", TermType.STRING)
        subst = unify(a, a, {})
        assert subst == {}

    def test_const_to_const_diff(self):
        subst = unify(
            Constant("a", TermType.STRING),
            Constant("b", TermType.STRING),
            {},
        )
        assert subst is None

    def test_var_to_var(self):
        subst = unify(Variable("X"), Variable("Y"), {})
        assert subst is not None
        assert "X" in subst or "Y" in subst

    def test_bound_var(self):
        subst = unify(Variable("X"), Constant("b", TermType.STRING), {"X": Constant("a", TermType.STRING)})
        assert subst is None  # X already bound to "a", can't unify with "b"

    def test_wildcard(self):
        subst = unify(Variable("_"), Constant("anything", TermType.STRING), {})
        assert subst is not None  # _ unifies with anything, no binding added

    def test_atom_unify(self):
        pattern = Atom("edge", (Variable("X"), Variable("Y")))
        fact = Atom("edge", (Constant("a", TermType.STRING), Constant("b", TermType.STRING)))
        subst = unify(pattern, fact, {})
        assert subst == {
            "X": Constant("a", TermType.STRING),
            "Y": Constant("b", TermType.STRING),
        }


class TestApplySubst:
    def test_apply_to_var(self):
        result = apply_subst(Variable("X"), {"X": Constant("a", TermType.STRING)})
        assert result == Constant("a", TermType.STRING)

    def test_apply_to_atom(self):
        atom = Atom("edge", (Variable("X"), Variable("Y")))
        result = apply_subst(atom, {
            "X": Constant("a", TermType.STRING),
            "Y": Constant("b", TermType.STRING),
        })
        assert result == Atom("edge", (Constant("a", TermType.STRING), Constant("b", TermType.STRING)))
