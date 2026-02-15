"""Tests for optional type bounds checking."""
from __future__ import annotations

import logging

from pymangle.ast_nodes import (
    Atom,
    BoundDecl,
    Constant,
    Decl,
    DeclDescriptor,
    TermType,
)
from pymangle.engine import eval_program
from pymangle.parser import parse
from pymangle.types import TypeChecker


class TestTypeBounds:
    def test_bounds_check_pass(self):
        """Fact matches declared types — no warnings."""
        decl = Decl("person", 2, bounds=(
            BoundDecl(("/string",)),
            BoundDecl(("/number",)),
        ))
        checker = TypeChecker([decl])
        fact = Atom("person", (Constant("alice", TermType.STRING), Constant(30, TermType.NUMBER)))
        assert checker.check_bounds(fact) is True

    def test_bounds_check_fail(self, caplog):
        """Fact violates bounds — warning logged, returns False."""
        decl = Decl("person", 2, bounds=(
            BoundDecl(("/string",)),
            BoundDecl(("/number",)),
        ))
        checker = TypeChecker([decl])
        # Second arg is string instead of number
        fact = Atom("person", (Constant("alice", TermType.STRING), Constant("old", TermType.STRING)))
        with caplog.at_level(logging.WARNING):
            result = checker.check_bounds(fact)
        assert result is False

    def test_no_decl_no_check(self):
        """No declaration for predicate — always passes."""
        checker = TypeChecker([])
        fact = Atom("unknown", (Constant("anything", TermType.STRING),))
        assert checker.check_bounds(fact) is True

    def test_union_bounds(self):
        """Multiple types in a bound (union) — any match passes."""
        decl = Decl("value", 1, bounds=(
            BoundDecl(("/string", "/number")),
        ))
        checker = TypeChecker([decl])
        assert checker.check_bounds(Atom("value", (Constant("hi", TermType.STRING),))) is True
        assert checker.check_bounds(Atom("value", (Constant(42, TermType.NUMBER),))) is True
        assert checker.check_bounds(Atom("value", (Constant(3.14, TermType.FLOAT),))) is False

    def test_mode_declaration(self):
        """Decl with descriptor is recognized."""
        decl = Decl("ext_pred", 2,
                     descriptors=(DeclDescriptor.EXTERNAL,),
                     bounds=(BoundDecl(("/string",)), BoundDecl(("/string",))))
        checker = TypeChecker([decl])
        assert checker.is_external("ext_pred") is True
        assert checker.is_external("other") is False


class TestTypeBoundsIntegration:
    def test_eval_with_bounds(self, caplog):
        """eval_program with decls logs warnings for type violations."""
        prog = parse("""
            Decl person(X, Y) bound [/string] bound [/number].
            person("alice", 30).
            person("bob", "old").
        """)
        with caplog.at_level(logging.WARNING):
            store = eval_program(prog)
        # Both facts are loaded (bounds checking is advisory)
        assert len(list(store.get_by_predicate("person"))) == 2
        # Warning about "bob", "old" having wrong type
        assert any("bounds" in r.message.lower() or "type" in r.message.lower() for r in caplog.records)
