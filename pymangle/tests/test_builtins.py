"""Tests for built-in functions and predicates."""
from __future__ import annotations

import pytest

from pymangle.ast_nodes import Constant, FunCall, ListTerm, TermType
from pymangle.builtins import eval_builtin_pred, eval_funcall


class TestArithmetic:
    def test_plus(self):
        result = eval_funcall(FunCall("fn:plus", (Constant(3, TermType.NUMBER), Constant(4, TermType.NUMBER))))
        assert result == Constant(7, TermType.NUMBER)

    def test_minus(self):
        result = eval_funcall(FunCall("fn:minus", (Constant(10, TermType.NUMBER), Constant(3, TermType.NUMBER))))
        assert result == Constant(7, TermType.NUMBER)

    def test_mult(self):
        result = eval_funcall(FunCall("fn:mult", (Constant(3, TermType.NUMBER), Constant(4, TermType.NUMBER))))
        assert result == Constant(12, TermType.NUMBER)

    def test_div(self):
        """Integer division."""
        result = eval_funcall(FunCall("fn:div", (Constant(10, TermType.NUMBER), Constant(3, TermType.NUMBER))))
        assert result == Constant(3, TermType.NUMBER)

    def test_float_div(self):
        result = eval_funcall(FunCall("fn:float_div", (Constant(10, TermType.NUMBER), Constant(4, TermType.NUMBER))))
        assert result == Constant(2.5, TermType.FLOAT)

    def test_float_result(self):
        """Float input produces float output."""
        result = eval_funcall(FunCall("fn:plus", (Constant(1.5, TermType.FLOAT), Constant(2, TermType.NUMBER))))
        assert result == Constant(3.5, TermType.FLOAT)

    def test_div_by_zero(self):
        result = eval_funcall(FunCall("fn:div", (Constant(10, TermType.NUMBER), Constant(0, TermType.NUMBER))))
        assert result is None


class TestString:
    def test_concat(self):
        result = eval_funcall(FunCall("fn:string:concat", (Constant("hello", TermType.STRING), Constant(" world", TermType.STRING))))
        assert result == Constant("hello world", TermType.STRING)

    def test_len(self):
        result = eval_funcall(FunCall("fn:string:len", (Constant("hello", TermType.STRING),)))
        assert result == Constant(5, TermType.NUMBER)

    def test_starts_with(self):
        assert eval_builtin_pred(":string:starts_with", [Constant("hello", TermType.STRING), Constant("hel", TermType.STRING)]) is True
        assert eval_builtin_pred(":string:starts_with", [Constant("hello", TermType.STRING), Constant("xyz", TermType.STRING)]) is False

    def test_contains(self):
        assert eval_builtin_pred(":string:contains", [Constant("hello world", TermType.STRING), Constant("lo wo", TermType.STRING)]) is True
        assert eval_builtin_pred(":string:contains", [Constant("hello", TermType.STRING), Constant("xyz", TermType.STRING)]) is False


class TestList:
    def test_list_constructor(self):
        result = eval_funcall(FunCall("fn:list", (Constant(1, TermType.NUMBER), Constant(2, TermType.NUMBER))))
        assert isinstance(result, ListTerm)
        assert len(result.elements) == 2

    def test_len(self):
        lst = ListTerm((Constant(1, TermType.NUMBER), Constant(2, TermType.NUMBER), Constant(3, TermType.NUMBER)))
        result = eval_funcall(FunCall("fn:len", (lst,)))
        assert result == Constant(3, TermType.NUMBER)

    def test_member(self):
        lst = ListTerm((Constant(1, TermType.NUMBER), Constant(2, TermType.NUMBER)))
        assert eval_builtin_pred(":list:member", [Constant(1, TermType.NUMBER), lst]) is True
        assert eval_builtin_pred(":list:member", [Constant(9, TermType.NUMBER), lst]) is False


class TestUnknown:
    def test_unknown_funcall(self):
        result = eval_funcall(FunCall("fn:nonexistent", ()))
        assert result is None

    def test_unknown_pred(self):
        result = eval_builtin_pred(":nonexistent", [])
        assert result is False
