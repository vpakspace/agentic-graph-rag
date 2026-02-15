"""Built-in functions and predicates for Mangle programs."""
from __future__ import annotations

from pymangle.ast_nodes import Constant, FunCall, ListTerm, Term, TermType


def _numeric_result(value: int | float) -> Constant:
    """Wrap a numeric result in the correct Constant type."""
    if isinstance(value, float):
        return Constant(value, TermType.FLOAT)
    return Constant(value, TermType.NUMBER)


def _two_nums(args: tuple[Term, ...]) -> tuple[int | float, int | float] | None:
    """Extract two numeric values from args, or None."""
    if len(args) != 2:
        return None
    a, b = args
    if isinstance(a, Constant) and isinstance(b, Constant):
        return a.value, b.value
    return None


def eval_funcall(funcall: FunCall) -> Constant | ListTerm | None:
    """Evaluate a built-in function call. Returns None if unknown."""
    name = funcall.name
    args = funcall.args

    # --- Arithmetic ---
    if name == "fn:plus":
        pair = _two_nums(args)
        if pair:
            return _numeric_result(pair[0] + pair[1])

    elif name == "fn:minus":
        pair = _two_nums(args)
        if pair:
            return _numeric_result(pair[0] - pair[1])

    elif name == "fn:mult":
        pair = _two_nums(args)
        if pair:
            return _numeric_result(pair[0] * pair[1])

    elif name == "fn:div":
        pair = _two_nums(args)
        if pair and pair[1] != 0:
            return _numeric_result(int(pair[0] // pair[1]))

    elif name == "fn:float_div":
        pair = _two_nums(args)
        if pair and pair[1] != 0:
            return Constant(pair[0] / pair[1], TermType.FLOAT)

    # --- String functions ---
    elif name == "fn:string:concat":
        if len(args) == 2 and isinstance(args[0], Constant) and isinstance(args[1], Constant):
            return Constant(str(args[0].value) + str(args[1].value), TermType.STRING)

    elif name == "fn:string:len":
        if len(args) == 1 and isinstance(args[0], Constant):
            return Constant(len(str(args[0].value)), TermType.NUMBER)

    # --- List functions ---
    elif name == "fn:list":
        return ListTerm(args)

    elif name == "fn:len":
        if len(args) == 1 and isinstance(args[0], ListTerm):
            return Constant(len(args[0].elements), TermType.NUMBER)

    return None


def eval_builtin_pred(name: str, args: list[Term]) -> bool:
    """Evaluate a built-in predicate (: prefix). Returns False if unknown."""

    # --- String predicates ---
    if name == ":string:starts_with":
        if len(args) == 2 and isinstance(args[0], Constant) and isinstance(args[1], Constant):
            return str(args[0].value).startswith(str(args[1].value))

    elif name == ":string:contains":
        if len(args) == 2 and isinstance(args[0], Constant) and isinstance(args[1], Constant):
            return str(args[1].value) in str(args[0].value)

    # --- List predicates ---
    elif name == ":list:member":
        if len(args) == 2 and isinstance(args[1], ListTerm):
            return args[0] in args[1].elements

    return False
