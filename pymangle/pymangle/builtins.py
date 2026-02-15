"""Built-in predicates and functions (stub — expanded in Phase 5)."""
from __future__ import annotations

from pymangle.ast_nodes import Constant, FunCall, TermType


def eval_funcall(funcall: FunCall) -> Constant | None:
    """Evaluate a built-in function call. Returns None if unknown."""
    name = funcall.name
    args = funcall.args

    if name == "fn:plus" and len(args) == 2:
        a, b = args
        if isinstance(a, Constant) and isinstance(b, Constant):
            result = a.value + b.value
            if isinstance(result, float):
                return Constant(result, TermType.FLOAT)
            return Constant(result, TermType.NUMBER)

    return None
