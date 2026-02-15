"""Optional type bounds checking for Mangle programs."""
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

logger = logging.getLogger(__name__)

# Map TermType to Mangle type name constants
_TYPE_MAP: dict[TermType, str] = {
    TermType.STRING: "/string",
    TermType.NUMBER: "/number",
    TermType.FLOAT: "/float",
    TermType.NAME: "/name",
    TermType.LIST: "/list",
    TermType.MAP: "/map",
    TermType.STRUCT: "/struct",
}


class TypeChecker:
    """Checks derived facts against declared type bounds.

    Bounds checking is advisory — violations produce warnings, not errors.
    """

    def __init__(self, decls: list[Decl]) -> None:
        self._decls: dict[str, Decl] = {d.predicate: d for d in decls}

    def check_bounds(self, fact: Atom) -> bool:
        """Check if a fact matches its declared type bounds.

        Returns True if no declaration exists or all args match.
        Returns False and logs warning if any arg violates bounds.
        """
        decl = self._decls.get(fact.predicate)
        if decl is None:
            return True

        if not decl.bounds:
            return True

        all_ok = True
        for i, (arg, bound) in enumerate(zip(fact.args, decl.bounds)):
            if not _arg_matches_bound(arg, bound):
                arg_type = arg.type.value if isinstance(arg, Constant) else "unknown"
                logger.warning(
                    "Type bounds violation: %s arg %d has type %s, "
                    "expected one of %s",
                    fact.predicate, i, arg_type, bound.types,
                )
                all_ok = False

        return all_ok

    def is_external(self, predicate: str) -> bool:
        """Check if a predicate is declared as external."""
        decl = self._decls.get(predicate)
        if decl is None:
            return False
        return DeclDescriptor.EXTERNAL in decl.descriptors


def _arg_matches_bound(arg: object, bound: BoundDecl) -> bool:
    """Check if an argument matches any type in the bound."""
    if not isinstance(arg, Constant):
        return True  # non-constant args (variables) skip checking

    arg_type_name = _TYPE_MAP.get(arg.type)
    if arg_type_name is None:
        return True  # unknown type — skip

    return arg_type_name in bound.types
