"""Substitution-based unification for Mangle terms."""
from __future__ import annotations

from pymangle.ast_nodes import Atom, Constant, FunCall, Term, Variable


def unify(t1: Term | Atom, t2: Term | Atom, subst: dict[str, Term]) -> dict[str, Term] | None:
    """Unify two terms under existing substitution.

    Returns extended substitution or None if unification fails.
    """
    # Resolve variables through existing substitution
    if isinstance(t1, Variable) and t1.name != "_" and t1.name in subst:
        return unify(subst[t1.name], t2, subst)
    if isinstance(t2, Variable) and t2.name != "_" and t2.name in subst:
        return unify(t1, subst[t2.name], subst)

    # Wildcard unifies with anything
    if isinstance(t1, Variable) and t1.name == "_":
        return dict(subst)
    if isinstance(t2, Variable) and t2.name == "_":
        return dict(subst)

    # Variable binds
    if isinstance(t1, Variable):
        new_subst = dict(subst)
        new_subst[t1.name] = t2
        return new_subst
    if isinstance(t2, Variable):
        new_subst = dict(subst)
        new_subst[t2.name] = t1
        return new_subst

    # Constants must be equal
    if isinstance(t1, Constant) and isinstance(t2, Constant):
        return dict(subst) if t1 == t2 else None

    # Atoms: same predicate + arity, unify args pairwise
    if isinstance(t1, Atom) and isinstance(t2, Atom):
        if t1.predicate != t2.predicate or t1.arity != t2.arity:
            return None
        current = dict(subst)
        for a1, a2 in zip(t1.args, t2.args):
            result = unify(a1, a2, current)
            if result is None:
                return None
            current = result
        return current

    # FunCall: same name, unify args
    if isinstance(t1, FunCall) and isinstance(t2, FunCall):
        if t1.name != t2.name or len(t1.args) != len(t2.args):
            return None
        current = dict(subst)
        for a1, a2 in zip(t1.args, t2.args):
            result = unify(a1, a2, current)
            if result is None:
                return None
            current = result
        return current

    return None


def apply_subst(term: Term | Atom, subst: dict[str, Term]) -> Term | Atom:
    """Apply substitution to a term, resolving all variables."""
    if isinstance(term, Variable):
        if term.name == "_" or term.name not in subst:
            return term
        resolved = subst[term.name]
        # Chase chains: X → Y → value
        if isinstance(resolved, Variable):
            return apply_subst(resolved, subst)
        return resolved

    if isinstance(term, Constant):
        return term

    if isinstance(term, Atom):
        return Atom(
            predicate=term.predicate,
            args=tuple(apply_subst(a, subst) for a in term.args),
        )

    if isinstance(term, FunCall):
        return FunCall(
            name=term.name,
            args=tuple(apply_subst(a, subst) for a in term.args),
        )

    return term


def is_ground(term: Term | Atom) -> bool:
    """Check if term contains no variables."""
    if isinstance(term, Variable):
        return False
    if isinstance(term, Constant):
        return True
    if isinstance(term, Atom):
        return all(is_ground(a) for a in term.args)
    if isinstance(term, FunCall):
        return all(is_ground(a) for a in term.args)
    return True
