"""Semi-naive bottom-up evaluation engine."""
from __future__ import annotations

import logging
from collections import defaultdict

from pymangle.ast_nodes import (
    Atom,
    Clause,
    Comparison,
    Constant,
    FunCall,
    NegAtom,
    Premise,
    Program,
    Variable,
)
from pymangle.factstore import IndexedFactStore
from pymangle.unifier import apply_subst, is_ground, unify

logger = logging.getLogger(__name__)


class FactLimitError(Exception):
    """Raised when derived fact limit is exceeded."""


def eval_program(
    program: Program,
    store: IndexedFactStore | None = None,
    fact_limit: int = 100_000,
) -> IndexedFactStore:
    """Evaluate a Mangle program using stratified semi-naive bottom-up evaluation.

    Returns the fact store containing all derived facts.
    """
    from pymangle.analysis import stratify

    if store is None:
        store = IndexedFactStore()

    # Load initial facts
    for fact_clause in program.facts:
        store.add(fact_clause.head)

    if not program.clauses:
        return store

    # Stratify and evaluate each stratum to fixpoint
    strata = stratify(program)
    total_derived = 0

    for stratum in strata:
        if not stratum.rules:
            continue
        total_derived += _eval_stratum(stratum.rules, store, fact_limit - total_derived)

    logger.info("Evaluation complete: %d facts derived", total_derived)
    return store


def _eval_stratum(
    rules: list[Clause],
    store: IndexedFactStore,
    fact_limit: int,
) -> int:
    """Evaluate a single stratum to fixpoint. Returns count of new facts."""
    total_derived = 0

    # Initial round: apply all rules
    delta = IndexedFactStore()
    for rule in rules:
        for fact in _eval_rule(rule, store, store):
            if store.add(fact):
                delta.add(fact)
                total_derived += 1
                if total_derived > fact_limit:
                    raise FactLimitError(f"Exceeded fact limit of {fact_limit}")

    # Delta iteration
    while len(delta) > 0:
        new_delta = IndexedFactStore()
        for rule in rules:
            for i, premise in enumerate(rule.premises):
                if isinstance(premise, (NegAtom, Comparison)):
                    continue
                for fact in _eval_rule_delta(rule, i, store, delta):
                    if store.add(fact):
                        new_delta.add(fact)
                        total_derived += 1
                        if total_derived > fact_limit:
                            raise FactLimitError(f"Exceeded fact limit of {fact_limit}")
        delta = new_delta

    return total_derived


def _eval_rule(
    rule: Clause,
    store: IndexedFactStore,
    search_store: IndexedFactStore,
) -> list[Atom]:
    """Evaluate a single rule against the store."""
    results = []
    for subst in _solve_premises(rule.premises, 0, {}, store, search_store):
        head = apply_subst(rule.head, subst)
        if isinstance(head, Atom) and is_ground(head):
            results.append(head)
    return results


def _eval_rule_delta(
    rule: Clause,
    delta_idx: int,
    store: IndexedFactStore,
    delta: IndexedFactStore,
) -> list[Atom]:
    """Evaluate rule with i-th premise using delta store."""
    results = []
    for subst in _solve_premises_delta(rule.premises, 0, delta_idx, {}, store, delta):
        head = apply_subst(rule.head, subst)
        if isinstance(head, Atom) and is_ground(head):
            results.append(head)
    return results


def _solve_premises(
    premises: tuple[Premise, ...],
    idx: int,
    subst: dict[str, object],
    store: IndexedFactStore,
    search_store: IndexedFactStore,
) -> list[dict]:
    """Recursively solve premises left-to-right."""
    if idx >= len(premises):
        return [subst]

    premise = premises[idx]
    results = []

    if isinstance(premise, Atom):
        pattern = apply_subst(premise, subst)
        for fact in search_store.query(pattern):
            new_subst = unify(pattern, fact, subst)
            if new_subst is not None:
                results.extend(_solve_premises(premises, idx + 1, new_subst, store, search_store))

    elif isinstance(premise, NegAtom):
        pattern = apply_subst(premise.atom, subst)
        matches = store.query(pattern)
        if not matches:
            results.extend(_solve_premises(premises, idx + 1, subst, store, search_store))

    elif isinstance(premise, Comparison):
        left = apply_subst(premise.left, subst)
        right = apply_subst(premise.right, subst)

        # Handle assignment: X = fn:plus(Y, 1) or X = value
        if premise.op == "=" and isinstance(left, Variable) and is_ground(right):
            if isinstance(right, FunCall):
                from pymangle.builtins import eval_funcall

                val = eval_funcall(right)
                if val is not None:
                    new_subst = dict(subst)
                    new_subst[left.name] = val
                    results.extend(_solve_premises(premises, idx + 1, new_subst, store, search_store))
            else:
                new_subst = dict(subst)
                new_subst[left.name] = right
                results.extend(_solve_premises(premises, idx + 1, new_subst, store, search_store))
        elif is_ground(left) and is_ground(right):
            if _eval_comparison(left, right, premise.op):
                results.extend(_solve_premises(premises, idx + 1, subst, store, search_store))

    return results


def _solve_premises_delta(
    premises: tuple[Premise, ...],
    idx: int,
    delta_idx: int,
    subst: dict[str, object],
    store: IndexedFactStore,
    delta: IndexedFactStore,
) -> list[dict]:
    """Solve premises with delta_idx-th premise using delta store."""
    if idx >= len(premises):
        return [subst]

    premise = premises[idx]
    results = []

    if isinstance(premise, Atom):
        pattern = apply_subst(premise, subst)
        # Use delta store for the delta premise, full store for others
        use_store = delta if idx == delta_idx else store
        for fact in use_store.query(pattern):
            new_subst = unify(pattern, fact, subst)
            if new_subst is not None:
                results.extend(_solve_premises_delta(premises, idx + 1, delta_idx, new_subst, store, delta))

    elif isinstance(premise, NegAtom):
        pattern = apply_subst(premise.atom, subst)
        matches = store.query(pattern)
        if not matches:
            results.extend(_solve_premises_delta(premises, idx + 1, delta_idx, subst, store, delta))

    elif isinstance(premise, Comparison):
        left = apply_subst(premise.left, subst)
        right = apply_subst(premise.right, subst)
        if premise.op == "=" and isinstance(left, Variable) and is_ground(right):
            if isinstance(right, FunCall):
                from pymangle.builtins import eval_funcall

                val = eval_funcall(right)
                if val is not None:
                    new_subst = dict(subst)
                    new_subst[left.name] = val
                    results.extend(_solve_premises_delta(premises, idx + 1, delta_idx, new_subst, store, delta))
            else:
                new_subst = dict(subst)
                new_subst[left.name] = right
                results.extend(_solve_premises_delta(premises, idx + 1, delta_idx, new_subst, store, delta))
        elif is_ground(left) and is_ground(right):
            if _eval_comparison(left, right, premise.op):
                results.extend(_solve_premises_delta(premises, idx + 1, delta_idx, subst, store, delta))

    return results


def _eval_comparison(left: object, right: object, op: str) -> bool:
    """Evaluate comparison between two ground terms."""
    lv = left.value if isinstance(left, Constant) else left
    rv = right.value if isinstance(right, Constant) else right

    if op == "==" or op == "=":
        return lv == rv
    if op == "!=":
        return lv != rv
    if op == "<":
        return lv < rv
    if op == "<=":
        return lv <= rv
    if op == ">":
        return lv > rv
    if op == ">=":
        return lv >= rv
    return False
