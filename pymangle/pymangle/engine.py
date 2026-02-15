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
    ListTerm,
    NegAtom,
    Premise,
    Program,
    TermType,
    Transform,
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
    from pymangle.types import TypeChecker

    if store is None:
        store = IndexedFactStore()

    # Set up type checker if declarations exist
    checker = TypeChecker(program.decls) if program.decls else None

    # Load initial facts
    for fact_clause in program.facts:
        store.add(fact_clause.head)
        if checker is not None:
            checker.check_bounds(fact_clause.head)

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

    # Separate plain rules from aggregation rules
    plain_rules = [r for r in rules if r.transform is None]
    agg_rules = [r for r in rules if r.transform is not None]

    # Initial round: apply plain rules
    delta = IndexedFactStore()
    for rule in plain_rules:
        for fact in _eval_rule(rule, store, store):
            if store.add(fact):
                delta.add(fact)
                total_derived += 1
                if total_derived > fact_limit:
                    raise FactLimitError(f"Exceeded fact limit of {fact_limit}")

    # Delta iteration for plain rules
    while len(delta) > 0:
        new_delta = IndexedFactStore()
        for rule in plain_rules:
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

    # Post-fixpoint: evaluate aggregation rules
    for rule in agg_rules:
        for fact in _eval_aggregate(rule, store):
            if store.add(fact):
                total_derived += 1
                if total_derived > fact_limit:
                    raise FactLimitError(f"Exceeded fact limit of {fact_limit}")

    return total_derived


def _eval_aggregate(rule: Clause, store: IndexedFactStore) -> list[Atom]:
    """Evaluate an aggregation rule: solve body, group, reduce, emit head facts."""
    transform = rule.transform
    if transform is None:
        return []

    # Collect all substitutions from body premises
    all_substs = _solve_premises(rule.premises, 0, {}, store, store)

    # Determine group-by variable names
    group_vars = []
    for t in transform.group_by:
        if isinstance(t, Variable):
            group_vars.append(t.name)

    # Group substitutions by group-by keys
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for subst in all_substs:
        key = tuple(subst.get(v) for v in group_vars) if group_vars else ()
        groups[key].append(subst)

    # Apply reducer per group and emit head facts
    results = []
    reducer = transform.reducer
    if reducer is None:
        return results

    reducer_name = reducer.name
    # Get the variable name being aggregated (first arg of reducer if any)
    agg_var = None
    if reducer.args and isinstance(reducer.args[0], Variable):
        agg_var = reducer.args[0].name

    for group_key, substs in groups.items():
        # Compute aggregate value
        agg_value = _apply_reducer(reducer_name, agg_var, substs)
        if agg_value is None:
            continue

        # Build substitution for head: group vars + aggregate variable
        head_subst: dict[str, object] = {}
        for i, var_name in enumerate(group_vars):
            head_subst[var_name] = group_key[i]
        if transform.variable is not None:
            head_subst[transform.variable.name] = agg_value

        head = apply_subst(rule.head, head_subst)
        if isinstance(head, Atom) and is_ground(head):
            results.append(head)

    return results


def _apply_reducer(name: str, agg_var: str | None, substs: list[dict]) -> Constant | ListTerm | None:
    """Apply a reducer function to a group of substitutions."""
    if name == "fn:count":
        return Constant(len(substs), TermType.NUMBER)

    if agg_var is None:
        return None

    values = [s[agg_var] for s in substs if agg_var in s]

    if name == "fn:sum":
        total = sum(v.value if isinstance(v, Constant) else 0 for v in values)
        if isinstance(total, float):
            return Constant(total, TermType.FLOAT)
        return Constant(total, TermType.NUMBER)

    if name == "fn:avg":
        if not values:
            return None
        total = sum(v.value if isinstance(v, Constant) else 0 for v in values)
        return Constant(total / len(values), TermType.FLOAT)

    if name == "fn:min":
        nums = [v.value for v in values if isinstance(v, Constant)]
        if not nums:
            return None
        result = min(nums)
        return Constant(result, TermType.FLOAT if isinstance(result, float) else TermType.NUMBER)

    if name == "fn:max":
        nums = [v.value for v in values if isinstance(v, Constant)]
        if not nums:
            return None
        result = max(nums)
        return Constant(result, TermType.FLOAT if isinstance(result, float) else TermType.NUMBER)

    if name == "fn:collect":
        return ListTerm(tuple(values))

    return None


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
