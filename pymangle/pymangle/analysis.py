"""Dependency analysis and stratification for Mangle programs."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from pymangle.ast_nodes import Atom, Clause, NegAtom, Premise, Program

logger = logging.getLogger(__name__)


class StratificationError(Exception):
    """Raised when program cannot be stratified (e.g. recursive negation)."""


@dataclass
class Stratum:
    """A group of rules that can be evaluated together."""
    rules: list[Clause] = field(default_factory=list)
    predicates: set[str] = field(default_factory=set)


class _DepEdge:
    """Dependency edge between predicates."""
    __slots__ = ("src", "dst", "is_negative")

    def __init__(self, src: str, dst: str, is_negative: bool) -> None:
        self.src = src
        self.dst = dst
        self.is_negative = is_negative


def build_dependency_graph(program: Program) -> tuple[dict[str, set[str]], list[_DepEdge]]:
    """Build predicate dependency graph from program clauses.

    Returns (adjacency dict, list of all edges with neg labels).
    """
    adj: dict[str, set[str]] = defaultdict(set)
    edges: list[_DepEdge] = []
    all_preds: set[str] = set()

    for clause in program.clauses:
        head_pred = clause.head.predicate
        all_preds.add(head_pred)
        has_transform = clause.transform is not None

        for premise in clause.premises:
            body_pred = _premise_predicate(premise)
            if body_pred is None:
                continue
            all_preds.add(body_pred)
            adj[head_pred].add(body_pred)
            is_neg = isinstance(premise, NegAtom)
            edges.append(_DepEdge(head_pred, body_pred, is_neg))

        # Transforms (aggregation) act like negation for stratification
        if has_transform:
            for premise in clause.premises:
                body_pred = _premise_predicate(premise)
                if body_pred is not None and body_pred != head_pred:
                    # Mark transform dependencies as negative-like
                    edges.append(_DepEdge(head_pred, body_pred, True))

    # Ensure all predicates are in adj
    for p in all_preds:
        if p not in adj:
            adj[p] = set()

    return adj, edges


def _premise_predicate(premise: Premise) -> str | None:
    """Extract predicate name from a premise."""
    if isinstance(premise, Atom):
        return premise.predicate
    if isinstance(premise, NegAtom):
        return premise.atom.predicate
    return None


def kosaraju_scc(adj: dict[str, set[str]]) -> list[set[str]]:
    """Compute strongly connected components using Kosaraju's algorithm.

    Returns list of SCCs in reverse topological order.
    """
    all_nodes = set(adj.keys())
    for deps in adj.values():
        all_nodes.update(deps)

    # Pass 1: DFS on original graph, record finish order
    visited: set[str] = set()
    finish_order: list[str] = []

    def _dfs1(node: str) -> None:
        visited.add(node)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                _dfs1(neighbor)
        finish_order.append(node)

    for node in sorted(all_nodes):
        if node not in visited:
            _dfs1(node)

    # Build reverse graph
    rev_adj: dict[str, set[str]] = defaultdict(set)
    for src, dsts in adj.items():
        for dst in dsts:
            rev_adj[dst].add(src)

    # Pass 2: DFS on reverse graph in reverse finish order
    visited.clear()
    sccs: list[set[str]] = []

    def _dfs2(node: str, scc: set[str]) -> None:
        visited.add(node)
        scc.add(node)
        for neighbor in rev_adj.get(node, set()):
            if neighbor not in visited:
                _dfs2(neighbor, scc)

    for node in reversed(finish_order):
        if node not in visited:
            scc: set[str] = set()
            _dfs2(node, scc)
            sccs.append(scc)

    return sccs


def stratify(program: Program) -> list[Stratum]:
    """Stratify program rules for correct evaluation order.

    Validates that no SCC contains a negative edge (which would make
    the program unstratifiable).

    Returns list of strata in evaluation order (lower index = earlier).
    """
    if not program.clauses:
        # No rules — single stratum with just facts
        return [Stratum()]

    adj, edges = build_dependency_graph(program)
    sccs = kosaraju_scc(adj)

    # Map each predicate to its SCC index
    pred_to_scc: dict[str, int] = {}
    for i, scc in enumerate(sccs):
        for pred in scc:
            pred_to_scc[pred] = i

    # Check for negative edges within an SCC
    for edge in edges:
        if edge.is_negative and edge.src in pred_to_scc and edge.dst in pred_to_scc:
            if pred_to_scc[edge.src] == pred_to_scc[edge.dst]:
                raise StratificationError(
                    f"Recursive negation: {edge.src} negatively depends on {edge.dst} "
                    f"within the same SCC"
                )

    # Assign stratum numbers based on negative dependencies
    # Start with all SCCs at stratum 0
    scc_stratum: dict[int, int] = {i: 0 for i in range(len(sccs))}

    # Propagate: if A negatively depends on B, A.stratum > B.stratum
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.src not in pred_to_scc or edge.dst not in pred_to_scc:
                continue
            src_scc = pred_to_scc[edge.src]
            dst_scc = pred_to_scc[edge.dst]
            if src_scc == dst_scc:
                continue
            if edge.is_negative:
                needed = scc_stratum[dst_scc] + 1
            else:
                needed = scc_stratum[dst_scc]
            if scc_stratum[src_scc] < needed:
                scc_stratum[src_scc] = needed
                changed = True

    # Group SCCs by stratum
    max_stratum = max(scc_stratum.values()) if scc_stratum else 0
    strata: list[Stratum] = [Stratum() for _ in range(max_stratum + 1)]

    for scc_idx, scc in enumerate(sccs):
        s = scc_stratum[scc_idx]
        strata[s].predicates.update(scc)

    # Assign rules to strata
    for clause in program.clauses:
        head_pred = clause.head.predicate
        if head_pred in pred_to_scc:
            scc_idx = pred_to_scc[head_pred]
            s = scc_stratum[scc_idx]
            strata[s].rules.append(clause)
        else:
            strata[0].rules.append(clause)

    # Remove empty strata
    strata = [s for s in strata if s.rules or s.predicates]

    logger.info("Stratified into %d strata", len(strata))
    return strata
