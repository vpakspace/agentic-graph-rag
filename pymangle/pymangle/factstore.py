"""Indexed in-memory fact storage."""
from __future__ import annotations

import logging
from collections import defaultdict

from pymangle.ast_nodes import Atom, Constant, Variable
from pymangle.unifier import unify

logger = logging.getLogger(__name__)


class IndexedFactStore:
    """Hash-indexed fact store with predicate and first-arg indexes."""

    def __init__(self) -> None:
        self._facts: set[Atom] = set()
        self._by_predicate: dict[str, set[Atom]] = defaultdict(set)
        self._by_first_arg: dict[tuple[str, Constant], set[Atom]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._facts)

    def add(self, fact: Atom) -> bool:
        """Add a ground fact. Returns True if fact is new."""
        if fact in self._facts:
            return False
        self._facts.add(fact)
        self._by_predicate[fact.predicate].add(fact)
        if fact.args and isinstance(fact.args[0], Constant):
            self._by_first_arg[(fact.predicate, fact.args[0])].add(fact)
        return True

    def query(self, pattern: Atom) -> list[Atom]:
        """Find facts matching pattern (with variables as wildcards)."""
        # Use first-arg index if first arg is ground
        if pattern.args and isinstance(pattern.args[0], Constant):
            candidates = self._by_first_arg.get((pattern.predicate, pattern.args[0]), set())
        else:
            candidates = self._by_predicate.get(pattern.predicate, set())

        results = []
        for fact in candidates:
            if unify(pattern, fact, {}) is not None:
                results.append(fact)
        return results

    def merge(self, other: IndexedFactStore) -> IndexedFactStore:
        """Merge other store into self. Returns store with only NEW facts."""
        delta = IndexedFactStore()
        for fact in other._facts:
            if self.add(fact):
                delta.add(fact)
        return delta

    def get_by_predicate(self, predicate: str) -> list[Atom]:
        """Get all facts for a predicate."""
        return list(self._by_predicate.get(predicate, []))

    def all_facts(self) -> list[Atom]:
        """Get all facts."""
        return list(self._facts)

    def contains(self, fact: Atom) -> bool:
        """Check if exact fact exists."""
        return fact in self._facts
