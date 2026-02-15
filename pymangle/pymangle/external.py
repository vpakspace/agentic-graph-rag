"""External predicate protocol for Mangle programs."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from pymangle.ast_nodes import Atom, Constant


@runtime_checkable
class ExternalPredicate(Protocol):
    """Protocol for external data sources."""

    def query(
        self, inputs: list[Constant], filters: list
    ) -> Iterator[list[Constant]]: ...


class ExternalPredicateRegistry:
    """Registry of external predicates available during evaluation."""

    def __init__(self) -> None:
        self._externals: dict[str, ExternalPredicate] = {}

    def register(self, predicate: str, callback: ExternalPredicate) -> None:
        """Register an external predicate implementation."""
        self._externals[predicate] = callback

    def has(self, predicate: str) -> bool:
        """Check if a predicate has an external implementation."""
        return predicate in self._externals

    def query(
        self, predicate: str, inputs: list[Constant], filters: list
    ) -> Iterator[list[Constant]]:
        """Query an external predicate."""
        ext = self._externals.get(predicate)
        if ext is None:
            return iter([])
        return ext.query(inputs, filters)

    def query_as_atoms(
        self, predicate: str, inputs: list[Constant], filters: list
    ) -> list[Atom]:
        """Query external predicate and wrap results as Atoms."""
        results = []
        for row in self.query(predicate, inputs, filters):
            results.append(Atom(predicate, tuple(row)))
        return results
