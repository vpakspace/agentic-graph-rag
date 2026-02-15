"""Temporal fact store and Allen's interval algebra."""
from __future__ import annotations

from datetime import datetime

from pymangle.ast_nodes import Atom


class TemporalFactStore:
    """Stores facts annotated with time intervals [start, end).

    Supports point-in-time queries, range queries, and interval coalescing.
    """

    def __init__(self) -> None:
        # atom -> list of (start, end) intervals, kept sorted and coalesced
        self._intervals: dict[Atom, list[tuple[datetime, datetime]]] = {}

    def add(self, atom: Atom, start: datetime, end: datetime) -> None:
        """Add a temporal fact. Overlapping/adjacent intervals are coalesced."""
        if atom not in self._intervals:
            self._intervals[atom] = [(start, end)]
            return

        intervals = self._intervals[atom]
        intervals.append((start, end))
        self._intervals[atom] = _coalesce(intervals)

    def get_intervals(self, atom: Atom) -> list[tuple[datetime, datetime]]:
        """Return all intervals for a given atom."""
        return list(self._intervals.get(atom, []))

    def query_at(self, point: datetime) -> list[tuple[Atom, datetime, datetime]]:
        """Return all facts valid at a specific point in time."""
        results = []
        for atom, intervals in self._intervals.items():
            for start, end in intervals:
                if start <= point < end:
                    results.append((atom, start, end))
        return results

    def query_range(
        self, q_start: datetime, q_end: datetime
    ) -> list[tuple[Atom, datetime, datetime]]:
        """Return all facts whose interval overlaps [q_start, q_end)."""
        results = []
        for atom, intervals in self._intervals.items():
            for start, end in intervals:
                if start < q_end and q_start < end:  # overlap check
                    results.append((atom, start, end))
        return results

    def all_atoms(self) -> set[Atom]:
        """Return all atoms in the store."""
        return set(self._intervals.keys())

    def __len__(self) -> int:
        return sum(len(ivs) for ivs in self._intervals.values())


def _coalesce(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Merge overlapping or adjacent intervals."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda iv: iv[0])
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:  # overlapping or adjacent
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------------------
# Allen's interval algebra — 13 relations (7 base + 6 inverses)
# Given intervals A=[a1,a2) and B=[b1,b2):
# ---------------------------------------------------------------------------


def allen_relation(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
) -> str:
    """Determine Allen's relation between intervals A and B.

    Returns one of: before, after, meets, met_by, overlaps, overlapped_by,
    starts, started_by, finishes, finished_by, during, contains, equals.
    """
    if a_end < b_start:
        return "before"
    if b_end < a_start:
        return "after"
    if a_end == b_start:
        return "meets"
    if b_end == a_start:
        return "met_by"

    # Both intervals overlap in some way
    if a_start == b_start and a_end == b_end:
        return "equals"
    if a_start == b_start:
        return "starts" if a_end < b_end else "started_by"
    if a_end == b_end:
        return "finishes" if a_start > b_start else "finished_by"

    # Proper overlap / containment
    if a_start < b_start:
        if a_end <= b_end:
            return "overlaps"
        return "contains"
    else:  # a_start > b_start
        if a_end >= b_end:
            return "overlapped_by"
        return "during"
