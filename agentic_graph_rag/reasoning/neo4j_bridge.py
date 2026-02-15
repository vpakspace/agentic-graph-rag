"""Neo4j external predicate — bridges PyMangle rules to graph database."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from pymangle.ast_nodes import Constant, TermType

if TYPE_CHECKING:
    from neo4j import Driver

logger = logging.getLogger(__name__)


def _to_constant(value: object) -> Constant:
    """Convert a Neo4j value to a PyMangle Constant."""
    if isinstance(value, bool):
        return Constant(str(value).lower(), TermType.STRING)
    if isinstance(value, int):
        return Constant(value, TermType.NUMBER)
    if isinstance(value, float):
        return Constant(value, TermType.FLOAT)
    return Constant(str(value), TermType.STRING)


class Neo4jExternalPredicate:
    """External predicate that executes a Cypher query against Neo4j.

    Implements the ExternalPredicate protocol from pymangle.external.
    Bound input args are passed as Cypher parameters $p0, $p1, ...
    """

    def __init__(self, driver: Driver, cypher_template: str) -> None:
        self._driver = driver
        self._cypher = cypher_template

    def query(
        self, inputs: list[Constant], filters: list
    ) -> Iterator[list[Constant]]:
        """Execute Cypher and yield rows as lists of Constants."""
        params: dict[str, object] = {}
        for i, inp in enumerate(inputs):
            params[f"p{i}"] = inp.value

        with self._driver.session() as session:
            result = session.run(self._cypher, params)
            for record in result:
                yield [_to_constant(v) for v in record.values()]
