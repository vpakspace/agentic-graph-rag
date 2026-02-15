"""ReasoningEngine facade — loads Mangle rules and provides high-level API."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pymangle.ast_nodes import Atom, Constant, TermType
from pymangle.engine import eval_program
from pymangle.external import ExternalPredicateRegistry
from pymangle.parser import parse

if TYPE_CHECKING:
    from neo4j import Driver

logger = logging.getLogger(__name__)


class _QueryContainsPredicate:
    """Built-in external: query_contains(Query, Keyword) succeeds if keyword is in query."""

    def __init__(self) -> None:
        self._query: str = ""

    def set_query(self, query: str) -> None:
        self._query = query.lower()

    def query(self, inputs: list[Constant], filters: list):
        """Check if keyword is contained in the current query string.

        Returns [query_string, keyword] rows so that the atom
        query_contains(Q, Keyword) unifies correctly.
        """
        if not inputs:
            return iter([])
        keyword = str(inputs[0].value).lower()
        if keyword in self._query:
            yield [Constant(self._query, TermType.STRING), inputs[0]]
        return


class ReasoningEngine:
    """High-level facade for Mangle-based reasoning.

    Loads .mg rule files from a directory and provides:
    - classify_query(query) — route queries using declarative rules
    - check_access(role, action) — evaluate access control rules
    - infer_connections(entity) — derive graph relationships
    """

    def __init__(self, rules_dir: str, driver: Driver | None = None) -> None:
        self._rules_dir = Path(rules_dir)
        self._driver = driver
        self._query_contains = _QueryContainsPredicate()
        self._rule_sources: dict[str, str] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        """Load all .mg files from rules directory."""
        if not self._rules_dir.exists():
            logger.warning("Rules directory does not exist: %s", self._rules_dir)
            return
        for path in sorted(self._rules_dir.glob("*.mg")):
            self._rule_sources[path.stem] = path.read_text()
            logger.info("Loaded rule file: %s", path.name)

    def _build_registry(self) -> ExternalPredicateRegistry:
        """Build external predicate registry with built-in and Neo4j predicates."""
        registry = ExternalPredicateRegistry()
        registry.register("query_contains", self._query_contains)
        return registry

    def classify_query(self, query: str) -> dict | None:
        """Classify a query using routing rules.

        Returns dict with 'tool' and 'query' keys, or None if no route matched.
        """
        source = self._rule_sources.get("routing")
        if source is None:
            return None

        self._query_contains.set_query(query)
        program = parse(source)
        registry = self._build_registry()

        # Add query fact
        query_fact = f'current_query("{_escape(query)}").\n'
        full_source = query_fact + source
        program = parse(full_source)

        store = eval_program(program, externals=registry)
        routes = store.get_by_predicate("route_to")

        for route in routes:
            if len(route.args) >= 2:
                tool = route.args[0].value if isinstance(route.args[0], Constant) else str(route.args[0])
                return {"tool": str(tool), "query": query}

        return None

    def check_access(self, role: str, action: str) -> bool:
        """Check if role is permitted to perform action.

        Returns True if permit(role, action) can be derived.
        """
        source = self._rule_sources.get("access")
        if source is None:
            return True  # No access rules = permit all

        program = parse(source)
        store = eval_program(program)

        permits = store.get_by_predicate("permit")
        for fact in permits:
            if len(fact.args) >= 2:
                fact_role = fact.args[0].value if isinstance(fact.args[0], Constant) else None
                fact_action = fact.args[1].value if isinstance(fact.args[1], Constant) else None
                if fact_role == role and fact_action == action:
                    return True

        return False

    def infer_connections(self, entity: str) -> list[dict]:
        """Derive connections for an entity using graph rules.

        Returns list of dicts with relationship info.
        """
        source = self._rule_sources.get("graph")
        if source is None:
            return []

        entity_fact = f'query_entity("{_escape(entity)}").\n'
        full_source = entity_fact + source
        program = parse(full_source)
        registry = self._build_registry()

        store = eval_program(program, externals=registry)
        connections = store.get_by_predicate("connected")

        results = []
        for fact in connections:
            args = [a.value if isinstance(a, Constant) else str(a) for a in fact.args]
            results.append({"args": args})

        return results


def _escape(s: str) -> str:
    """Escape string for embedding in Mangle source."""
    return s.replace("\\", "\\\\").replace('"', '\\"')
