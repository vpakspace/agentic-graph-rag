"""ReasoningEngine facade — loads Mangle rules and provides high-level API."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pymangle.ast_nodes import Constant, TermType
from pymangle.engine import eval_program
from pymangle.external import ExternalPredicateRegistry
from pymangle.parser import parse

if TYPE_CHECKING:
    from neo4j import Driver

logger = logging.getLogger(__name__)


class _Neo4jEdgePredicate:
    """External predicate: edge(Source, Relation, Target) from Neo4j RELATED_TO edges."""

    def __init__(self, driver: Driver | None = None) -> None:
        self._driver = driver

    def query(self, inputs: list[Constant], filters: list):
        """Query Neo4j for edges. If Source is bound, returns outgoing edges."""
        if self._driver is None:
            return iter([])

        if inputs:
            source_name = str(inputs[0].value)
            cypher = (
                "MATCH (a:PhraseNode)-[r:RELATED_TO]->(b:PhraseNode) "
                "WHERE a.name = $name "
                "RETURN a.name AS src, type(r) AS rel, b.name AS tgt LIMIT 50"
            )
            with self._driver.session() as session:
                result = session.run(cypher, name=source_name)
                for record in result:
                    yield [
                        Constant(record["src"], TermType.STRING),
                        Constant(record["rel"], TermType.STRING),
                        Constant(record["tgt"], TermType.STRING),
                    ]
        else:
            cypher = (
                "MATCH (a:PhraseNode)-[r:RELATED_TO]->(b:PhraseNode) "
                "RETURN a.name AS src, type(r) AS rel, b.name AS tgt LIMIT 200"
            )
            with self._driver.session() as session:
                result = session.run(cypher)
                for record in result:
                    yield [
                        Constant(record["src"], TermType.STRING),
                        Constant(record["rel"], TermType.STRING),
                        Constant(record["tgt"], TermType.STRING),
                    ]


class _Neo4jMentionedInPredicate:
    """External predicate: mentioned_in(Entity, PassageId, Text) from Neo4j."""

    def __init__(self, driver: Driver | None = None) -> None:
        self._driver = driver

    def query(self, inputs: list[Constant], filters: list):
        """Query Neo4j for MENTIONED_IN relationships."""
        if self._driver is None:
            return iter([])

        if inputs:
            entity_name = str(inputs[0].value)
            cypher = (
                "MATCH (p:PhraseNode)-[:MENTIONED_IN]->(s:PassageNode) "
                "WHERE p.name = $name "
                "RETURN p.name AS entity, s.chunk_id AS passage_id, "
                "LEFT(s.text, 200) AS text LIMIT 50"
            )
            with self._driver.session() as session:
                result = session.run(cypher, name=entity_name)
                for record in result:
                    yield [
                        Constant(record["entity"], TermType.STRING),
                        Constant(record["passage_id"], TermType.STRING),
                        Constant(record["text"] or "", TermType.STRING),
                    ]
        else:
            return iter([])


class _QueryContainsPredicate:
    """Built-in external: query_contains(Query, Keyword) succeeds if keyword is in query."""

    def __init__(self) -> None:
        self._query: str = ""
        self._query_lower: str = ""

    def set_query(self, query: str) -> None:
        self._query = query
        self._query_lower = query.lower()

    def query(self, inputs: list[Constant], filters: list):
        """Check if keyword is contained in the current query string.

        Returns [query_string, keyword] rows so that the atom
        query_contains(Q, Keyword) unifies correctly.
        Uses original query string (not lowercased) to match current_query fact.
        """
        if not inputs:
            return iter([])
        keyword = str(inputs[0].value).lower()
        if keyword in self._query_lower:
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

    @classmethod
    def from_sources(cls, sources: dict[str, str]) -> ReasoningEngine:
        """Create engine from rule source strings (no filesystem).

        ``sources`` maps logical name (e.g. "routing") to Mangle source text.
        """
        import tempfile

        tmp = tempfile.mkdtemp()
        engine = cls.__new__(cls)
        engine._rules_dir = Path(tmp)
        engine._driver = None
        engine._query_contains = _QueryContainsPredicate()
        engine._rule_sources = dict(sources)
        return engine

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
        # Neo4j graph predicates for graph.mg rules
        registry.register("edge", _Neo4jEdgePredicate(self._driver))
        registry.register("mentioned_in", _Neo4jMentionedInPredicate(self._driver))
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

    def check_access(
        self, user: str, role: str, action: str, resource_type: str = "/public",
    ) -> bool:
        """Check if user with role is allowed to perform action on resource type.

        Evaluates the full RBAC model: role inheritance + permit + deny override.
        Returns True if allowed(user, action, resource_type) can be derived.
        """
        source = self._rule_sources.get("access")
        if source is None:
            return True  # No access rules = permit all

        user_fact = f'user_role("{_escape(user)}", {role}).\n'
        full_source = user_fact + source
        program = parse(full_source)
        store = eval_program(program)

        allowed_facts = store.get_by_predicate("allowed")
        # Mangle names like /read store value as "read" (without /), normalize for comparison
        norm_action = action.lstrip("/")
        norm_res = resource_type.lstrip("/")
        for fact in allowed_facts:
            if len(fact.args) >= 3:
                fact_user = fact.args[0].value if isinstance(fact.args[0], Constant) else None
                fact_action = str(fact.args[1].value).lstrip("/") if isinstance(fact.args[1], Constant) else None
                fact_res = str(fact.args[2].value).lstrip("/") if isinstance(fact.args[2], Constant) else None
                if fact_user == user and fact_action == norm_action and fact_res == norm_res:
                    return True

        return False

    def filter_results_by_access(
        self, results: list, user: str, role: str,
    ) -> list:
        """Filter retrieval results by access control rules.

        Each result's chunk may have a sensitivity level (public/sensitive/pii).
        Returns only results the user is allowed to read.
        """
        source = self._rule_sources.get("access")
        if source is None:
            return results  # No access rules = return all

        user_fact = f'user_role("{_escape(user)}", {role}).\n'
        full_source = user_fact + source
        program = parse(full_source)
        store = eval_program(program)

        allowed_facts = store.get_by_predicate("allowed")
        allowed_types: set[str] = set()
        for fact in allowed_facts:
            if len(fact.args) >= 3:
                fact_user = fact.args[0].value if isinstance(fact.args[0], Constant) else None
                fact_action = str(fact.args[1].value).lstrip("/") if isinstance(fact.args[1], Constant) else None
                fact_res = str(fact.args[2].value).lstrip("/") if isinstance(fact.args[2], Constant) else None
                if fact_user == user and fact_action == "read" and fact_res:
                    allowed_types.add(fact_res)

        if not allowed_types:
            return results  # No specific permissions derived = return all

        filtered = []
        for r in results:
            sensitivity = getattr(getattr(r, "chunk", None), "sensitivity", "/public")
            norm_sens = str(sensitivity).lstrip("/")
            if norm_sens in allowed_types:
                filtered.append(r)
        return filtered if filtered else results  # fallback to all if nothing passes

    def infer_connections(self, entity: str) -> list[dict]:
        """Derive connections for an entity using graph rules.

        Uses graph.mg rules to compute:
        - reachable(entity, target, depth) — transitive closure
        - common_neighbor(entity, other, neighbor) — shared neighbors
        - evidence(entity, passage_id) — passages mentioning the entity

        Returns list of dicts with type and relationship info.
        """
        source = self._rule_sources.get("graph")
        if source is None:
            return []

        registry = self._build_registry()

        try:
            program = parse(source)
            store = eval_program(program, externals=registry)
        except Exception as exc:
            logger.warning("Graph reasoning failed: %s", exc)
            return []

        results: list[dict] = []

        # Collect reachable entities
        for fact in store.get_by_predicate("reachable"):
            args = [a.value if isinstance(a, Constant) else str(a) for a in fact.args]
            if len(args) >= 2 and str(args[0]) == entity:
                results.append({
                    "type": "reachable",
                    "source": str(args[0]),
                    "target": str(args[1]),
                    "depth": args[2] if len(args) > 2 else 1,
                })

        # Collect common neighbors
        for fact in store.get_by_predicate("common_neighbor"):
            args = [a.value if isinstance(a, Constant) else str(a) for a in fact.args]
            if len(args) >= 3 and str(args[0]) == entity:
                results.append({
                    "type": "common_neighbor",
                    "entity_a": str(args[0]),
                    "entity_b": str(args[1]),
                    "neighbor": str(args[2]),
                })

        # Collect evidence passages
        for fact in store.get_by_predicate("evidence"):
            args = [a.value if isinstance(a, Constant) else str(a) for a in fact.args]
            if len(args) >= 2 and str(args[0]) == entity:
                results.append({
                    "type": "evidence",
                    "entity": str(args[0]),
                    "passage_id": str(args[1]),
                })

        return results

    def expand_entities_for_retrieval(self, entities: list[str]) -> list[str]:
        """Expand a set of entities using graph reasoning rules.

        Finds reachable and common-neighbor entities to augment retrieval.
        Returns the expanded set of entity names.
        """
        source = self._rule_sources.get("graph")
        if source is None or self._driver is None:
            return entities

        registry = self._build_registry()
        try:
            program = parse(source)
            store = eval_program(program, externals=registry)
        except Exception as exc:
            logger.warning("Entity expansion failed: %s", exc)
            return entities

        expanded = set(entities)
        entity_set = set(entities)

        for fact in store.get_by_predicate("reachable"):
            args = [a.value if isinstance(a, Constant) else str(a) for a in fact.args]
            if len(args) >= 2 and str(args[0]) in entity_set:
                expanded.add(str(args[1]))

        for fact in store.get_by_predicate("common_neighbor"):
            args = [a.value if isinstance(a, Constant) else str(a) for a in fact.args]
            if len(args) >= 3 and str(args[0]) in entity_set:
                expanded.add(str(args[1]))

        return list(expanded)


    @property
    def rule_sources(self) -> dict[str, str]:
        """Return loaded rule sources keyed by logical name."""
        return dict(self._rule_sources)

    def get_strata(self, source_name: str) -> list[list[str]]:
        """Return stratification for a rule source as list of predicate lists.

        Each inner list represents one stratum (evaluation layer).
        """
        source = self._rule_sources.get(source_name)
        if source is None:
            return []
        try:
            from pymangle.analysis import stratify

            program = parse(source)
            strata = stratify(program)
            return [sorted(s.predicates) for s in strata]
        except Exception as exc:
            logger.warning("Stratification failed for %s: %s", source_name, exc)
            return []


def _escape(s: str) -> str:
    """Escape string for embedding in Mangle source."""
    return s.replace("\\", "\\\\").replace('"', '\\"')
