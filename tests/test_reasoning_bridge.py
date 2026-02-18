"""Tests for Neo4j bridge and ReasoningEngine facade."""
from __future__ import annotations

from unittest.mock import MagicMock

from pymangle.ast_nodes import Constant, TermType
from pymangle.external import ExternalPredicateRegistry


class TestNeo4jExternalPredicate:
    def test_neo4j_external_predicate(self):
        """Mock driver, verify Cypher executed and results converted to Constants."""
        from agentic_graph_rag.reasoning.neo4j_bridge import Neo4jExternalPredicate

        # Mock neo4j driver & session
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock query result — simulate two records
        record1 = MagicMock()
        record1.values.return_value = ["alice", "knows", "bob"]
        record2 = MagicMock()
        record2.values.return_value = ["bob", "knows", "carol"]
        mock_session.run.return_value = [record1, record2]

        cypher = "MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name"
        pred = Neo4jExternalPredicate(mock_driver, cypher)

        results = list(pred.query([], []))
        assert len(results) == 2
        assert results[0] == [
            Constant("alice", TermType.STRING),
            Constant("knows", TermType.STRING),
            Constant("bob", TermType.STRING),
        ]
        mock_session.run.assert_called_once_with(cypher, {})

    def test_filter_pushdown(self):
        """Bound args passed to Cypher as positional parameters."""
        from agentic_graph_rag.reasoning.neo4j_bridge import Neo4jExternalPredicate

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = []

        cypher = "MATCH (a {name: $p0})-[r]->(b) RETURN a.name, type(r), b.name"
        pred = Neo4jExternalPredicate(mock_driver, cypher)

        inputs = [Constant("alice", TermType.STRING)]
        list(pred.query(inputs, []))

        mock_session.run.assert_called_once_with(cypher, {"p0": "alice"})

    def test_numeric_values_converted(self):
        """Numeric Neo4j values become NUMBER constants."""
        from agentic_graph_rag.reasoning.neo4j_bridge import Neo4jExternalPredicate

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        record = MagicMock()
        record.values.return_value = ["alice", 42]
        mock_session.run.return_value = [record]

        pred = Neo4jExternalPredicate(mock_driver, "MATCH (a) RETURN a.name, a.age")
        results = list(pred.query([], []))
        assert len(results) == 1
        assert results[0][1] == Constant(42, TermType.NUMBER)

    def test_registers_in_registry(self):
        """Neo4jExternalPredicate works with ExternalPredicateRegistry."""
        from agentic_graph_rag.reasoning.neo4j_bridge import Neo4jExternalPredicate

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        record = MagicMock()
        record.values.return_value = ["x"]
        mock_session.run.return_value = [record]

        pred = Neo4jExternalPredicate(mock_driver, "MATCH (n) RETURN n.name")
        registry = ExternalPredicateRegistry()
        registry.register("neo4j_nodes", pred)

        assert registry.has("neo4j_nodes")
        results = list(registry.query("neo4j_nodes", [], []))
        assert len(results) == 1


class TestReasoningEngine:
    def test_classify_query(self, tmp_path):
        """ReasoningEngine loads .mg files and classifies queries."""
        from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        (rules_dir / "routing.mg").write_text(
            'route_to(/vector_search, Q) :- query_contains(Q, "simple").\n'
            'route_to(/cypher_traverse, Q) :- query_contains(Q, "связ").\n'
        )

        engine = ReasoningEngine(str(rules_dir))
        result = engine.classify_query("простой simple запрос")
        assert result is not None
        assert result["tool"] == "vector_search"

    def test_classify_query_no_match(self, tmp_path):
        """No matching route returns None."""
        from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        (rules_dir / "routing.mg").write_text(
            'route_to(/vector_search, Q) :- query_contains(Q, "specific_keyword").\n'
        )

        engine = ReasoningEngine(str(rules_dir))
        result = engine.classify_query("unrelated query")
        assert result is None

    def test_check_access(self, tmp_path):
        """ReasoningEngine checks access control rules with full RBAC."""
        from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        (rules_dir / "access.mg").write_text(
            'role_inherits(/admin, /viewer).\n'
            'has_role(User, Role) :- user_role(User, Role).\n'
            'has_role(User, Parent) :- has_role(User, Child), role_inherits(Child, Parent).\n'
            'permit(/viewer, /read, /public).\n'
            'permit(/admin, /read, /sensitive).\n'
            'allowed(User, Action, ResType) :- has_role(User, Role), permit(Role, Action, ResType).\n'
        )

        engine = ReasoningEngine(str(rules_dir))
        # Admin can read sensitive
        assert engine.check_access("alice", "/admin", "/read", "/sensitive") is True
        # Admin inherits viewer permissions (public)
        assert engine.check_access("alice", "/admin", "/read", "/public") is True
        # Viewer can read public
        assert engine.check_access("bob", "/viewer", "/read", "/public") is True
        # Viewer cannot read sensitive
        assert engine.check_access("bob", "/viewer", "/read", "/sensitive") is False
