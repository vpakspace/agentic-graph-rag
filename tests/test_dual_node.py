"""Tests for agentic_graph_rag.indexing.dual_node."""

from unittest.mock import MagicMock, patch

import networkx as nx
from rag_core.models import Chunk, Entity, PassageNode, PhraseNode, Relationship

from agentic_graph_rag.indexing.dual_node import (
    build_dual_graph,
    compute_ppr,
    create_passage_nodes,
    create_phrase_nodes,
    create_phrase_relationships,
    link_entities_to_passages,
    link_phrase_to_passage,
)


def _mock_driver() -> MagicMock:
    """Create mock Neo4j driver with session context manager."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


# ---------------------------------------------------------------------------
# create_phrase_nodes
# ---------------------------------------------------------------------------

class TestCreatePhraseNodes:
    def test_empty(self):
        driver = _mock_driver()
        result = create_phrase_nodes([], driver)
        assert result == []

    def test_creates_nodes(self):
        driver = _mock_driver()
        entities = [
            Entity(id="e1", name="Python", entity_type="Language"),
            Entity(id="e2", name="ML", entity_type="Field"),
        ]
        result = create_phrase_nodes(entities, driver)

        assert len(result) == 2
        assert isinstance(result[0], PhraseNode)
        assert result[0].id == "e1"
        assert result[0].name == "Python"
        assert result[1].name == "ML"

    def test_applies_pagerank_scores(self):
        driver = _mock_driver()
        entities = [Entity(id="e1", name="Python")]
        scores = {"e1": 0.75}

        result = create_phrase_nodes(entities, driver, pagerank_scores=scores)
        assert result[0].pagerank_score == 0.75

    def test_generates_id_from_name(self):
        driver = _mock_driver()
        entities = [Entity(name="TestEntity")]  # no id
        result = create_phrase_nodes(entities, driver)
        assert result[0].id != ""
        assert len(result[0].id) == 8  # md5[:8]


# ---------------------------------------------------------------------------
# create_passage_nodes
# ---------------------------------------------------------------------------

class TestCreatePassageNodes:
    def test_empty(self):
        driver = _mock_driver()
        result = create_passage_nodes([], driver)
        assert result == []

    def test_creates_nodes(self):
        driver = _mock_driver()
        chunks = [
            Chunk(id="c1", content="Content one"),
            Chunk(id="c2", content="Content two", context="Extra context"),
        ]
        result = create_passage_nodes(chunks, driver)

        assert len(result) == 2
        assert isinstance(result[0], PassageNode)
        assert result[0].id == "c1"
        assert result[0].text == "Content one"
        assert result[1].text == "Extra context\n\nContent two"  # enriched_content

    def test_generates_id_from_content(self):
        driver = _mock_driver()
        chunks = [Chunk(content="Some text")]  # no id
        result = create_passage_nodes(chunks, driver)
        assert result[0].id != ""


# ---------------------------------------------------------------------------
# link_phrase_to_passage
# ---------------------------------------------------------------------------

class TestLinkPhraseToPassage:
    def test_creates_relationship(self):
        driver = _mock_driver()
        link_phrase_to_passage("ph1", "pa1", driver)

        session = driver.session().__enter__()
        session.run.assert_called_once()
        call_args = session.run.call_args
        assert "MENTIONED_IN" in call_args[0][0]
        assert call_args[1]["phrase_id"] == "ph1"
        assert call_args[1]["passage_id"] == "pa1"


# ---------------------------------------------------------------------------
# link_entities_to_passages
# ---------------------------------------------------------------------------

class TestLinkEntitiesToPassages:
    def test_empty(self):
        driver = _mock_driver()
        assert link_entities_to_passages([], [], driver) == 0

    def test_links_matching(self):
        driver = _mock_driver()
        entities = [Entity(id="e1", name="Python")]
        chunks = [
            Chunk(id="c1", content="Python is great"),
            Chunk(id="c2", content="Java is ok"),
        ]
        count = link_entities_to_passages(entities, chunks, driver)
        assert count == 1  # only c1 matches

    def test_case_insensitive(self):
        driver = _mock_driver()
        entities = [Entity(id="e1", name="PYTHON")]
        chunks = [Chunk(id="c1", content="python rocks")]
        count = link_entities_to_passages(entities, chunks, driver)
        assert count == 1

    def test_skips_short_names(self):
        driver = _mock_driver()
        entities = [Entity(id="e1", name="A")]
        chunks = [Chunk(id="c1", content="A is here")]
        count = link_entities_to_passages(entities, chunks, driver)
        assert count == 0


# ---------------------------------------------------------------------------
# compute_ppr
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# create_phrase_relationships
# ---------------------------------------------------------------------------

class TestCreatePhraseRelationships:
    def test_empty(self):
        driver = _mock_driver()
        assert create_phrase_relationships([], driver) == 0

    def test_creates_edges(self):
        driver = _mock_driver()
        session = driver.session().__enter__()

        # Mock the result of MERGE query
        rec = MagicMock()
        rec.__getitem__ = lambda self, key: 1 if key == "cnt" else None
        result = MagicMock()
        result.single.return_value = rec
        session.run.return_value = result

        rels = [
            Relationship(source="Python", target="ML", relation_type="USED_FOR"),
            Relationship(source="Neo4j", target="Graph", relation_type="STORES"),
        ]
        count = create_phrase_relationships(rels, driver)
        assert count == 2
        assert session.run.call_count == 2

    def test_skips_self_references(self):
        driver = _mock_driver()
        session = driver.session().__enter__()

        rels = [
            Relationship(source="Python", target="python", relation_type="SAME"),
        ]
        count = create_phrase_relationships(rels, driver)
        assert count == 0
        session.run.assert_not_called()

    def test_skips_empty_names(self):
        driver = _mock_driver()
        driver.session().__enter__()

        rels = [
            Relationship(source="", target="ML", relation_type="R"),
            Relationship(source="Python", target="", relation_type="R"),
        ]
        count = create_phrase_relationships(rels, driver)
        assert count == 0

    def test_cypher_contains_related_to(self):
        driver = _mock_driver()
        session = driver.session().__enter__()

        rec = MagicMock()
        rec.__getitem__ = lambda self, key: 1 if key == "cnt" else None
        result = MagicMock()
        result.single.return_value = rec
        session.run.return_value = result

        rels = [Relationship(source="A", target="B", relation_type="REL")]
        create_phrase_relationships(rels, driver)

        call_args = session.run.call_args
        assert "RELATED_TO" in call_args[0][0]
        assert call_args[1]["src"] == "A"
        assert call_args[1]["tgt"] == "B"
        assert call_args[1]["rel_type"] == "REL"


class TestComputePPR:
    def test_empty_graph(self):
        g = nx.Graph()
        scores = compute_ppr(g, [0], alpha=0.15)
        assert scores == {}

    def test_empty_query_nodes(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2)])
        scores = compute_ppr(g, [], alpha=0.15)
        assert scores == {}

    def test_invalid_query_nodes(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1)])
        scores = compute_ppr(g, [99], alpha=0.15)  # 99 not in graph
        assert scores == {}

    def test_simple_ppr(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        scores = compute_ppr(g, [0], alpha=0.15)
        assert len(scores) == 4
        # Node 0 (query node) should have highest score
        assert scores[0] == max(scores.values())
        # Sum should be ~1.0
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_multiple_query_nodes(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        scores = compute_ppr(g, [0, 4], alpha=0.15)
        assert len(scores) == 5
        # Endpoints should have higher scores than middle
        assert scores[0] > scores[2]
        assert scores[4] > scores[2]

    @patch("agentic_graph_rag.indexing.dual_node.get_settings")
    def test_uses_settings_alpha(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.ppr_alpha = 0.3
        mock_settings.return_value = cfg

        g = nx.Graph()
        g.add_edges_from([(0, 1)])
        scores = compute_ppr(g, [0])
        assert len(scores) == 2

    def test_directed_graph(self):
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2)])
        scores = compute_ppr(g, [0], alpha=0.15)
        assert len(scores) == 3


# ---------------------------------------------------------------------------
# build_dual_graph (orchestrator)
# ---------------------------------------------------------------------------

class TestBuildDualGraph:
    def test_empty(self):
        driver = _mock_driver()
        phrases, passages, links = build_dual_graph([], [], driver)
        assert phrases == []
        assert passages == []
        assert links == 0

    def test_full_pipeline(self):
        driver = _mock_driver()
        entities = [
            Entity(id="e1", name="Python", entity_type="Language"),
            Entity(id="e2", name="ML", entity_type="Field"),
        ]
        chunks = [
            Chunk(id="c1", content="Python for ML is great"),
            Chunk(id="c2", content="Java is also useful"),
        ]

        phrases, passages, links = build_dual_graph(entities, chunks, driver)

        assert len(phrases) == 2
        assert len(passages) == 2
        # "Python" in c1, "ML" in c1 → 2 links
        assert links == 2

    def test_passes_relationships(self):
        driver = _mock_driver()
        session = driver.session().__enter__()

        # Mock for create_phrase_relationships MERGE query
        rec = MagicMock()
        rec.__getitem__ = lambda self, key: 1 if key == "cnt" else None
        result = MagicMock()
        result.single.return_value = rec
        session.run.return_value = result

        entities = [Entity(id="e1", name="A")]
        chunks = [Chunk(id="c1", content="A is here")]
        rels = [Relationship(source="A", target="B", relation_type="REL")]

        build_dual_graph(entities, chunks, driver, relationships=rels)

        # Verify RELATED_TO was attempted (among other calls)
        cypher_calls = [
            str(c) for c in session.run.call_args_list
        ]
        assert any("RELATED_TO" in c for c in cypher_calls)
