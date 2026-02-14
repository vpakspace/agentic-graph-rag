"""Tests for agentic_graph_rag.indexing.skeleton."""

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
from rag_core.models import Chunk, Entity

from agentic_graph_rag.indexing.skeleton import (
    _parse_extraction_response,
    build_knn_graph,
    build_skeleton_index,
    compute_pagerank,
    extract_entities_full,
    extract_keywords,
    link_peripheral_keywords,
    select_skeletal_chunks,
)


def _make_chunks(n: int) -> list[Chunk]:
    return [Chunk(id=f"c{i}", content=f"Content of chunk {i}") for i in range(n)]


def _make_embeddings(n: int, dim: int = 4) -> list[list[float]]:
    """Generate deterministic embeddings for testing."""
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((n, dim))
    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.where(norms == 0, 1.0, norms)
    return normed.tolist()


# ---------------------------------------------------------------------------
# build_knn_graph
# ---------------------------------------------------------------------------

class TestBuildKnnGraph:
    def test_empty_chunks(self):
        g = build_knn_graph([], [], k=3)
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_single_chunk(self):
        chunks = _make_chunks(1)
        embs = _make_embeddings(1)
        g = build_knn_graph(chunks, embs, k=3)
        assert g.number_of_nodes() == 1
        assert g.number_of_edges() == 0  # no neighbours

    def test_two_chunks(self):
        chunks = _make_chunks(2)
        embs = _make_embeddings(2)
        g = build_knn_graph(chunks, embs, k=5)
        assert g.number_of_nodes() == 2
        # k=5 but only 1 neighbour available → 1 edge per node = 2 edges
        assert g.number_of_edges() == 2

    def test_five_chunks_k3(self):
        chunks = _make_chunks(5)
        embs = _make_embeddings(5)
        g = build_knn_graph(chunks, embs, k=3)
        assert g.number_of_nodes() == 5
        # Each node connects to 3 neighbours → at most 15 edges
        assert g.number_of_edges() <= 15
        assert g.number_of_edges() >= 5  # at least some edges

    def test_edges_have_weight(self):
        chunks = _make_chunks(3)
        embs = _make_embeddings(3)
        g = build_knn_graph(chunks, embs, k=2)
        for _, _, data in g.edges(data=True):
            assert "weight" in data
            assert -1.0 <= data["weight"] <= 1.0

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    def test_uses_settings_k(self, mock_settings):
        cfg = MagicMock()
        cfg.indexing.knn_k = 2
        mock_settings.return_value = cfg

        chunks = _make_chunks(5)
        embs = _make_embeddings(5)
        g = build_knn_graph(chunks, embs)
        # k=2 → each node gets 2 edges → 10 total max
        assert g.number_of_edges() <= 10


# ---------------------------------------------------------------------------
# compute_pagerank
# ---------------------------------------------------------------------------

class TestComputePagerank:
    def test_empty_graph(self):
        g = nx.DiGraph()
        scores = compute_pagerank(g, damping=0.85)
        assert scores == {}

    def test_simple_graph(self):
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])
        scores = compute_pagerank(g, damping=0.85)
        assert len(scores) == 3
        assert all(0 < s < 1 for s in scores.values())
        # Sum should be ~1.0
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_star_graph_center_highest(self):
        """Center of star should have highest PageRank."""
        g = nx.DiGraph()
        for i in range(1, 6):
            g.add_edge(i, 0)  # all point to center
        scores = compute_pagerank(g, damping=0.85)
        assert scores[0] == max(scores.values())

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    def test_uses_settings_damping(self, mock_settings):
        cfg = MagicMock()
        cfg.indexing.pagerank_damping = 0.5
        mock_settings.return_value = cfg

        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 0)])
        scores = compute_pagerank(g)
        assert len(scores) == 2


# ---------------------------------------------------------------------------
# select_skeletal_chunks
# ---------------------------------------------------------------------------

class TestSelectSkeletalChunks:
    def test_empty(self):
        skeletal, peripheral = select_skeletal_chunks([], {}, beta=0.25)
        assert skeletal == []
        assert peripheral == []

    def test_selects_top_beta(self):
        chunks = _make_chunks(10)
        scores = {i: float(i) for i in range(10)}  # 9 is highest
        skeletal, peripheral = select_skeletal_chunks(chunks, scores, beta=0.3)
        # beta=0.3 → 3 chunks
        assert len(skeletal) == 3
        assert len(peripheral) == 7
        # Highest-scored chunks should be skeletal
        assert chunks[9] in skeletal
        assert chunks[8] in skeletal
        assert chunks[7] in skeletal

    def test_at_least_one_skeletal(self):
        chunks = _make_chunks(2)
        scores = {0: 0.5, 1: 0.3}
        skeletal, peripheral = select_skeletal_chunks(chunks, scores, beta=0.1)
        assert len(skeletal) >= 1  # min 1

    @patch("agentic_graph_rag.indexing.skeleton.get_settings")
    def test_uses_settings_beta(self, mock_settings):
        cfg = MagicMock()
        cfg.indexing.skeleton_beta = 0.5
        mock_settings.return_value = cfg

        chunks = _make_chunks(10)
        scores = {i: float(i) for i in range(10)}
        skeletal, peripheral = select_skeletal_chunks(chunks, scores)
        assert len(skeletal) == 5


# ---------------------------------------------------------------------------
# _parse_extraction_response
# ---------------------------------------------------------------------------

class TestParseExtractionResponse:
    def test_parses_entities(self):
        text = "ENTITY: COVID-19 | Disease | Respiratory illness\nENTITY: WHO | Organization | World Health"
        entities, rels = _parse_extraction_response(text, "c1")
        assert len(entities) == 2
        assert entities[0].name == "COVID-19"
        assert entities[0].entity_type == "Disease"
        assert entities[0].description == "Respiratory illness"

    def test_parses_relationships(self):
        text = "RELATIONSHIP: COVID-19 | caused_by | SARS-CoV-2"
        entities, rels = _parse_extraction_response(text, "c1")
        assert len(rels) == 1
        assert rels[0].source == "COVID-19"
        assert rels[0].target == "SARS-CoV-2"
        assert rels[0].relation_type == "caused_by"

    def test_mixed_output(self):
        text = (
            "ENTITY: Python | Language | Programming language\n"
            "RELATIONSHIP: Python | used_for | Machine Learning\n"
            "ENTITY: ML | Field | Machine Learning\n"
        )
        entities, rels = _parse_extraction_response(text, "c1")
        assert len(entities) == 2
        assert len(rels) == 1

    def test_malformed_lines_skipped(self):
        text = "ENTITY: lonely\nGARBAGE LINE\nENTITY: A | B | C"
        entities, rels = _parse_extraction_response(text, "c1")
        # "lonely" has only 1 part → skipped (need ≥2)
        assert len(entities) == 1
        assert entities[0].name == "A"

    def test_empty_text(self):
        entities, rels = _parse_extraction_response("", "c1")
        assert entities == []
        assert rels == []


# ---------------------------------------------------------------------------
# extract_entities_full
# ---------------------------------------------------------------------------

class TestExtractEntitiesFull:
    def test_empty_chunks(self):
        entities, rels = extract_entities_full([])
        assert entities == []
        assert rels == []

    def test_extracts_from_llm(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = (
            "ENTITY: Machine Learning | Field | AI subfield\n"
            "RELATIONSHIP: ML | part_of | AI"
        )
        client.chat.completions.create.return_value = resp

        chunks = [Chunk(id="c1", content="Machine Learning is part of AI")]
        entities, rels = extract_entities_full(chunks, openai_client=client)

        assert len(entities) == 1
        assert len(rels) == 1
        client.chat.completions.create.assert_called_once()

    def test_handles_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")

        chunks = [Chunk(id="c1", content="test")]
        entities, rels = extract_entities_full(chunks, openai_client=client)
        assert entities == []
        assert rels == []


# ---------------------------------------------------------------------------
# link_peripheral_keywords
# ---------------------------------------------------------------------------

class TestLinkPeripheralKeywords:
    def test_empty(self):
        assert link_peripheral_keywords([], []) == []

    def test_links_matching_entities(self):
        entities = [Entity(id="e1", name="Python", entity_type="Language")]
        chunks = [
            Chunk(id="c1", content="Python is great"),
            Chunk(id="c2", content="Java is also good"),
        ]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 1
        assert rels[0].source == "Python"
        assert rels[0].target == "c1"
        assert rels[0].relation_type == "MENTIONED_IN"

    def test_case_insensitive(self):
        entities = [Entity(id="e1", name="PYTHON")]
        chunks = [Chunk(id="c1", content="python is great")]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 1

    def test_skips_short_names(self):
        entities = [Entity(id="e1", name="A")]
        chunks = [Chunk(id="c1", content="A is a letter")]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 0  # "A" too short (<2 chars)

    def test_multiple_matches(self):
        entities = [
            Entity(id="e1", name="Python"),
            Entity(id="e2", name="ML"),
        ]
        chunks = [Chunk(id="c1", content="Python for ML is popular")]
        rels = link_peripheral_keywords(chunks, entities)
        assert len(rels) == 2


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_basic(self):
        keywords = extract_keywords("Python machine learning is very good for data")
        assert "python" in keywords
        assert "machine" in keywords

    def test_stop_words_removed(self):
        keywords = extract_keywords("the quick brown fox is very fast")
        assert "the" not in keywords
        assert "quick" in keywords

    def test_max_keywords(self):
        words = ["alpha", "bravo", "charlie", "delta", "echo",
                 "foxtrot", "golf", "hotel", "india", "juliet"]
        text = " ".join(words * 2)  # repeat so all have freq
        keywords = extract_keywords(text, max_keywords=5)
        assert len(keywords) == 5

    def test_empty(self):
        assert extract_keywords("") == []


# ---------------------------------------------------------------------------
# build_skeleton_index (orchestrator)
# ---------------------------------------------------------------------------

class TestBuildSkeletonIndex:
    def test_empty(self):
        entities, rels, skel, peri = build_skeleton_index([], [])
        assert entities == []
        assert rels == []
        assert skel == []
        assert peri == []

    def test_full_pipeline(self):
        """Integration test with mock LLM."""
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ENTITY: TestEntity | Concept | A test"
        client.chat.completions.create.return_value = resp

        chunks = _make_chunks(8)
        embs = _make_embeddings(8)

        entities, rels, skeletal, peripheral = build_skeleton_index(
            chunks, embs, openai_client=client
        )

        # beta=0.25 → 2 skeletal chunks (8 * 0.25 = 2)
        assert len(skeletal) == 2
        assert len(peripheral) == 6
        assert len(entities) >= 1  # at least something extracted
        # LLM called for each skeletal chunk
        assert client.chat.completions.create.call_count == 2
