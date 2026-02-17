"""Tests for agentic_graph_rag.retrieval.vector_cypher."""

from unittest.mock import MagicMock, patch

from rag_core.models import GraphContext

from agentic_graph_rag.retrieval.vector_cypher import (
    collect_context,
    find_entry_points,
    search,
    traverse_graph,
)


def _mock_driver() -> MagicMock:
    """Create mock Neo4j driver with session context manager."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


def _mock_records(data: list[dict]) -> MagicMock:
    """Create mock Neo4j result that iterates over records."""
    result = MagicMock()
    records = []
    for d in data:
        rec = MagicMock()
        rec.__getitem__ = lambda self, key, _d=d: _d[key]
        rec.get = lambda key, default=None, _d=d: _d.get(key, default)
        records.append(rec)
    result.__iter__ = MagicMock(return_value=iter(records))
    return result


# ---------------------------------------------------------------------------
# find_entry_points
# ---------------------------------------------------------------------------

class TestFindEntryPoints:
    def test_returns_entries(self):
        driver = _mock_driver()
        session = driver.session().__enter__()

        session.run.return_value = _mock_records([
            {"id": "e1", "name": "Python", "entity_type": "Language",
             "pagerank_score": 0.5, "score": 0.92},
            {"id": "e2", "name": "ML", "entity_type": "Field",
             "pagerank_score": 0.3, "score": 0.85},
        ])

        entries = find_entry_points([1.0, 0.0], driver, top_k=5, threshold=0.7)

        assert len(entries) == 2
        assert entries[0]["id"] == "e1"
        assert entries[0]["name"] == "Python"
        assert entries[0]["score"] == 0.92
        assert entries[1]["id"] == "e2"

    def test_empty_results(self):
        driver = _mock_driver()
        session = driver.session().__enter__()
        session.run.return_value = _mock_records([])

        entries = find_entry_points([1.0, 0.0], driver, top_k=5, threshold=0.7)
        assert entries == []

    def test_handles_none_fields(self):
        driver = _mock_driver()
        session = driver.session().__enter__()

        session.run.return_value = _mock_records([
            {"id": "e1", "name": None, "entity_type": None,
             "pagerank_score": None, "score": 0.8},
        ])

        entries = find_entry_points([1.0], driver, top_k=5, threshold=0.5)
        assert len(entries) == 1
        assert entries[0]["name"] == ""
        assert entries[0]["entity_type"] == ""
        assert entries[0]["pagerank_score"] == 0.0

    @patch("agentic_graph_rag.retrieval.vector_cypher.get_settings")
    def test_uses_settings_defaults(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.top_k_vector = 3
        cfg.retrieval.vector_threshold = 0.8
        mock_settings.return_value = cfg

        driver = _mock_driver()
        session = driver.session().__enter__()
        session.run.return_value = _mock_records([])

        find_entry_points([1.0], driver)
        # Verify run was called (settings used internally)
        session.run.assert_called_once()


# ---------------------------------------------------------------------------
# traverse_graph
# ---------------------------------------------------------------------------

class TestTraverseGraph:
    def test_empty_entry_ids(self):
        driver = _mock_driver()
        result = traverse_graph([], driver, max_hops=2)
        assert result["phrase_nodes"] == []
        assert result["passage_nodes"] == []
        assert result["relationships"] == []

    def test_traversal_uses_related_to_for_step1(self):
        """Verify step 1 Cypher uses RELATED_TO edges."""
        driver = _mock_driver()
        session = driver.session().__enter__()

        session.run.side_effect = [
            _mock_records([]),  # step 1: traversal via RELATED_TO
            MagicMock(single=MagicMock(return_value=None)),  # entry lookup
            _mock_records([]),  # step 2: co-occurrence expansion
            _mock_records([]),  # step 3: passage lookup
        ]

        traverse_graph(["e1"], driver, max_hops=2)

        # First call should be the RELATED_TO traversal query
        first_call = session.run.call_args_list[0]
        cypher = first_call[0][0]
        assert "RELATED_TO" in cypher

    def test_traversal_with_results(self):
        driver = _mock_driver()
        session = driver.session().__enter__()

        # Run 1: phrase traversal via RELATED_TO
        traversal_records = _mock_records([
            {
                "connected_id": "e2", "connected_name": "ML",
                "connected_type": "Field",
                "src_id": "e1", "src_name": "Python",
                "rel_type": "USED_FOR",
                "tgt_id": "e2", "tgt_name": "ML",
            },
        ])

        # Run 2: entry node lookup
        entry_record = MagicMock()
        entry_record.__getitem__ = lambda self, key: {"id": "e1", "name": "Python", "entity_type": "Language"}[key]
        entry_result = MagicMock()
        entry_result.single.return_value = entry_record

        # Run 3: co-occurrence expansion via MENTIONED_IN
        cooccur_records = _mock_records([])

        # Run 4: passage lookup
        passage_records = _mock_records([
            {"id": "p1", "text": "Python is used for ML", "chunk_id": "c1"},
        ])

        session.run.side_effect = [traversal_records, entry_result, cooccur_records, passage_records]

        result = traverse_graph(["e1"], driver, max_hops=2)

        assert len(result["phrase_nodes"]) == 2  # e1 + e2
        assert len(result["passage_nodes"]) == 1
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["relation"] == "USED_FOR"

    def test_cooccurrence_expansion_finds_neighbors(self):
        """Verify co-occurrence step discovers PhraseNodes sharing a PassageNode."""
        driver = _mock_driver()
        session = driver.session().__enter__()

        # Step 1: RELATED_TO traversal — no connections
        traversal_records = _mock_records([])

        # Entry node lookup
        entry_record = MagicMock()
        entry_record.__getitem__ = lambda self, key: {
            "id": "e1", "name": "Python", "entity_type": "Language",
        }[key]
        entry_result = MagicMock()
        entry_result.single.return_value = entry_record

        # Step 2: co-occurrence — finds neighbor via shared passage
        cooccur_records = _mock_records([
            {"id": "e3", "name": "Django", "entity_type": "Framework"},
        ])

        # Step 3: passage lookup — passages from both e1 and e3
        passage_records = _mock_records([
            {"id": "p1", "text": "Python uses Django", "chunk_id": "c1"},
            {"id": "p2", "text": "Django is a framework", "chunk_id": "c2"},
        ])

        session.run.side_effect = [
            traversal_records, entry_result, cooccur_records, passage_records,
        ]

        result = traverse_graph(["e1"], driver, max_hops=2)

        # e1 (entry) + e3 (co-occurrence neighbor)
        assert len(result["phrase_nodes"]) == 2
        names = {p["name"] for p in result["phrase_nodes"]}
        assert "Python" in names
        assert "Django" in names
        # Both passages collected
        assert len(result["passage_nodes"]) == 2

    @patch("agentic_graph_rag.retrieval.vector_cypher.get_settings")
    def test_uses_settings_max_hops(self, mock_settings):
        cfg = MagicMock()
        cfg.retrieval.max_hops = 3
        mock_settings.return_value = cfg

        driver = _mock_driver()
        session = driver.session().__enter__()

        # Entry node lookup returns None
        entry_result = MagicMock()
        entry_result.single.return_value = None
        session.run.side_effect = [
            _mock_records([]),   # step 1: traversal via RELATED_TO
            entry_result,        # entry lookup
            _mock_records([]),   # step 2: co-occurrence expansion
            _mock_records([]),   # step 3: passage lookup
        ]

        traverse_graph(["e1"], driver)
        assert session.run.call_count >= 1


# ---------------------------------------------------------------------------
# collect_context
# ---------------------------------------------------------------------------

class TestCollectContext:
    def test_empty_traversal(self):
        ctx = collect_context({
            "phrase_nodes": [],
            "passage_nodes": [],
            "relationships": [],
        })
        assert isinstance(ctx, GraphContext)
        assert ctx.triplets == []
        assert ctx.passages == []
        assert ctx.entities == []
        assert ctx.source_ids == []

    def test_assembles_triplets(self):
        ctx = collect_context({
            "phrase_nodes": [
                {"id": "e1", "name": "Python", "entity_type": "Language"},
            ],
            "passage_nodes": [
                {"id": "p1", "text": "Python is great", "chunk_id": "c1"},
            ],
            "relationships": [
                {"source": "Python", "relation": "USED_FOR", "target": "ML"},
            ],
        })

        assert len(ctx.triplets) == 1
        assert ctx.triplets[0]["source"] == "Python"
        assert ctx.triplets[0]["relation"] == "USED_FOR"
        assert ctx.triplets[0]["target"] == "ML"
        assert len(ctx.passages) == 1
        assert "Python is great" in ctx.passages
        assert len(ctx.entities) == 1
        assert ctx.entities[0].name == "Python"
        assert ctx.source_ids == ["c1"]

    def test_deduplicates_triplets(self):
        ctx = collect_context({
            "phrase_nodes": [],
            "passage_nodes": [],
            "relationships": [
                {"source": "A", "relation": "R", "target": "B"},
                {"source": "A", "relation": "R", "target": "B"},  # duplicate
                {"source": "C", "relation": "R", "target": "D"},
            ],
        })
        assert len(ctx.triplets) == 2

    def test_filters_empty_passages(self):
        ctx = collect_context({
            "phrase_nodes": [],
            "passage_nodes": [
                {"id": "p1", "text": "", "chunk_id": "c1"},
                {"id": "p2", "text": "Has content", "chunk_id": "c2"},
            ],
            "relationships": [],
        })
        assert len(ctx.passages) == 1
        assert ctx.passages[0] == "Has content"

    def test_filters_empty_chunk_ids(self):
        ctx = collect_context({
            "phrase_nodes": [],
            "passage_nodes": [
                {"id": "p1", "text": "text", "chunk_id": ""},
                {"id": "p2", "text": "text2", "chunk_id": "c2"},
            ],
            "relationships": [],
        })
        assert ctx.source_ids == ["c2"]

    def test_missing_keys(self):
        ctx = collect_context({})
        assert ctx.triplets == []
        assert ctx.passages == []


# ---------------------------------------------------------------------------
# search (full pipeline)
# ---------------------------------------------------------------------------

class TestSearch:
    def test_no_entry_points(self):
        driver = _mock_driver()
        session = driver.session().__enter__()
        session.run.return_value = _mock_records([])

        ctx = search([1.0, 0.0], driver, top_k=5, threshold=0.7)
        assert isinstance(ctx, GraphContext)
        assert ctx.triplets == []
        assert ctx.passages == []

    def test_full_pipeline(self):
        driver = _mock_driver()
        session = driver.session().__enter__()

        # Call 1: find_entry_points
        entry_records = _mock_records([
            {"id": "e1", "name": "Python", "entity_type": "Language",
             "pagerank_score": 0.5, "score": 0.9},
        ])

        # Call 2: traverse - phrase relationships via RELATED_TO
        traversal_records = _mock_records([])

        # Call 3: entry node lookup
        entry_node = MagicMock()
        entry_node.__getitem__ = lambda self, key: {"id": "e1", "name": "Python", "entity_type": "Language"}[key]
        entry_result = MagicMock()
        entry_result.single.return_value = entry_node

        # Call 4: co-occurrence expansion
        cooccur_records = _mock_records([])

        # Call 5: passage lookup
        passage_records = _mock_records([
            {"id": "p1", "text": "Python content", "chunk_id": "c1"},
        ])

        session.run.side_effect = [
            entry_records,
            traversal_records,
            entry_result,
            cooccur_records,
            passage_records,
        ]

        ctx = search([1.0, 0.0], driver, top_k=5, max_hops=2, threshold=0.5)

        assert isinstance(ctx, GraphContext)
        assert len(ctx.passages) == 1
        assert "Python content" in ctx.passages[0]
        assert len(ctx.entities) == 1
        assert ctx.entities[0].name == "Python"
