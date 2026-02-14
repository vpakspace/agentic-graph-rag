"""Tests for rag_core.kg_client."""

import pytest

from rag_core.kg_client import (
    DEFAULT_MAX_EPISODE_CHARS,
    KGClient,
    KGFact,
    sanitize_lucene,
    split_episodes,
)


class TestSanitizeLucene:
    def test_removes_special_chars(self):
        assert sanitize_lucene("hello/world") == "hello world"
        assert sanitize_lucene("a*b?c") == "a b c"
        assert sanitize_lucene("term1 AND term2") == "term1 AND term2"

    def test_preserves_normal_text(self):
        assert sanitize_lucene("normal text") == "normal text"

    def test_removes_brackets(self):
        result = sanitize_lucene("[array]{dict}")
        assert "[" not in result
        assert "{" not in result


class TestSplitEpisodes:
    def test_short_text_no_split(self):
        result = split_episodes("short text", max_chars=1000)
        assert result == ["short text"]

    def test_splits_at_paragraphs(self):
        text = ("A" * 5000) + "\n\n" + ("B" * 5000)
        result = split_episodes(text, max_chars=6000, overlap=0)
        assert len(result) >= 2

    def test_overlap(self):
        text = ("A" * 100) + "\n\n" + ("B" * 100)
        result = split_episodes(text, max_chars=120, overlap=20)
        assert len(result) >= 2
        # Second episode should contain overlap from first
        assert result[1].startswith("A" * 20) or len(result[1]) > 100

    def test_default_max_chars(self):
        assert DEFAULT_MAX_EPISODE_CHARS == 8_000

    def test_force_split_large_paragraph(self):
        text = "X" * 20000
        result = split_episodes(text, max_chars=5000, overlap=0)
        assert len(result) >= 4


class TestKGFact:
    def test_creation(self):
        fact = KGFact(content="Neo4j is a graph database")
        assert fact.content == "Neo4j is a graph database"
        assert fact.source == "kg"
        assert fact.score == 0.8

    def test_with_temporal(self):
        fact = KGFact(
            content="Event happened",
            valid_at="2025-01-01",
            entity_name="Neo4j",
        )
        assert fact.valid_at == "2025-01-01"
        assert fact.entity_name == "Neo4j"


class TestKGClientInit:
    def test_initial_state(self):
        client = KGClient()
        assert client._graphiti is None
        assert client._driver is None


class TestKGClientNotConnected:
    @pytest.mark.asyncio
    async def test_ingest_without_connect_raises(self):
        client = KGClient()
        with pytest.raises(RuntimeError, match="not connected"):
            await client.ingest_text("test")

    @pytest.mark.asyncio
    async def test_search_without_connect_raises(self):
        client = KGClient()
        with pytest.raises(RuntimeError, match="not connected"):
            await client.search("query")

    def test_temporal_query_without_connect_raises(self):
        client = KGClient()
        with pytest.raises(RuntimeError, match="not connected"):
            client.temporal_query()


class TestKGClientDisconnected:
    def test_entity_count_no_driver(self):
        client = KGClient()
        assert client.entity_count() == 0

    def test_get_entities_no_driver(self):
        client = KGClient()
        assert client.get_entities() == []

    def test_get_relationships_no_driver(self):
        client = KGClient()
        assert client.get_relationships() == []

    @pytest.mark.asyncio
    async def test_close_without_connect(self):
        client = KGClient()
        await client.close()  # Should not raise
