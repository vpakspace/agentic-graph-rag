"""Tests for rag_core.chunker."""

from rag_core.chunker import (
    MAX_EPISODE_CHARS,
    chunk_text,
    sanitize_for_graphiti,
    split_large_content,
)


class TestSanitizeForGraphiti:
    def test_removes_lucene_chars(self):
        assert sanitize_for_graphiti("hello/world") == "hello world"
        assert sanitize_for_graphiti("a*b?c") == "a b c"

    def test_preserves_normal_text(self):
        assert sanitize_for_graphiti("hello world") == "hello world"

    def test_removes_brackets(self):
        result = sanitize_for_graphiti("array[0]{key}")
        assert "[" not in result
        assert "{" not in result


class TestSplitLargeContent:
    def test_short_text_no_split(self):
        parts = split_large_content("short", "src")
        assert parts == [("short", "src")]

    def test_splits_at_paragraphs(self):
        text = ("A" * 5000) + "\n\n" + ("B" * 5000)
        parts = split_large_content(text, "doc", max_chars=6000)
        assert len(parts) >= 2
        assert all(name.startswith("doc_part_") for _, name in parts)

    def test_max_episode_chars_default(self):
        assert MAX_EPISODE_CHARS == 8_000


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world", chunk_size=1000, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"
        assert chunks[0].id != ""

    def test_chunk_has_id(self):
        chunks = chunk_text("Some text here", chunk_size=1000, chunk_overlap=0)
        assert len(chunks[0].id) == 8  # md5[:8]

    def test_paragraph_splitting(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_header_splitting(self):
        text = "## Section A\n\nContent A\n\n## Section B\n\nContent B"
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=0)
        # Should split by headers
        assert len(chunks) >= 2
        titles = [c.metadata.get("section_title", "") for c in chunks]
        assert "Section A" in titles
        assert "Section B" in titles

    def test_table_kept_atomic(self):
        table = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        chunks = chunk_text(table, chunk_size=10, chunk_overlap=0)
        # Table should be one chunk regardless of size
        assert len(chunks) == 1

    def test_chunk_index_in_metadata(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=0)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap(self):
        text = ("A" * 100) + "\n\n" + ("B" * 100) + "\n\n" + ("C" * 100)
        chunks = chunk_text(text, chunk_size=120, chunk_overlap=20)
        assert len(chunks) >= 2
