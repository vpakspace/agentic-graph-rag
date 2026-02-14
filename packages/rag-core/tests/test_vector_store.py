"""Tests for rag_core.vector_store."""

from unittest.mock import MagicMock, patch

from rag_core.models import Chunk
from rag_core.vector_store import INDEX_NAME, NODE_LABEL, VectorStore


class TestVectorStoreInit:
    @patch("rag_core.vector_store.get_settings")
    def test_accepts_driver(self, mock_settings):
        mock_driver = MagicMock()
        store = VectorStore(driver=mock_driver)
        assert store._driver is mock_driver

    @patch("rag_core.vector_store.get_settings")
    def test_constants(self, mock_settings):
        assert INDEX_NAME == "rag_chunks_index"
        assert NODE_LABEL == "RagChunk"


class TestVectorStoreAddChunks:
    @patch("rag_core.vector_store.get_settings")
    def test_add_empty_list(self, mock_settings):
        store = VectorStore(driver=MagicMock())
        assert store.add_chunks([]) == 0

    @patch("rag_core.vector_store.get_settings")
    def test_add_chunks(self, mock_settings):
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = VectorStore(driver=mock_driver)
        chunks = [
            Chunk(id="c1", content="hello", embedding=[0.1, 0.2]),
            Chunk(id="c2", content="world", embedding=[0.3, 0.4]),
        ]
        count = store.add_chunks(chunks)

        assert count == 2
        assert mock_session.run.call_count == 2


class TestVectorStoreSearch:
    @patch("rag_core.vector_store.get_settings")
    def test_search_returns_results(self, mock_settings):
        mock_settings.return_value.retrieval.top_k_vector = 5

        mock_record = {
            "id": "c1",
            "content": "test content",
            "context": "",
            "score": 0.95,
        }
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([mock_record]))

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = VectorStore(driver=mock_driver)
        results = store.search([0.1, 0.2, 0.3])

        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].chunk.content == "test content"
        assert results[0].rank == 1

    @patch("rag_core.vector_store.get_settings")
    def test_search_with_explicit_top_k(self, mock_settings):
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = VectorStore(driver=mock_driver)
        results = store.search([0.1], top_k=3)

        assert results == []
        call_kwargs = mock_session.run.call_args.kwargs
        assert call_kwargs["top_k"] == 3


class TestVectorStoreDeleteAndCount:
    @patch("rag_core.vector_store.get_settings")
    def test_count(self, mock_settings):
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=42)

        mock_result = MagicMock()
        mock_result.single.return_value = mock_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = VectorStore(driver=mock_driver)
        assert store.count() == 42

    @patch("rag_core.vector_store.get_settings")
    def test_delete_all(self, mock_settings):
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=10)

        mock_result = MagicMock()
        mock_result.single.return_value = mock_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = VectorStore(driver=mock_driver)
        assert store.delete_all() == 10

    @patch("rag_core.vector_store.get_settings")
    def test_close(self, mock_settings):
        mock_driver = MagicMock()
        store = VectorStore(driver=mock_driver)
        store.close()
        mock_driver.close.assert_called_once()
