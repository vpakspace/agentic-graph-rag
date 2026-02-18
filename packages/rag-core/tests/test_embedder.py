"""Tests for rag_core.embedder."""

from unittest.mock import MagicMock, patch

import pytest
from rag_core.embedder import embed_chunks
from rag_core.models import Chunk


class TestEmbedChunks:
    def test_empty_list(self):
        assert embed_chunks([]) == []

    @patch("rag_core.embedder.make_openai_client")
    @patch("rag_core.embedder.get_settings")
    def test_embeds_chunks(self, mock_settings, mock_make_client):
        cfg = MagicMock()
        cfg.openai.embedding_model = "text-embedding-3-small"
        mock_settings.return_value = cfg

        # Mock embedding response
        emb1 = MagicMock()
        emb1.embedding = [0.1, 0.2, 0.3]
        emb2 = MagicMock()
        emb2.embedding = [0.4, 0.5, 0.6]

        mock_response = MagicMock()
        mock_response.data = [emb1, emb2]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_make_client.return_value = mock_client

        chunks = [Chunk(content="hello"), Chunk(content="world")]
        result = embed_chunks(chunks)

        assert len(result) == 2
        assert result[0].embedding == [0.1, 0.2, 0.3]
        assert result[1].embedding == [0.4, 0.5, 0.6]

        # Verify called with enriched_content
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["model"] == "text-embedding-3-small"

    @patch("rag_core.embedder.make_openai_client")
    @patch("rag_core.embedder.get_settings")
    def test_uses_enriched_content(self, mock_settings, mock_make_client):
        cfg = MagicMock()
        cfg.openai.embedding_model = "text-embedding-3-small"
        mock_settings.return_value = cfg

        emb = MagicMock()
        emb.embedding = [0.1]
        mock_response = MagicMock()
        mock_response.data = [emb]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_make_client.return_value = mock_client

        chunk = Chunk(content="text", context="prefix")
        embed_chunks([chunk])

        call_args = mock_client.embeddings.create.call_args
        texts = call_args.kwargs["input"]
        assert texts == ["prefix\n\ntext"]  # enriched_content

    @patch("rag_core.embedder.make_openai_client")
    @patch("rag_core.embedder.get_settings")
    def test_raises_on_api_error(self, mock_settings, mock_make_client):
        cfg = MagicMock()
        cfg.openai.embedding_model = "text-embedding-3-small"
        mock_settings.return_value = cfg

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API down")
        mock_make_client.return_value = mock_client

        with pytest.raises(Exception, match="API down"):
            embed_chunks([Chunk(content="test")])
