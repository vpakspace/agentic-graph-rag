"""Tests for rag_core.enricher."""

from unittest.mock import MagicMock, patch

from rag_core.enricher import enrich_chunks
from rag_core.models import Chunk


class TestEnrichChunks:
    def test_empty_list(self):
        assert enrich_chunks([]) == []

    @patch("rag_core.enricher.openai")
    @patch("rag_core.enricher.get_settings")
    def test_enriches_with_summary(self, mock_settings, mock_openai):
        # Setup settings
        cfg = MagicMock()
        cfg.openai.api_key = "sk-test"
        cfg.openai.llm_model = "gpt-4o-mini"
        mock_settings.return_value = cfg

        # Setup OpenAI mock
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Context about the chunk"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        chunks = [Chunk(content="test content")]
        result = enrich_chunks(chunks, document_summary="A test document")

        assert len(result) == 1
        assert result[0].context == "Context about the chunk"

    @patch("rag_core.enricher.openai")
    @patch("rag_core.enricher.get_settings")
    def test_generates_summary_when_missing(self, mock_settings, mock_openai):
        cfg = MagicMock()
        cfg.openai.api_key = "sk-test"
        cfg.openai.llm_model = "gpt-4o-mini"
        mock_settings.return_value = cfg

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated text"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        chunks = [Chunk(content="content one"), Chunk(content="content two")]
        result = enrich_chunks(chunks)

        assert len(result) == 2
        # Should have called create at least 3 times: 1 summary + 2 contexts
        assert mock_client.chat.completions.create.call_count >= 3

    @patch("rag_core.enricher.openai")
    @patch("rag_core.enricher.get_settings")
    def test_handles_api_error_gracefully(self, mock_settings, mock_openai):
        cfg = MagicMock()
        cfg.openai.api_key = "sk-test"
        cfg.openai.llm_model = "gpt-4o-mini"
        mock_settings.return_value = cfg

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.OpenAI.return_value = mock_client

        chunks = [Chunk(content="test")]
        result = enrich_chunks(chunks, document_summary="doc")

        # Should return chunks even if enrichment fails
        assert len(result) == 1
        assert result[0].context == ""
