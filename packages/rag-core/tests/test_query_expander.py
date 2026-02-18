"""Tests for rag_core.query_expander."""

from unittest.mock import MagicMock, patch

from rag_core.query_expander import expand_query, generate_multi_queries


def _mock_openai_response(content: str) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestExpandQuery:
    def test_returns_expanded_query(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "What are the detailed mechanisms of caching in distributed systems?"
        )

        result = expand_query("caching", openai_client=client)

        assert "caching" in result.lower()
        client.chat.completions.create.assert_called_once()

    def test_strips_whitespace(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "  expanded query  \n"
        )

        result = expand_query("test", openai_client=client)
        assert result == "expanded query"

    def test_falls_back_to_original_on_none_content(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(None)

        result = expand_query("original query", openai_client=client)
        assert result == "original query"

    @patch("rag_core.query_expander.make_openai_client")
    @patch("rag_core.query_expander.get_settings")
    def test_creates_client_when_none(self, mock_settings, mock_make_client):
        cfg = MagicMock()
        cfg.openai.llm_model = "gpt-4o-mini"
        mock_settings.return_value = cfg

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "expanded"
        )
        mock_make_client.return_value = mock_client

        result = expand_query("test")
        mock_make_client.assert_called_once_with(cfg)
        assert result == "expanded"


class TestGenerateMultiQueries:
    def test_returns_n_queries(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "Query variant A\nQuery variant B\nQuery variant C"
        )

        result = generate_multi_queries("test", n=3, openai_client=client)
        assert len(result) == 3

    def test_pads_if_too_few(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "Only one variant"
        )

        result = generate_multi_queries("original", n=3, openai_client=client)
        assert len(result) == 3
        assert result[1] == "original"
        assert result[2] == "original"

    def test_truncates_if_too_many(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "A\nB\nC\nD\nE"
        )

        result = generate_multi_queries("test", n=3, openai_client=client)
        assert len(result) == 3

    def test_filters_empty_lines(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "A\n\nB\n\nC"
        )

        result = generate_multi_queries("test", n=3, openai_client=client)
        assert len(result) == 3
        assert all(q.strip() for q in result)

    def test_handles_none_content(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(None)

        result = generate_multi_queries("fallback", n=2, openai_client=client)
        assert len(result) == 2
        assert result[0] == "fallback"

    @patch("rag_core.query_expander.make_openai_client")
    @patch("rag_core.query_expander.get_settings")
    def test_creates_client_when_none(self, mock_settings, mock_make_client):
        cfg = MagicMock()
        cfg.openai.llm_model = "gpt-4o-mini"
        mock_settings.return_value = cfg

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "A\nB\nC"
        )
        mock_make_client.return_value = mock_client

        result = generate_multi_queries("test", n=3)
        mock_make_client.assert_called_once_with(cfg)
        assert len(result) == 3
