"""Tests for rag_core.reflector."""

from unittest.mock import MagicMock, patch

from rag_core.models import Chunk, SearchResult
from rag_core.reflector import evaluate_completeness, evaluate_relevance, generate_retry_query


def _mock_openai_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_result(text: str = "chunk content") -> SearchResult:
    return SearchResult(chunk=Chunk(content=text), score=0.8, rank=1)


class TestEvaluateRelevance:
    def test_empty_results_returns_zero(self):
        client = MagicMock()
        score = evaluate_relevance("q", [], openai_client=client)
        assert score == 0.0
        client.chat.completions.create.assert_not_called()

    def test_parses_comma_separated_scores(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("4, 5, 3")

        results = [_make_result(f"c{i}") for i in range(3)]
        score = evaluate_relevance("q", results, openai_client=client)

        assert abs(score - 4.0) < 1e-6  # (4+5+3)/3 = 4.0

    def test_handles_invalid_scores(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "4, invalid, 3"
        )

        results = [_make_result(f"c{i}") for i in range(3)]
        score = evaluate_relevance("q", results, openai_client=client)

        # (4 + 2.5 + 3) / 3 = 3.1666...
        assert abs(score - 3.1666) < 0.01

    def test_pads_missing_scores(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("5")

        results = [_make_result(f"c{i}") for i in range(3)]
        score = evaluate_relevance("q", results, openai_client=client)

        # (5 + 2.5 + 2.5) / 3 = 3.333...
        assert abs(score - 3.333) < 0.01

    def test_handles_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")

        results = [_make_result()]
        score = evaluate_relevance("q", results, openai_client=client)
        assert score == 2.5

    @patch("rag_core.reflector.get_settings")
    def test_creates_client_when_none(self, mock_settings):
        cfg = MagicMock()
        cfg.openai.api_key = "sk-test"
        cfg.openai.llm_model = "gpt-4o-mini"
        mock_settings.return_value = cfg

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response(
                "4, 5"
            )
            mock_cls.return_value = mock_client

            results = [_make_result(), _make_result()]
            score = evaluate_relevance("q", results)
            mock_cls.assert_called_once_with(api_key="sk-test")
            assert score == 4.5


class TestGenerateRetryQuery:
    def test_generates_retry_query(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "improved search query about topic"
        )

        results = [_make_result("partial content")]
        retry = generate_retry_query("original", results, openai_client=client)
        assert retry == "improved search query about topic"

    def test_empty_results_mentions_no_content(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("retry q")

        retry = generate_retry_query("original", [], openai_client=client)
        assert retry == "retry q"

        call_args = client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "No relevant content found" in user_msg

    def test_handles_api_error_returns_original(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("fail")

        result = generate_retry_query("original query", [], openai_client=client)
        assert result == "original query"

    def test_handles_none_content(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(None)

        result = generate_retry_query("fallback", [], openai_client=client)
        assert result == "fallback"

    @patch("rag_core.reflector.get_settings")
    def test_creates_client_when_none(self, mock_settings):
        cfg = MagicMock()
        cfg.openai.api_key = "sk-test"
        cfg.openai.llm_model = "gpt-4o-mini"
        mock_settings.return_value = cfg

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response(
                "better query"
            )
            mock_cls.return_value = mock_client

            result = generate_retry_query("q", [])
            mock_cls.assert_called_once_with(api_key="sk-test")
            assert result == "better query"


class TestEvaluateCompleteness:
    def test_returns_true_when_yes(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "YES, the answer covers all aspects."
        )
        assert evaluate_completeness("list all X", "Here are all X: A, B, C", openai_client=client) is True

    def test_returns_false_when_no(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "NO, the answer only mentions 2 out of 5 items."
        )
        assert evaluate_completeness("list all X", "Here are X: A, B", openai_client=client) is False

    def test_handles_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")
        # Should return True on error to avoid extra retries
        assert evaluate_completeness("q", "answer", openai_client=client) is True

    def test_handles_empty_response(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("")
        # Empty response doesn't start with YES → False
        assert evaluate_completeness("q", "answer", openai_client=client) is False

    @patch("rag_core.reflector.get_settings")
    def test_creates_client_when_none(self, mock_settings):
        cfg = MagicMock()
        cfg.openai.api_key = "sk-test"
        cfg.openai.llm_model_mini = "gpt-4o-mini"
        mock_settings.return_value = cfg

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response(
                "YES, complete"
            )
            mock_cls.return_value = mock_client

            result = evaluate_completeness("q", "answer")
            mock_cls.assert_called_once_with(api_key="sk-test")
            assert result is True
