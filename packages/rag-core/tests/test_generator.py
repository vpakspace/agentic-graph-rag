"""Tests for rag_core.generator."""

from unittest.mock import MagicMock, patch

from rag_core.generator import generate_answer
from rag_core.models import Chunk, SearchResult


def _mock_openai_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_result(text: str = "chunk content") -> SearchResult:
    return SearchResult(chunk=Chunk(content=text), score=0.8, rank=1)


class TestGenerateAnswer:
    def test_no_results_returns_fallback(self):
        client = MagicMock()
        result = generate_answer("question?", [], openai_client=client)
        assert result.confidence == 0.0
        assert "don't have enough context" in result.answer
        assert result.query == "question?"
        assert result.sources == []
        client.chat.completions.create.assert_not_called()

    def test_generates_answer_from_chunks(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "The answer based on Chunk 1 is X."
        )

        results = [_make_result("relevant content")]
        qa = generate_answer("What is X?", results, openai_client=client)

        assert qa.answer == "The answer based on Chunk 1 is X."
        assert 0.0 < qa.confidence <= 1.0
        assert qa.query == "What is X?"
        assert len(qa.sources) == 1
        client.chat.completions.create.assert_called_once()

    def test_includes_all_chunks_in_context(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("answer")

        results = [_make_result(f"content {i}") for i in range(3)]
        generate_answer("q", results, openai_client=client)

        call_args = client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert "[Chunk 1]" in user_msg
        assert "[Chunk 2]" in user_msg
        assert "[Chunk 3]" in user_msg

    def test_handles_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")

        results = [_make_result()]
        qa = generate_answer("q", results, openai_client=client)

        assert qa.confidence == 0.0
        assert "Error" in qa.answer
        assert len(qa.sources) == 1

    def test_handles_none_content(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(None)

        results = [_make_result()]
        qa = generate_answer("q", results, openai_client=client)
        assert qa.answer == ""

    @patch("rag_core.generator.get_settings")
    def test_creates_client_when_none(self, mock_settings):
        cfg = MagicMock()
        cfg.openai.api_key = "sk-test"
        cfg.openai.llm_model = "gpt-4o-mini"
        cfg.openai.llm_temperature = 0.1
        mock_settings.return_value = cfg

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response(
                "answer"
            )
            mock_cls.return_value = mock_client

            qa = generate_answer("q", [_make_result()])
            mock_cls.assert_called_once_with(api_key="sk-test")
            assert qa.answer == "answer"
