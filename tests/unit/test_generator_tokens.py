"""Tests for token usage capture in generator."""
from unittest.mock import MagicMock

from rag_core.models import Chunk, SearchResult


def _make_results():
    return [SearchResult(chunk=Chunk(id="c1", content="text"), score=0.9, rank=1)]


def _mock_openai_response(prompt_tokens=100, completion_tokens=50):
    choice = MagicMock()
    choice.message.content = "Generated answer"
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def test_generate_answer_returns_token_usage():
    from rag_core.generator import generate_answer

    client = MagicMock()
    client.chat.completions.create.return_value = _mock_openai_response(200, 80)

    qa = generate_answer("test", _make_results(), openai_client=client)
    assert qa.prompt_tokens == 200
    assert qa.completion_tokens == 80


def test_generate_answer_tokens_default_zero_on_error():
    from rag_core.generator import generate_answer

    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("API error")

    qa = generate_answer("test", _make_results(), openai_client=client)
    assert qa.prompt_tokens == 0
    assert qa.completion_tokens == 0
