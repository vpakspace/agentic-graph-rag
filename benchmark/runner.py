"""Benchmark runner — evaluate retrieval modes across test questions.

Runs 5 modes: vector-only, cypher, hybrid, agent (pattern), agent (LLM).
Computes accuracy via LLM-as-judge, records latency and retries.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_core.config import get_settings
from rag_core.models import QAResult

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).parent / "questions.json"

# ---------------------------------------------------------------------------
# LLM-as-judge evaluation
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are evaluating a RAG system answer.

Question: {question}
Expected keywords: {keywords}
System answer: {answer}

Does the answer correctly address the question and mention the key concepts?
Reply with ONLY one word: PASS or FAIL"""


def evaluate_answer(
    question: str,
    answer: str,
    keywords: list[str],
    openai_client: OpenAI,
) -> bool:
    """Use LLM to judge if the answer is correct."""
    cfg = get_settings()
    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model_mini,
            messages=[
                {
                    "role": "user",
                    "content": _JUDGE_PROMPT.format(
                        question=question,
                        keywords=", ".join(keywords),
                        answer=answer[:500],
                    ),
                }
            ],
            temperature=0.0,
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
        return verdict == "PASS"
    except Exception as e:
        logger.error("Judge failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Benchmark modes
# ---------------------------------------------------------------------------

def _run_vector_only(
    query: str, driver: Any, client: OpenAI,
) -> QAResult:
    """Vector search → generate answer."""
    from rag_core.generator import generate_answer

    from agentic_graph_rag.agent.tools import vector_search

    results = vector_search(query, driver, client)
    return generate_answer(query, results, client)


def _run_cypher(
    query: str, driver: Any, client: OpenAI,
) -> QAResult:
    """Cypher traversal → generate answer."""
    from rag_core.generator import generate_answer

    from agentic_graph_rag.agent.tools import cypher_traverse

    results = cypher_traverse(query, driver, client)
    return generate_answer(query, results, client)


def _run_hybrid(
    query: str, driver: Any, client: OpenAI,
) -> QAResult:
    """Hybrid (vector + graph RRF) → generate answer."""
    from rag_core.generator import generate_answer

    from agentic_graph_rag.agent.tools import hybrid_search

    results = hybrid_search(query, driver, client)
    return generate_answer(query, results, client)


def _run_agent_pattern(
    query: str, driver: Any, client: OpenAI,
) -> QAResult:
    """Agent with pattern-based router."""
    from agentic_graph_rag.agent.retrieval_agent import run

    return run(query, driver, openai_client=client, use_llm_router=False)


def _run_agent_llm(
    query: str, driver: Any, client: OpenAI,
) -> QAResult:
    """Agent with LLM-based router."""
    from agentic_graph_rag.agent.retrieval_agent import run

    return run(query, driver, openai_client=client, use_llm_router=True)


MODES = {
    "vector": _run_vector_only,
    "cypher": _run_cypher,
    "hybrid": _run_hybrid,
    "agent_pattern": _run_agent_pattern,
    "agent_llm": _run_agent_llm,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_questions(path: Path | None = None) -> list[dict[str, Any]]:
    """Load benchmark questions from JSON file."""
    p = path or QUESTIONS_PATH
    with open(p) as f:
        return json.load(f)


def run_benchmark(
    driver: Any,
    openai_client: OpenAI,
    modes: list[str] | None = None,
    questions: list[dict[str, Any]] | None = None,
    lang: str = "en",
) -> dict[str, list[dict[str, Any]]]:
    """Run benchmark across specified modes.

    Returns dict[mode_name → list of result dicts].
    """
    if questions is None:
        questions = load_questions()

    run_modes = modes or list(MODES.keys())
    all_results: dict[str, list[dict[str, Any]]] = {}

    for mode_name in run_modes:
        mode_fn = MODES.get(mode_name)
        if mode_fn is None:
            logger.warning("Unknown mode: %s — skipping", mode_name)
            continue

        mode_results: list[dict[str, Any]] = []
        for q in questions:
            query = q.get(f"question_{lang}", q["question"])
            start = time.monotonic()
            try:
                qa: QAResult = mode_fn(query, driver, openai_client)
                latency = time.monotonic() - start
                passed = evaluate_answer(
                    query, qa.answer, q.get("keywords", []), openai_client,
                )
                mode_results.append({
                    "id": q["id"],
                    "question": query,
                    "type": q["type"],
                    "answer": qa.answer,
                    "confidence": qa.confidence,
                    "retries": qa.retries,
                    "latency": round(latency, 3),
                    "passed": passed,
                })
            except Exception as e:
                latency = time.monotonic() - start
                logger.error("Benchmark error [%s] q%d: %s", mode_name, q["id"], e)
                mode_results.append({
                    "id": q["id"],
                    "question": query,
                    "type": q["type"],
                    "answer": f"ERROR: {e}",
                    "confidence": 0.0,
                    "retries": 0,
                    "latency": round(latency, 3),
                    "passed": False,
                })

        all_results[mode_name] = mode_results
        logger.info(
            "Mode %s: %d/%d passed",
            mode_name,
            sum(1 for r in mode_results if r["passed"]),
            len(mode_results),
        )

    return all_results
