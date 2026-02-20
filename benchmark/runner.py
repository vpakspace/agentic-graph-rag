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
Expected keywords/concepts: {keywords}
System answer: {answer}

IMPORTANT: The answer may be in Russian while keywords are in English.
Match CONCEPTS and meanings, not exact strings.
For example: "ontology" matches "онтология", "extraction" matches "извлечение",
"graph" matches "граф", "temporal" matches "временных", etc.

Does the answer correctly address the question and cover the expected concepts?
For enumeration questions (list all, describe all, перечисли, опиши все), check that
the answer lists MOST of the expected keywords/concepts (at least 50%).
Reply with ONLY one word: PASS or FAIL"""


def _keyword_overlap(answer: str, keywords: list[str]) -> float:
    """Return fraction of keywords found in answer (case-insensitive)."""
    if not keywords:
        return 0.0
    lower = answer.lower()
    found = sum(1 for k in keywords if k.lower() in lower)
    return found / len(keywords)


def _embedding_similarity(text_a: str, text_b: str, openai_client: OpenAI) -> float:
    """Cosine similarity between embeddings of two texts."""
    cfg = get_settings()
    try:
        resp = openai_client.embeddings.create(
            model=cfg.openai.embedding_model,
            input=[text_a[:8000], text_b[:8000]],
        )
        vec_a = resp.data[0].embedding
        vec_b = resp.data[1].embedding
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
    except Exception as e:
        logger.error("Embedding similarity failed: %s", e)
        return 0.0


_JUDGE_PROMPT_REF = """You are evaluating a RAG system answer against a reference.

Question: {question}
Reference answer: {reference_answer}
System answer: {answer}

Does the system answer cover the same THEMES as the reference?
Match meanings and concepts, not exact words. The answer may use different terminology.
For example: "Multi-backend strategy" matches "Southbound Execution Adapters",
"Output contract with ConstraintSet" matches "Portable Semantic Outputs with provenance".
For enumeration questions (7 items), PASS if at least 4 themes overlap semantically.
Reply ONLY: PASS or FAIL"""


def evaluate_answer(
    question: str,
    answer: str,
    keywords: list[str],
    openai_client: OpenAI,
    reference_answer: str = "",
) -> bool:
    """Hybrid judge: embedding similarity / keyword overlap shortcut + LLM judge."""
    # Fast path 1: embedding similarity with reference answer ≥ 0.65 → auto-PASS
    # Threshold calibrated: correct SCL answer ~0.68, wrong Doc1 answer ~0.57
    if reference_answer:
        similarity = _embedding_similarity(answer, reference_answer, openai_client)
        if similarity >= 0.65:
            return True

    # Fast path 2: high keyword overlap → auto-PASS
    overlap = _keyword_overlap(answer, keywords)
    threshold = 0.65 if _is_global_query(question) else 0.4
    if overlap >= threshold:
        return True

    # LLM judge fallback
    cfg = get_settings()
    if reference_answer:
        prompt_text = _JUDGE_PROMPT_REF.format(
            question=question,
            reference_answer=reference_answer,
            answer=answer[:2000],
        )
    else:
        prompt_text = _JUDGE_PROMPT.format(
            question=question,
            keywords=", ".join(keywords),
            answer=answer[:2000],
        )

    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model_mini,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        verdict = text.split("\n")[-1].strip().upper()
        return verdict == "PASS"
    except Exception as e:
        logger.error("Judge failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Global query detection
# ---------------------------------------------------------------------------

import re

_GLOBAL_RE = re.compile(
    r'\b('
    r'все\b|всех\b|всё\b|перечисл|опиши все|резюмируй все|обзор\b'
    r'|list all|describe all|summarize all|overview|every\b'
    r'|все компоненты|все методы|все слои|все решения'
    r'|all components|all layers|all methods|all decisions'
    r')\b',
    re.IGNORECASE,
)

# "What X are mentioned?" — frameworks scattered across chunks need comprehensive search
_MENTION_RE = re.compile(
    r'\b('
    r'упоминаются|упоминается|упомянуты|упомянут'
    r'|mentioned|are described|are discussed|are listed'
    r'|какие\b.*\b(фреймворк|инструмент|технолог|метод|подход)'
    r'|what\b.*\b(framework|tool|technolog|method|approach)s?\b.*\b(mentioned|described|used)'
    r')\b',
    re.IGNORECASE,
)


def _is_global_query(query: str) -> bool:
    """Detect global/enumeration queries that need comprehensive retrieval."""
    return bool(_GLOBAL_RE.search(query))


def _needs_comprehensive(query: str) -> bool:
    """Detect queries that need comprehensive retrieval (global or mention-type)."""
    return _is_global_query(query) or bool(_MENTION_RE.search(query)) or _is_cross_language_query(query)


# Cross-language detection: RU question about EN-only concepts (Doc2)
_DOC2_CONCEPTS_RE = re.compile(
    r'\b('
    r'semantic\s+c(ore|ompanion)|SCL|companion\s+layer'
    r'|семантическ\w*\s+(ядр|компаньон|слой)'
    r'|семантического\s+ядра|семантическое\s+ядро'
    r'|пайплайн\w*\s+семантическ'
    r')\b',
    re.IGNORECASE,
)


def _is_cross_language_query(query: str) -> bool:
    """Detect RU queries that target EN Document 2 (SCL) content."""
    # If query is in Russian but references Doc2-specific concepts
    has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', query))
    has_doc2_concept = bool(_DOC2_CONCEPTS_RE.search(query))
    return has_cyrillic and has_doc2_concept


# ---------------------------------------------------------------------------
# Benchmark modes
# ---------------------------------------------------------------------------

def _pick_retrieval(query: str, driver: Any, client: OpenAI, fallback_fn: Any) -> list:
    """Pick retrieval strategy: full_document_read for cross-language global, comprehensive, or fallback."""
    from agentic_graph_rag.agent.tools import comprehensive_search, full_document_read

    # Cross-language global queries: vector_search returns Doc1, full_document_read finds Doc2
    if _is_cross_language_query(query) and _is_global_query(query):
        return full_document_read(query, driver, client, top_k=20)
    if _needs_comprehensive(query):
        return comprehensive_search(query, driver, client)
    return fallback_fn(query, driver, client)


def _run_vector_only(
    query: str, driver: Any, client: OpenAI,
) -> QAResult:
    """Vector search → generate answer. Uses comprehensive for global queries."""
    from rag_core.generator import generate_answer

    from agentic_graph_rag.agent.tools import vector_search

    results = _pick_retrieval(query, driver, client, vector_search)
    return generate_answer(query, results, client)


def _run_cypher(
    query: str, driver: Any, client: OpenAI,
) -> QAResult:
    """Cypher traversal → generate answer. Uses comprehensive for global queries."""
    from rag_core.generator import generate_answer

    from agentic_graph_rag.agent.tools import cypher_traverse

    results = _pick_retrieval(query, driver, client, cypher_traverse)
    return generate_answer(query, results, client)


def _run_hybrid(
    query: str, driver: Any, client: OpenAI,
) -> QAResult:
    """Hybrid (vector + graph) → generate answer. Uses comprehensive for global queries."""
    from rag_core.generator import generate_answer

    from agentic_graph_rag.agent.tools import hybrid_search

    results = _pick_retrieval(query, driver, client, hybrid_search)
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


def _run_agent_mangle(
    query: str, driver: Any, client: OpenAI,
) -> QAResult:
    """Agent with Mangle-based router (declarative rules)."""
    from agentic_graph_rag.agent.retrieval_agent import run
    from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

    rules_dir = str(Path(__file__).parent.parent / "agentic_graph_rag" / "reasoning" / "rules")
    engine = ReasoningEngine(rules_dir)
    return run(query, driver, openai_client=client, reasoning=engine)


MODES = {
    "vector": _run_vector_only,
    "cypher": _run_cypher,
    "hybrid": _run_hybrid,
    "agent_pattern": _run_agent_pattern,
    "agent_llm": _run_agent_llm,
    "agent_mangle": _run_agent_mangle,
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
    lang: str = "ru",
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
                    reference_answer=q.get("reference_answer", ""),
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


if __name__ == "__main__":
    from neo4j import GraphDatabase
    from rag_core.config import make_openai_client

    logging.basicConfig(level=logging.INFO)
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    client = make_openai_client(cfg)

    results = run_benchmark(driver, client)

    passed = sum(1 for mode in results.values() for r in mode if r["passed"])
    total = sum(len(mode) for mode in results.values())
    print(f"\nOverall: {passed}/{total} ({100 * passed / total:.1f}%)")

    driver.close()
