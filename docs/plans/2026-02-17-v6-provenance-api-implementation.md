# v6: Provenance + Typed API Contract — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add full pipeline provenance and a typed API contract (FastAPI REST + MCP server) to Agentic Graph RAG.

**Architecture:** Monolith API — FastAPI + MCP in one process on port 8507. PipelineService as internal typed contract. Streamlit becomes thin HTTP client for Search & Q&A tab. Provenance collected as structured PipelineTrace at every pipeline step.

**Tech Stack:** FastAPI, uvicorn, FastMCP, httpx, Pydantic v2

**Design doc:** `docs/plans/2026-02-17-v6-provenance-api-design.md`

---

## Task 1: Add dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add new packages**

Add after the `streamlit>=1.30.0` line in `requirements.txt`:

```
# API server (v6)
fastapi>=0.115.0
uvicorn>=0.30.0
httpx>=0.27.0
fastmcp>=2.0.0
```

**Step 2: Install**

Run: `pip install fastapi uvicorn httpx fastmcp`
Expected: Success, no conflicts

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add fastapi, uvicorn, httpx, fastmcp dependencies"
```

---

## Task 2: Provenance models

**Files:**
- Modify: `packages/rag-core/rag_core/models.py:132-143` (add models before QAResult, add trace field)
- Test: `tests/unit/test_provenance_models.py`

**Step 1: Write failing tests**

Create `tests/unit/test_provenance_models.py`:

```python
"""Tests for provenance models."""
import json
from rag_core.models import (
    EscalationStep,
    GeneratorStep,
    PipelineTrace,
    QAResult,
    RouterDecision,
    RouterStep,
    QueryType,
    ToolStep,
)


def test_tool_step_defaults():
    step = ToolStep(tool_name="vector_search")
    assert step.tool_name == "vector_search"
    assert step.results_count == 0
    assert step.relevance_score == 0.0
    assert step.duration_ms == 0
    assert step.query_used == ""


def test_escalation_step():
    step = EscalationStep(
        from_tool="vector_search",
        to_tool="cypher_traverse",
        reason="relevance 1.6 < threshold 2.0",
        rephrased_query="rephrased query",
    )
    assert step.from_tool == "vector_search"
    assert step.to_tool == "cypher_traverse"


def test_router_step_with_rules():
    decision = RouterDecision(
        query_type=QueryType.GLOBAL,
        confidence=0.7,
        reasoning="Mangle rule matched",
        suggested_tool="comprehensive_search",
    )
    step = RouterStep(
        method="mangle",
        decision=decision,
        duration_ms=12,
        rules_fired=["routing.mg:global_query"],
    )
    assert step.method == "mangle"
    assert step.rules_fired == ["routing.mg:global_query"]
    assert step.decision.query_type == QueryType.GLOBAL


def test_generator_step():
    step = GeneratorStep(
        model="gpt-4o-mini",
        prompt_tokens=2100,
        completion_tokens=350,
        confidence=0.82,
        completeness_check=True,
    )
    assert step.prompt_tokens == 2100
    assert step.completeness_check is True


def test_pipeline_trace_serialization():
    trace = PipelineTrace(
        trace_id="tr_abc123",
        timestamp="2026-02-17T14:30:00Z",
        query="test query",
        tool_steps=[
            ToolStep(tool_name="vector_search", results_count=5, relevance_score=3.2),
        ],
        total_duration_ms=1200,
    )
    data = trace.model_dump()
    assert data["trace_id"] == "tr_abc123"
    assert len(data["tool_steps"]) == 1
    assert data["tool_steps"][0]["tool_name"] == "vector_search"

    # Round-trip
    restored = PipelineTrace.model_validate(data)
    assert restored.trace_id == trace.trace_id


def test_pipeline_trace_json():
    trace = PipelineTrace(
        trace_id="tr_xyz",
        timestamp="2026-02-17T00:00:00Z",
        query="q",
    )
    json_str = trace.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["trace_id"] == "tr_xyz"
    assert parsed["tool_steps"] == []
    assert parsed["escalation_steps"] == []


def test_qa_result_has_trace_field():
    qa = QAResult(answer="test", query="q")
    assert qa.trace is None

    trace = PipelineTrace(
        trace_id="tr_001",
        timestamp="2026-02-17T00:00:00Z",
        query="q",
    )
    qa.trace = trace
    assert qa.trace.trace_id == "tr_001"


def test_full_pipeline_trace():
    """Full trace with all sections populated."""
    decision = RouterDecision(
        query_type=QueryType.SIMPLE,
        confidence=0.5,
        reasoning="Pattern matched",
        suggested_tool="vector_search",
    )
    trace = PipelineTrace(
        trace_id="tr_full",
        timestamp="2026-02-17T12:00:00Z",
        query="test",
        router_step=RouterStep(method="pattern", decision=decision, duration_ms=5),
        tool_steps=[
            ToolStep(tool_name="vector_search", results_count=10, relevance_score=1.5, duration_ms=300),
            ToolStep(tool_name="cypher_traverse", results_count=8, relevance_score=3.1, duration_ms=500),
        ],
        escalation_steps=[
            EscalationStep(
                from_tool="vector_search",
                to_tool="cypher_traverse",
                reason="relevance 1.5 < threshold 2.0",
            ),
        ],
        generator_step=GeneratorStep(model="gpt-4o-mini", confidence=0.75),
        total_duration_ms=1200,
    )
    data = trace.model_dump()
    assert len(data["tool_steps"]) == 2
    assert len(data["escalation_steps"]) == 1
    assert data["generator_step"]["model"] == "gpt-4o-mini"
```

**Step 2: Run tests — verify they fail**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_provenance_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'ToolStep'`

**Step 3: Implement provenance models**

In `packages/rag-core/rag_core/models.py`, add BEFORE `class QAResult` (line 132):

```python
# ---------------------------------------------------------------------------
# Provenance models (v6 — pipeline trace)
# ---------------------------------------------------------------------------

class ToolStep(BaseModel):
    """One tool execution step in the pipeline."""

    tool_name: str
    results_count: int = 0
    relevance_score: float = 0.0
    duration_ms: int = 0
    query_used: str = ""


class EscalationStep(BaseModel):
    """Tool-to-tool escalation record."""

    from_tool: str
    to_tool: str
    reason: str = ""
    rephrased_query: str = ""


class RouterStep(BaseModel):
    """Router classification result with timing."""

    method: str  # "pattern", "llm", "mangle"
    decision: RouterDecision
    duration_ms: int = 0
    rules_fired: list[str] = Field(default_factory=list)


class GeneratorStep(BaseModel):
    """Answer generation metadata."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    confidence: float = 0.0
    completeness_check: bool | None = None


class PipelineTrace(BaseModel):
    """Full pipeline provenance artifact."""

    trace_id: str
    timestamp: str
    query: str
    router_step: RouterStep | None = None
    tool_steps: list[ToolStep] = Field(default_factory=list)
    escalation_steps: list[EscalationStep] = Field(default_factory=list)
    generator_step: GeneratorStep | None = None
    total_duration_ms: int = 0
```

Then add `trace` field to `QAResult`:

```python
class QAResult(BaseModel):
    # ... existing fields ...
    graph_context: GraphContext | None = None
    trace: PipelineTrace | None = None  # NEW: v6 provenance
```

**Step 4: Run tests — verify they pass**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_provenance_models.py -v`
Expected: 8 PASSED

**Step 5: Run all existing tests — verify no regressions**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/ -x -q`
Expected: 398+ passed (existing + 8 new)

**Step 6: Commit**

```bash
git add packages/rag-core/rag_core/models.py tests/unit/test_provenance_models.py
git commit -m "feat(models): add provenance models — ToolStep, PipelineTrace, etc."
```

---

## Task 3: Instrument retrieval_agent with provenance

**Files:**
- Modify: `agentic_graph_rag/agent/retrieval_agent.py`
- Test: `tests/unit/test_provenance_collection.py`

**Step 1: Write failing tests**

Create `tests/unit/test_provenance_collection.py`:

```python
"""Tests for provenance collection in retrieval agent."""
from unittest.mock import MagicMock, patch

from rag_core.models import (
    Chunk,
    PipelineTrace,
    QAResult,
    QueryType,
    RouterDecision,
    SearchResult,
    ToolStep,
)


def _make_results(n=3):
    return [
        SearchResult(chunk=Chunk(id=f"c{i}", content=f"text {i}"), score=0.9 - i * 0.1, rank=i + 1)
        for i in range(n)
    ]


def _make_decision(tool="vector_search", qtype=QueryType.SIMPLE):
    return RouterDecision(
        query_type=qtype, confidence=0.5, reasoning="test", suggested_tool=tool,
    )


@patch("agentic_graph_rag.agent.retrieval_agent.evaluate_relevance", return_value=4.0)
@patch("agentic_graph_rag.agent.retrieval_agent._TOOL_REGISTRY")
def test_self_correction_loop_records_tool_step(mock_registry, mock_eval):
    mock_fn = MagicMock(return_value=_make_results())
    mock_registry.__getitem__ = MagicMock(return_value=mock_fn)
    mock_registry.__contains__ = MagicMock(return_value=True)

    from agentic_graph_rag.agent.retrieval_agent import self_correction_loop

    decision = _make_decision()
    trace = PipelineTrace(trace_id="tr_test", timestamp="T", query="q")

    results, retries = self_correction_loop(
        "q", MagicMock(), MagicMock(), decision, trace=trace,
    )
    assert len(trace.tool_steps) >= 1
    assert trace.tool_steps[0].tool_name == "vector_search"
    assert trace.tool_steps[0].results_count == 3
    assert trace.tool_steps[0].relevance_score == 4.0


@patch("agentic_graph_rag.agent.retrieval_agent.evaluate_relevance", return_value=1.0)
@patch("agentic_graph_rag.agent.retrieval_agent.generate_retry_query", return_value="rephrased")
@patch("agentic_graph_rag.agent.retrieval_agent._TOOL_REGISTRY")
def test_escalation_recorded_in_trace(mock_registry, mock_retry, mock_eval):
    mock_fn = MagicMock(return_value=_make_results())
    mock_registry.__getitem__ = MagicMock(return_value=mock_fn)
    mock_registry.__contains__ = MagicMock(return_value=True)

    from agentic_graph_rag.agent.retrieval_agent import self_correction_loop

    decision = _make_decision()
    trace = PipelineTrace(trace_id="tr_esc", timestamp="T", query="q")

    results, retries = self_correction_loop(
        "q", MagicMock(), MagicMock(), decision, trace=trace,
        max_retries=1, relevance_threshold=3.0,
    )
    assert len(trace.escalation_steps) >= 1
    assert trace.escalation_steps[0].from_tool == "vector_search"


@patch("agentic_graph_rag.agent.retrieval_agent.classify_query")
@patch("agentic_graph_rag.agent.retrieval_agent.self_correction_loop")
@patch("agentic_graph_rag.agent.retrieval_agent.generate_answer")
def test_run_returns_qa_with_trace(mock_gen, mock_loop, mock_classify):
    mock_classify.return_value = _make_decision()
    mock_loop.return_value = (_make_results(), 0)
    mock_gen.return_value = QAResult(answer="test", query="q", sources=_make_results())

    from agentic_graph_rag.agent.retrieval_agent import run

    qa = run("q", MagicMock(), MagicMock())
    assert qa.trace is not None
    assert qa.trace.trace_id.startswith("tr_")
    assert qa.trace.router_step is not None
```

**Step 2: Run tests — verify they fail**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_provenance_collection.py -v`
Expected: FAIL — `self_correction_loop() got an unexpected keyword argument 'trace'`

**Step 3: Modify `retrieval_agent.py`**

Changes to `self_correction_loop()` (line 75):
- Add `trace: PipelineTrace | None = None` parameter
- Wrap each tool call with timing
- Record `ToolStep` after each execution
- Record `EscalationStep` on each escalation

Changes to `run()` (line 163):
- Create `PipelineTrace` at the top with UUID + timestamp
- Wrap `classify_query()` with timing → create `RouterStep`
- Pass `trace` to `self_correction_loop()`
- Create `GeneratorStep` from `generate_answer()` return
- Attach `trace` to `QAResult`

Key implementation for `self_correction_loop`:

```python
import time
import uuid

def self_correction_loop(
    query, driver, openai_client, decision,
    max_retries=None, relevance_threshold=None,
    trace=None,  # NEW: PipelineTrace
):
    # ... existing setup ...

    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        tool_fn = _TOOL_REGISTRY[tool_name]
        results = tool_fn(query, driver, openai_client)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        tried_tools.add(tool_name)

        score = 0.0
        if results:
            score = evaluate_relevance(query, results, openai_client=openai_client)

        # Record tool step
        if trace is not None:
            trace.tool_steps.append(ToolStep(
                tool_name=tool_name,
                results_count=len(results),
                relevance_score=score,
                duration_ms=elapsed_ms,
                query_used=query,
            ))

        # ... existing score tracking ...

        if attempt < max_retries:
            next_tool = _get_next_tool(tool_name, tried_tools)
            if next_tool:
                rephrased = generate_retry_query(query, results, openai_client=openai_client)
                # Record escalation
                if trace is not None:
                    trace.escalation_steps.append(EscalationStep(
                        from_tool=tool_name,
                        to_tool=next_tool,
                        reason=f"relevance {score:.1f} < threshold {relevance_threshold}",
                        rephrased_query=rephrased,
                    ))
                query = rephrased
                tool_name = next_tool
            # ...
```

Key implementation for `run`:

```python
def run(query, driver, openai_client=None, use_llm_router=False, reasoning=None):
    from rag_core.models import GeneratorStep, PipelineTrace, RouterStep

    cfg = get_settings()
    # ... existing client init ...

    t_start = time.perf_counter()
    trace = PipelineTrace(
        trace_id=f"tr_{uuid.uuid4().hex[:12]}",
        timestamp=datetime.utcnow().isoformat() + "Z",
        query=query,
    )

    # Step 1: Classify with timing
    t0 = time.perf_counter()
    decision = classify_query(query, use_llm=use_llm_router, ...)
    router_ms = int((time.perf_counter() - t0) * 1000)

    router_method = "mangle" if reasoning else ("llm" if use_llm_router else "pattern")
    trace.router_step = RouterStep(
        method=router_method,
        decision=decision,
        duration_ms=router_ms,
    )

    # Step 2: Self-correction loop (passes trace)
    results, retries = self_correction_loop(
        query, driver, openai_client, decision, trace=trace,
    )

    # Step 3: Generate answer
    qa_result = generate_answer(query, results, openai_client=openai_client)

    # ... existing completeness check ...

    # Generator step
    trace.generator_step = GeneratorStep(
        model=cfg.openai.llm_model,
        confidence=qa_result.confidence,
    )
    trace.total_duration_ms = int((time.perf_counter() - t_start) * 1000)

    qa_result.retries = retries
    qa_result.router_decision = decision
    qa_result.trace = trace
    return qa_result
```

**Step 4: Run tests — verify they pass**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_provenance_collection.py -v`
Expected: 3 PASSED

**Step 5: Run all tests**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/ -x -q`
Expected: 406+ passed

**Step 6: Commit**

```bash
git add agentic_graph_rag/agent/retrieval_agent.py tests/unit/test_provenance_collection.py
git commit -m "feat(agent): instrument retrieval pipeline with provenance trace"
```

---

## Task 4: Capture token usage in generator

**Files:**
- Modify: `packages/rag-core/rag_core/generator.py:21-88`
- Test: `tests/unit/test_generator_tokens.py`

**Step 1: Write failing test**

Create `tests/unit/test_generator_tokens.py`:

```python
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
```

**Step 2: Run tests — verify they fail**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_generator_tokens.py -v`
Expected: FAIL — `AttributeError: 'QAResult' has no attribute 'prompt_tokens'`

**Step 3: Add token fields to QAResult and capture in generator**

In `models.py`, add to `QAResult`:
```python
    prompt_tokens: int = 0
    completion_tokens: int = 0
```

In `generator.py`, after `answer_text = response.choices[0].message.content`:
```python
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
```

And in the return:
```python
        return QAResult(
            answer=answer_text,
            sources=results,
            confidence=confidence,
            query=query,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
```

**Step 4: Run tests — verify pass**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_generator_tokens.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add packages/rag-core/rag_core/models.py packages/rag-core/rag_core/generator.py tests/unit/test_generator_tokens.py
git commit -m "feat(generator): capture prompt/completion token usage in QAResult"
```

---

## Task 5: PipelineService (internal contract)

**Files:**
- Create: `agentic_graph_rag/service.py`
- Test: `tests/unit/test_pipeline_service.py`

**Step 1: Write failing tests**

Create `tests/unit/test_pipeline_service.py`:

```python
"""Tests for PipelineService."""
from unittest.mock import MagicMock, patch

from rag_core.models import QAResult, Chunk, SearchResult


def _mock_qa():
    return QAResult(answer="test answer", query="q", sources=[
        SearchResult(chunk=Chunk(id="c1", content="text"), score=0.9, rank=1),
    ], confidence=0.8)


@patch("agentic_graph_rag.service.agent_run")
def test_service_query_returns_qa_with_trace(mock_run):
    mock_run.return_value = _mock_qa()
    mock_run.return_value.trace = MagicMock(trace_id="tr_abc")

    from agentic_graph_rag.service import PipelineService

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())
    qa = svc.query("test question")
    assert qa.answer == "test answer"
    mock_run.assert_called_once()


@patch("agentic_graph_rag.service.agent_run")
def test_service_caches_trace(mock_run):
    from rag_core.models import PipelineTrace
    qa = _mock_qa()
    qa.trace = PipelineTrace(trace_id="tr_cached", timestamp="T", query="q")
    mock_run.return_value = qa

    from agentic_graph_rag.service import PipelineService

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())
    svc.query("test")
    assert svc.get_trace("tr_cached") is not None
    assert svc.get_trace("nonexistent") is None


def test_service_health():
    from agentic_graph_rag.service import PipelineService

    driver = MagicMock()
    session = MagicMock()
    session.run.return_value.single.return_value = [1]
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    svc = PipelineService(driver=driver, openai_client=MagicMock())
    health = svc.health()
    assert health["status"] == "ok"


def test_service_trace_cache_bounded():
    from agentic_graph_rag.service import PipelineService

    svc = PipelineService(driver=MagicMock(), openai_client=MagicMock())
    from rag_core.models import PipelineTrace

    # Fill cache beyond limit
    for i in range(105):
        trace = PipelineTrace(trace_id=f"tr_{i:04d}", timestamp="T", query="q")
        svc._cache_trace(trace)

    # Oldest should be evicted (cache max 100)
    assert svc.get_trace("tr_0000") is None
    assert svc.get_trace("tr_0104") is not None
```

**Step 2: Run tests — verify they fail**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_pipeline_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_graph_rag.service'`

**Step 3: Implement PipelineService**

Create `agentic_graph_rag/service.py`:

```python
"""PipelineService — typed contract for Agentic Graph RAG pipeline.

All clients (FastAPI, MCP, Streamlit) use this service.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from agentic_graph_rag.agent.retrieval_agent import run as agent_run
from rag_core.models import PipelineTrace, QAResult

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

    from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

logger = logging.getLogger(__name__)

_TRACE_CACHE_MAX = 100


class PipelineService:
    """Typed contract for the Agentic Graph RAG pipeline."""

    def __init__(
        self,
        driver: Driver,
        openai_client: OpenAI,
        reasoning: ReasoningEngine | None = None,
    ):
        self._driver = driver
        self._client = openai_client
        self._reasoning = reasoning
        self._trace_cache: OrderedDict[str, PipelineTrace] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        mode: str = "agent_pattern",
        lang: str = "ru",
    ) -> QAResult:
        """Full pipeline: route -> retrieve -> generate -> trace."""
        use_llm = mode == "agent_llm"
        reasoning = self._reasoning if mode == "agent_mangle" else None

        qa = agent_run(
            text,
            self._driver,
            openai_client=self._client,
            use_llm_router=use_llm,
            reasoning=reasoning,
        )

        if qa.trace:
            self._cache_trace(qa.trace)

        return qa

    def get_trace(self, trace_id: str) -> PipelineTrace | None:
        """Retrieve trace from in-memory cache."""
        return self._trace_cache.get(trace_id)

    def health(self) -> dict:
        """Neo4j connectivity check."""
        try:
            with self._driver.session() as session:
                session.run("RETURN 1").single()
            return {"status": "ok", "neo4j": "connected"}
        except Exception as e:
            return {"status": "degraded", "neo4j": str(e)}

    def graph_stats(self) -> dict:
        """Node and edge counts."""
        try:
            with self._driver.session() as session:
                result = session.run(
                    "MATCH (n) RETURN count(n) AS nodes "
                    "UNION ALL "
                    "MATCH ()-[r]->() RETURN count(r) AS nodes"
                )
                counts = [r["nodes"] for r in result]
            return {"nodes": counts[0] if counts else 0, "edges": counts[1] if len(counts) > 1 else 0}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cache_trace(self, trace: PipelineTrace) -> None:
        """Add trace to bounded cache (LRU eviction)."""
        self._trace_cache[trace.trace_id] = trace
        while len(self._trace_cache) > _TRACE_CACHE_MAX:
            self._trace_cache.popitem(last=False)
```

**Step 4: Run tests — verify pass**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_pipeline_service.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add agentic_graph_rag/service.py tests/unit/test_pipeline_service.py
git commit -m "feat(service): add PipelineService — typed contract with trace cache"
```

---

## Task 6: FastAPI application

**Files:**
- Create: `api/__init__.py`
- Create: `api/deps.py`
- Create: `api/routes.py`
- Create: `api/app.py`
- Test: `tests/unit/test_api_routes.py`

**Step 1: Write failing tests**

Create `tests/unit/test_api_routes.py`:

```python
"""Tests for FastAPI routes."""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_service():
    from rag_core.models import (
        Chunk, PipelineTrace, QAResult, QueryType,
        RouterDecision, RouterStep, SearchResult, ToolStep,
    )

    svc = MagicMock()
    trace = PipelineTrace(
        trace_id="tr_test123",
        timestamp="2026-02-17T00:00:00Z",
        query="test",
        router_step=RouterStep(
            method="pattern",
            decision=RouterDecision(
                query_type=QueryType.SIMPLE, confidence=0.5,
                reasoning="test", suggested_tool="vector_search",
            ),
        ),
        tool_steps=[ToolStep(tool_name="vector_search", results_count=3)],
        total_duration_ms=500,
    )
    qa = QAResult(
        answer="Test answer",
        query="test",
        sources=[SearchResult(chunk=Chunk(id="c1", content="text"), score=0.9, rank=1)],
        confidence=0.8,
        trace=trace,
    )
    svc.query.return_value = qa
    svc.health.return_value = {"status": "ok", "neo4j": "connected"}
    svc.get_trace.return_value = trace
    svc.graph_stats.return_value = {"nodes": 100, "edges": 200}
    return svc


@pytest.fixture
def client(mock_service):
    from fastapi.testclient import TestClient

    from api.app import create_app

    app = create_app(service=mock_service)
    return TestClient(app)


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query(client, mock_service):
    resp = client.post("/api/v1/query", json={"text": "test question"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer"
    assert data["trace"]["trace_id"] == "tr_test123"
    mock_service.query.assert_called_once()


def test_query_missing_text(client):
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422


def test_get_trace(client):
    resp = client.get("/api/v1/trace/tr_test123")
    assert resp.status_code == 200
    assert resp.json()["trace_id"] == "tr_test123"


def test_get_trace_not_found(client, mock_service):
    mock_service.get_trace.return_value = None
    resp = client.get("/api/v1/trace/tr_nonexistent")
    assert resp.status_code == 404


def test_graph_stats(client):
    resp = client.get("/api/v1/graph/stats")
    assert resp.status_code == 200
    assert resp.json()["nodes"] == 100
```

**Step 2: Run tests — verify they fail**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_api_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api'`

**Step 3: Create API module**

Create `api/__init__.py` (empty).

Create `api/deps.py`:
```python
"""Dependency injection for FastAPI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_graph_rag.service import PipelineService

_service: PipelineService | None = None


def get_service() -> PipelineService:
    """Get the PipelineService singleton."""
    if _service is None:
        raise RuntimeError("PipelineService not initialized")
    return _service


def set_service(service: PipelineService) -> None:
    """Set the PipelineService singleton (called at startup)."""
    global _service  # noqa: PLW0603
    _service = service
```

Create `api/routes.py`:
```python
"""FastAPI route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_service

router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    text: str
    mode: str = "agent_pattern"
    lang: str = "ru"


class SearchRequest(BaseModel):
    text: str
    tool: str = "vector_search"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    svc = get_service()
    return svc.health()


@router.post("/query")
def query(req: QueryRequest):
    svc = get_service()
    qa = svc.query(req.text, mode=req.mode, lang=req.lang)
    return qa.model_dump()


@router.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    svc = get_service()
    trace = svc.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.model_dump()


@router.get("/graph/stats")
def graph_stats():
    svc = get_service()
    return svc.graph_stats()
```

Create `api/app.py`:
```python
"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from api.deps import set_service
from api.routes import router

if TYPE_CHECKING:
    from agentic_graph_rag.service import PipelineService


def create_app(service: PipelineService | None = None) -> FastAPI:
    """Create FastAPI app. If service is provided, use it (for testing).
    Otherwise, create from config at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            # Production: create service from config
            from neo4j import GraphDatabase
            from openai import OpenAI

            from rag_core.config import get_settings

            cfg = get_settings()
            driver = GraphDatabase.driver(
                cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
            )
            client = OpenAI(api_key=cfg.openai.api_key)

            reasoning = None
            try:
                from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine
                reasoning = ReasoningEngine()
            except Exception:
                pass

            from agentic_graph_rag.service import PipelineService
            svc = PipelineService(driver, client, reasoning)
            set_service(svc)
            yield
            driver.close()
        else:
            # Testing: use provided service
            set_service(service)
            yield

    app = FastAPI(
        title="Agentic Graph RAG API",
        version="0.6.0",
        description="Typed API contract with full pipeline provenance",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
```

**Step 4: Run tests — verify pass**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_api_routes.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add api/ tests/unit/test_api_routes.py
git commit -m "feat(api): add FastAPI REST endpoints with typed contract"
```

---

## Task 7: MCP server

**Files:**
- Create: `api/mcp_server.py`
- Test: `tests/unit/test_mcp_tools.py`

**Step 1: Write failing tests**

Create `tests/unit/test_mcp_tools.py`:

```python
"""Tests for MCP server tools."""
from unittest.mock import MagicMock

from rag_core.models import (
    Chunk, PipelineTrace, QAResult, QueryType,
    RouterDecision, RouterStep, SearchResult, ToolStep,
)


def _make_qa():
    trace = PipelineTrace(
        trace_id="tr_mcp",
        timestamp="2026-02-17T00:00:00Z",
        query="test",
    )
    return QAResult(answer="answer", query="test", confidence=0.8, trace=trace,
                    sources=[SearchResult(chunk=Chunk(id="c1", content="t"), score=0.9, rank=1)])


def test_resolve_intent_tool():
    from api.mcp_server import create_mcp_tools

    svc = MagicMock()
    svc.query.return_value = _make_qa()

    tools = create_mcp_tools(svc)
    result = tools["resolve_intent"]("test query", "agent_pattern")
    assert result["answer"] == "answer"
    assert result["trace"]["trace_id"] == "tr_mcp"


def test_explain_trace_tool():
    from api.mcp_server import create_mcp_tools

    trace = PipelineTrace(trace_id="tr_exp", timestamp="T", query="q")
    svc = MagicMock()
    svc.get_trace.return_value = trace

    tools = create_mcp_tools(svc)
    result = tools["explain_trace"]("tr_exp")
    assert result["trace_id"] == "tr_exp"


def test_explain_trace_not_found():
    from api.mcp_server import create_mcp_tools

    svc = MagicMock()
    svc.get_trace.return_value = None

    tools = create_mcp_tools(svc)
    result = tools["explain_trace"]("nonexistent")
    assert "error" in result
```

**Step 2: Run tests — verify they fail**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_mcp_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.mcp_server'`

**Step 3: Create MCP server module**

Create `api/mcp_server.py`:

```python
"""MCP server tools for Agentic Graph RAG.

Provides 3 tools: resolve_intent, search_graph, explain_trace.
Tools are functions that can be registered with FastMCP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_graph_rag.service import PipelineService


def create_mcp_tools(service: PipelineService) -> dict:
    """Create MCP tool functions bound to a PipelineService instance.

    Returns a dict of {tool_name: callable} for registration with FastMCP
    or for direct testing.
    """

    def resolve_intent(query: str, mode: str = "agent_pattern") -> dict:
        """Resolve user query via Agentic Graph RAG pipeline."""
        qa = service.query(query, mode=mode)
        return qa.model_dump()

    def search_graph(query: str, tool: str = "vector_search") -> dict:
        """Search the knowledge graph without answer generation."""
        from agentic_graph_rag.agent.tools import _TOOL_REGISTRY  # noqa: PLC2701

        if tool not in ("vector_search", "cypher_traverse", "hybrid_search",
                        "comprehensive_search", "temporal_query", "full_document_read"):
            return {"error": f"Unknown tool: {tool}"}

        qa = service.query(query, mode="agent_pattern")
        return {
            "results": [r.model_dump() for r in qa.sources],
            "trace": qa.trace.model_dump() if qa.trace else None,
        }

    def explain_trace(trace_id: str) -> dict:
        """Get provenance trace by ID."""
        trace = service.get_trace(trace_id)
        if trace is None:
            return {"error": f"Trace {trace_id} not found"}
        return trace.model_dump()

    return {
        "resolve_intent": resolve_intent,
        "search_graph": search_graph,
        "explain_trace": explain_trace,
    }


def mount_mcp(app, service: PipelineService):
    """Mount FastMCP server on a FastAPI/Starlette app.

    Uses SSE transport at /mcp/sse.
    """
    try:
        from fastmcp import FastMCP

        mcp = FastMCP("Agentic Graph RAG")
        tools = create_mcp_tools(service)

        @mcp.tool()
        def resolve_intent(query: str, mode: str = "agent_pattern") -> dict:
            """Resolve user query via Agentic Graph RAG pipeline.
            Returns answer with full provenance trace."""
            return tools["resolve_intent"](query, mode)

        @mcp.tool()
        def search_graph(query: str, tool: str = "vector_search") -> dict:
            """Search the knowledge graph. Tools: vector_search, cypher_traverse,
            hybrid_search, comprehensive_search, temporal_query, full_document_read."""
            return tools["search_graph"](query, tool)

        @mcp.tool()
        def explain_trace(trace_id: str) -> dict:
            """Get full provenance trace by trace ID."""
            return tools["explain_trace"](trace_id)

        mcp.mount(app, path="/mcp")

    except ImportError:
        import logging
        logging.getLogger(__name__).warning("fastmcp not installed — MCP server disabled")
```

**Step 4: Run tests — verify pass**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_mcp_tools.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add api/mcp_server.py tests/unit/test_mcp_tools.py
git commit -m "feat(mcp): add MCP server tools — resolve_intent, search_graph, explain_trace"
```

---

## Task 8: Wire MCP into FastAPI app

**Files:**
- Modify: `api/app.py` (add MCP mount in lifespan)

**Step 1: Add MCP mount to app.py**

In `api/app.py`, after `app.include_router(router)`, add:

```python
    # Mount MCP server (if fastmcp available)
    if service is not None:
        from api.mcp_server import mount_mcp
        mount_mcp(app, service)

    return app
```

And in the production lifespan branch, after `set_service(svc)`:
```python
            from api.mcp_server import mount_mcp
            mount_mcp(app, svc)
```

**Step 2: Run all API + MCP tests**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/unit/test_api_routes.py tests/unit/test_mcp_tools.py -v`
Expected: 9 PASSED

**Step 3: Commit**

```bash
git add api/app.py
git commit -m "feat(api): wire MCP server into FastAPI app"
```

---

## Task 9: API launcher script

**Files:**
- Create: `run_api.py`

**Step 1: Create launcher**

Create `run_api.py`:

```python
#!/usr/bin/env python3
"""Launch the Agentic Graph RAG API server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8507,
        reload=False,
        log_level="info",
    )
```

**Step 2: Test launch (smoke test)**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle python run_api.py &` then `sleep 3 && curl -s http://localhost:8507/api/v1/health | head -20` then stop.

Note: This requires Neo4j running. If not available, skip smoke test — unit tests already cover routes.

**Step 3: Commit**

```bash
git add run_api.py
git commit -m "feat(api): add API launcher script (port 8507)"
```

---

## Task 10: Streamlit → thin client (Search tab)

**Files:**
- Modify: `ui/streamlit_app.py:256-300` (Search & Q&A tab)
- Modify: `ui/streamlit_app.py:380-410` (Agent Trace tab)

**Step 1: Add API_URL config at top of streamlit_app.py**

Near the top of the file, after imports:
```python
import httpx

API_URL = os.environ.get("AGR_API_URL", "http://localhost:8507")
```

**Step 2: Replace direct Python calls in Search tab**

Replace the `with st.spinner(...)` block (lines ~262-284) with:

```python
            with st.spinner(t("search_thinking")):
                try:
                    resp = httpx.post(
                        f"{API_URL}/api/v1/query",
                        json={"text": query, "mode": api_mode, "lang": lang},
                        timeout=120.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.ConnectError:
                    st.error("API server not running. Start with: python run_api.py")
                    st.stop()
```

Where `api_mode` is mapped from UI mode selection:
```python
            mode_map = {
                t("search_mode_vector"): "vector",
                t("search_mode_hybrid"): "hybrid",
                t("search_mode_agent"): "agent_pattern",
            }
            api_mode = mode_map.get(mode, "agent_pattern")
            if use_mangle_router:
                api_mode = "agent_mangle"
            elif use_llm_router:
                api_mode = "agent_llm"
```

Then parse response into QAResult:
```python
            from rag_core.models import QAResult
            qa = QAResult.model_validate(data)
            st.session_state.last_qa = qa
            st.session_state.last_trace = data.get("trace")
```

**Step 3: Update Agent Trace tab to render PipelineTrace**

Replace the Agent Trace tab (lines ~380-410) with:

```python
with tab_trace:
    st.header(t("trace_header"))

    trace_data = st.session_state.get("last_trace")
    if trace_data is None:
        st.info(t("trace_no_data"))
    else:
        # Router decision
        if trace_data.get("router_step"):
            st.subheader(t("trace_routing"))
            rs = trace_data["router_step"]
            d = rs.get("decision", {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Method", rs.get("method", "—"))
            with col2:
                st.metric(t("trace_query_type"), d.get("query_type", "—"))
            with col3:
                st.metric(t("trace_confidence"), f"{d.get('confidence', 0):.0%}")
            with col4:
                st.metric(t("trace_tool"), d.get("suggested_tool", "—"))
            if rs.get("rules_fired"):
                st.caption(f"Rules: {', '.join(rs['rules_fired'])}")
            st.caption(f"{t('trace_reasoning')}: {d.get('reasoning', '')}")
            st.caption(f"Duration: {rs.get('duration_ms', 0)}ms")

        # Tool steps
        if trace_data.get("tool_steps"):
            st.divider()
            st.subheader("Tool Steps")
            for i, step in enumerate(trace_data["tool_steps"], 1):
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(f"Step {i}", step["tool_name"])
                    with c2:
                        st.metric("Results", step.get("results_count", 0))
                    with c3:
                        score = step.get("relevance_score", 0)
                        st.metric("Relevance", f"{score:.1f}/5.0")
                    with c4:
                        st.metric("Duration", f"{step.get('duration_ms', 0)}ms")

        # Escalation steps
        if trace_data.get("escalation_steps"):
            st.divider()
            st.subheader("Escalations")
            for esc in trace_data["escalation_steps"]:
                st.warning(
                    f"**{esc['from_tool']}** → **{esc['to_tool']}**: {esc.get('reason', '')}"
                )

        # Generator step
        if trace_data.get("generator_step"):
            st.divider()
            st.subheader("Generator")
            gs = trace_data["generator_step"]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Model", gs.get("model", "—"))
            with c2:
                tokens = gs.get("prompt_tokens", 0) + gs.get("completion_tokens", 0)
                st.metric("Tokens", tokens)
            with c3:
                st.metric("Confidence", f"{gs.get('confidence', 0):.0%}")

        # Summary
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Duration", f"{trace_data.get('total_duration_ms', 0)}ms")
        with c2:
            st.metric("Trace ID", trace_data.get("trace_id", "—"))

        # Raw JSON (expandable)
        with st.expander("Raw Trace JSON"):
            st.json(trace_data)
```

**Step 4: Verify manually**

Start API: `python run_api.py`
Start Streamlit: `streamlit run ui/streamlit_app.py --server.port 8506`
Test a query and check Agent Trace tab.

**Step 5: Commit**

```bash
git add ui/streamlit_app.py
git commit -m "feat(ui): Streamlit as thin client via httpx, enhanced Agent Trace tab"
```

---

## Task 11: Run full test suite + benchmark regression

**Step 1: Run all unit tests**

Run: `PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle pytest tests/ -v --tb=short`
Expected: ~420+ passed (398 existing + ~22 new)

**Step 2: Run benchmark regression (requires Neo4j)**

If Neo4j is running:
Run benchmark for vector mode to verify no regression:
```bash
PYTHONPATH=/home/vladspace_ubuntu24/agentic-graph-rag:/home/vladspace_ubuntu24/agentic-graph-rag/pymangle python -m benchmark.runner --modes vector --lang ru
```
Expected: vector = 11/15 (same as v5)

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: test suite adjustments for v6"
```

---

## Task 12: Update README and project memory

**Files:**
- Modify: `README.md` (add API section)

**Step 1: Add API section to README**

Add section documenting:
- API endpoints (table)
- MCP server setup for Claude Code
- How to start API (`python run_api.py`)
- Example curl commands

**Step 2: Update project MEMORY.md**

Add v6 section with key changes and benchmark results.

**Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: update README with v6 API endpoints and MCP setup"
```

---

## Summary

| Task | Files | Tests | Description |
|------|-------|-------|-------------|
| 1 | requirements.txt | — | Add dependencies |
| 2 | models.py | 8 | Provenance models |
| 3 | retrieval_agent.py | 3 | Instrument pipeline with trace |
| 4 | generator.py, models.py | 2 | Token usage capture |
| 5 | service.py | 4 | PipelineService contract |
| 6 | api/*.py | 6 | FastAPI REST endpoints |
| 7 | api/mcp_server.py | 3 | MCP server tools |
| 8 | api/app.py | — | Wire MCP into app |
| 9 | run_api.py | — | Launcher script |
| 10 | streamlit_app.py | — | Thin client + enhanced trace UI |
| 11 | — | — | Full test suite + regression |
| 12 | README.md | — | Documentation |
| **Total** | **~12 files** | **~26 new** | **~450 total tests** |
