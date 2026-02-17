"""Tests for provenance models."""
import json

from rag_core.models import (
    EscalationStep,
    GeneratorStep,
    PipelineTrace,
    QAResult,
    QueryType,
    RouterDecision,
    RouterStep,
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
