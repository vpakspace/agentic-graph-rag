"""Unified data models for Agentic Graph RAG."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ingestion models (from RAG 2.0)
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A text chunk with optional contextual enrichment and embedding."""

    id: str = ""
    content: str
    context: str = ""
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def enriched_content(self) -> str:
        if self.context:
            return f"{self.context}\n\n{self.content}"
        return self.content


# ---------------------------------------------------------------------------
# Knowledge Graph models (from TKB)
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """An entity extracted from text."""

    id: str = ""
    name: str
    entity_type: str = ""
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    """A relationship between two entities."""

    id: str = ""
    source: str
    target: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemporalEvent(BaseModel):
    """A temporal event from the knowledge graph."""

    id: str = ""
    content: str
    valid_from: str = ""
    valid_to: str = ""
    entity_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph RAG models (NEW — KET-RAG / HippoRAG 2)
# ---------------------------------------------------------------------------

class PhraseNode(BaseModel):
    """Entity-level node for graph navigation (HippoRAG 2)."""

    id: str = ""
    name: str
    entity_type: str = ""
    pagerank_score: float = 0.0
    passage_ids: list[str] = Field(default_factory=list)


class PassageNode(BaseModel):
    """Full-text passage node for context preservation (HippoRAG 2)."""

    id: str = ""
    text: str
    chunk_id: str = ""
    embedding: list[float] = Field(default_factory=list)
    phrase_ids: list[str] = Field(default_factory=list)


class GraphContext(BaseModel):
    """Assembled context from graph traversal."""

    triplets: list[dict[str, str]] = Field(default_factory=list)
    passages: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Router + retrieval models (NEW)
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    """Query complexity categories for the agentic router."""

    SIMPLE = "simple"
    RELATION = "relation"
    MULTI_HOP = "multi_hop"
    GLOBAL = "global"
    TEMPORAL = "temporal"


class RouterDecision(BaseModel):
    """Output of the query router."""

    query_type: QueryType
    confidence: float = 0.0
    reasoning: str = ""
    suggested_tool: str = ""


class SearchResult(BaseModel):
    """A single search result from vector store or graph."""

    chunk: Chunk
    score: float = 0.0
    rank: int = 0
    source: str = "vector"  # "vector", "graph", "hybrid"


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


class QAResult(BaseModel):
    """Final Q&A result with answer, sources, and confidence."""

    answer: str
    sources: list[SearchResult] = Field(default_factory=list)
    confidence: float = 0.0
    query: str = ""
    expanded_query: str = ""
    retries: int = 0
    router_decision: RouterDecision | None = None
    graph_context: GraphContext | None = None
    trace: PipelineTrace | None = None  # v6 provenance
