"""Tests for rag_core.models."""

import pytest

from rag_core.models import (
    Chunk,
    Entity,
    GraphContext,
    PassageNode,
    PhraseNode,
    QAResult,
    QueryType,
    Relationship,
    RouterDecision,
    SearchResult,
    TemporalEvent,
)


class TestChunk:
    def test_basic_creation(self):
        c = Chunk(content="hello world")
        assert c.content == "hello world"
        assert c.id == ""
        assert c.context == ""
        assert c.embedding == []
        assert c.metadata == {}

    def test_enriched_content_without_context(self):
        c = Chunk(content="hello")
        assert c.enriched_content == "hello"

    def test_enriched_content_with_context(self):
        c = Chunk(content="hello", context="prefix")
        assert c.enriched_content == "prefix\n\nhello"

    def test_serialization(self):
        c = Chunk(id="c1", content="text", embedding=[0.1, 0.2])
        d = c.model_dump()
        assert d["id"] == "c1"
        assert d["embedding"] == [0.1, 0.2]
        c2 = Chunk.model_validate(d)
        assert c2 == c


class TestEntity:
    def test_creation(self):
        e = Entity(name="Neo4j", entity_type="Technology")
        assert e.name == "Neo4j"
        assert e.entity_type == "Technology"

    def test_serialization(self):
        e = Entity(id="e1", name="Test", description="desc")
        d = e.model_dump()
        e2 = Entity.model_validate(d)
        assert e2 == e


class TestRelationship:
    def test_creation(self):
        r = Relationship(source="e1", target="e2", relation_type="USES")
        assert r.source == "e1"
        assert r.weight == 1.0

    def test_serialization(self):
        r = Relationship(source="a", target="b", relation_type="X", weight=0.5)
        d = r.model_dump()
        r2 = Relationship.model_validate(d)
        assert r2 == r


class TestTemporalEvent:
    def test_creation(self):
        t = TemporalEvent(content="event happened", valid_from="2025-01-01")
        assert t.content == "event happened"
        assert t.valid_from == "2025-01-01"


class TestPhraseNode:
    def test_creation(self):
        p = PhraseNode(name="GraphRAG", pagerank_score=0.85)
        assert p.name == "GraphRAG"
        assert p.pagerank_score == 0.85
        assert p.passage_ids == []

    def test_with_passages(self):
        p = PhraseNode(name="X", passage_ids=["p1", "p2"])
        assert len(p.passage_ids) == 2


class TestPassageNode:
    def test_creation(self):
        p = PassageNode(text="full text here", chunk_id="c1")
        assert p.text == "full text here"
        assert p.phrase_ids == []
        assert p.embedding == []


class TestGraphContext:
    def test_empty(self):
        g = GraphContext()
        assert g.triplets == []
        assert g.passages == []

    def test_with_data(self):
        g = GraphContext(
            triplets=[{"s": "A", "p": "USES", "o": "B"}],
            passages=["some text"],
            source_ids=["s1"],
        )
        assert len(g.triplets) == 1
        assert g.passages[0] == "some text"


class TestQueryType:
    def test_values(self):
        assert QueryType.SIMPLE == "simple"
        assert QueryType.RELATION == "relation"
        assert QueryType.MULTI_HOP == "multi_hop"
        assert QueryType.GLOBAL == "global"
        assert QueryType.TEMPORAL == "temporal"


class TestRouterDecision:
    def test_creation(self):
        d = RouterDecision(
            query_type=QueryType.RELATION,
            confidence=0.9,
            reasoning="contains relationship keywords",
            suggested_tool="cypher_traverse",
        )
        assert d.query_type == QueryType.RELATION
        assert d.confidence == 0.9

    def test_serialization(self):
        d = RouterDecision(query_type=QueryType.SIMPLE)
        data = d.model_dump()
        d2 = RouterDecision.model_validate(data)
        assert d2.query_type == QueryType.SIMPLE


class TestSearchResult:
    def test_creation(self):
        c = Chunk(content="test")
        sr = SearchResult(chunk=c, score=0.95, rank=1, source="graph")
        assert sr.score == 0.95
        assert sr.source == "graph"

    def test_default_source(self):
        sr = SearchResult(chunk=Chunk(content="x"))
        assert sr.source == "vector"


class TestQAResult:
    def test_basic(self):
        qa = QAResult(answer="42", query="what is the answer?")
        assert qa.answer == "42"
        assert qa.sources == []
        assert qa.retries == 0
        assert qa.router_decision is None
        assert qa.graph_context is None

    def test_with_router_and_graph(self):
        qa = QAResult(
            answer="answer",
            router_decision=RouterDecision(query_type=QueryType.MULTI_HOP),
            graph_context=GraphContext(passages=["p1"]),
        )
        assert qa.router_decision.query_type == QueryType.MULTI_HOP
        assert len(qa.graph_context.passages) == 1

    def test_full_serialization(self):
        qa = QAResult(
            answer="a",
            query="q",
            confidence=0.8,
            retries=1,
            sources=[SearchResult(chunk=Chunk(content="c"), score=0.9)],
            router_decision=RouterDecision(query_type=QueryType.GLOBAL),
        )
        d = qa.model_dump()
        qa2 = QAResult.model_validate(d)
        assert qa2.answer == "a"
        assert qa2.sources[0].score == 0.9
        assert qa2.router_decision.query_type == QueryType.GLOBAL
