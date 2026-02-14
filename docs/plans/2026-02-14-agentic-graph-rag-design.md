# Agentic Graph RAG — Design Document

**Date**: 2026-02-14
**Status**: Approved
**Source**: Alexander Shereshevsky — "Graph RAG in 2026: A Practitioner's Guide" (Medium, Feb 2026)

---

## Goals

1. **Phase 1** (current): Production tool — полная реализация 4 этапов Agentic Graph RAG
2. **Phase 2** (future): Объединить все RAG проекты (rag-2.0, rag-temporal, pageindex, temporal-kb)
3. **Phase 3** (future): Бенчмарк + статья (сравнение vector vs graph vs hybrid vs agentic)

## Architecture Decisions

- **Monorepo** with shared `rag-core` pip package
- **Standalone Neo4j** (reuse `temporal-kb-neo4j` Docker container)
- **All 4 stages**: Skeleton indexing + VectorCypher + Agentic Router + Optimization
- **Test data**: Russian PDFs + English papers + custom documents

## Project Structure

```
~/agentic-graph-rag/
├── packages/
│   └── rag-core/                    # Shared pip package
│       ├── rag_core/
│       │   ├── config.py            # ← TKB (nested Pydantic Settings)
│       │   ├── models.py            # ← Merged (Chunk, Entity, SearchResult, QAResult)
│       │   ├── loader.py            # ← TKB (Docling: PDF/DOCX/PPTX + GPU)
│       │   ├── chunker.py           # ← TKB (table-aware, sanitize_for_graphiti)
│       │   ├── enricher.py          # ← RAG 2.0 (contextual enrichment via OpenAI)
│       │   ├── embedder.py          # ← RAG 2.0 (text-embedding-3-small, batch)
│       │   ├── vector_store.py      # ← RAG 2.0 (Neo4j Vector Index)
│       │   ├── kg_client.py         # ← rag-temporal (Graphiti wrapper + Cypher)
│       │   ├── query_expander.py    # ← RAG 2.0 (multi-query expansion)
│       │   ├── reranker.py          # ← RAG 2.0 (cosine reranking)
│       │   ├── generator.py         # ← RAG 2.0 (LLM answer generation)
│       │   ├── reflector.py         # ← RAG 2.0 (relevance eval + retry)
│       │   └── i18n.py              # ← TKB (~180 keys, RU/EN)
│       ├── pyproject.toml
│       └── tests/
│
├── agentic_graph_rag/               # === NEW CODE ===
│   ├── indexing/
│   │   ├── skeleton.py              # KET-RAG: KNN-graph → PageRank → top-β extraction
│   │   └── dual_node.py             # HippoRAG 2: phrase nodes + passage nodes + PPR
│   ├── retrieval/
│   │   └── vector_cypher.py         # VectorCypher: vector entry → Cypher traversal → context
│   ├── agent/
│   │   ├── router.py                # Query complexity classifier
│   │   ├── retrieval_agent.py       # Agentic orchestrator + self-correction loop
│   │   └── tools.py                 # 6 tools
│   ├── generation/
│   │   └── graph_verifier.py        # Contradiction detection via graph
│   └── optimization/
│       ├── cache.py                 # Subgraph + community cache
│       └── monitor.py               # Query type stats + PageRank tuning
│
├── ui/
│   ├── streamlit_app.py             # 6 tabs
│   └── components/
│
├── benchmark/
│   ├── questions.json
│   ├── runner.py
│   └── compare.py
│
├── data/
├── tests/
├── .github/workflows/ci.yml
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Data Flow

### Ingestion
```
Document → Loader (Docling) → Chunker (table-aware) → Enricher (contextual)
    → Embedder (OpenAI) → Skeleton Indexer (KET-RAG: PageRank top-β)
    → Dual Node Builder (HippoRAG 2: phrase + passage nodes)
    → Neo4j (Vector Index + KG + Passage Nodes)
```

### Retrieval
```
Query → Router (classify complexity)
    → simple:    Vector Search only
    → relation:  VectorCypher (vector entry → Cypher depth=3)
    → multi_hop: Hybrid Pipeline (vector + KG merge)
    → global:    Community Search (graph communities)
    → temporal:  Graphiti temporal query
    → Self-Correction Loop (eval relevance, expand if < 3.0, check contradictions)
    → Graph Verifier (cross-check via traversal)
    → Generator (GPT-4o, citations)
```

## New Components (not in existing projects)

### Skeleton Indexer (KET-RAG, KDD 2025)
- KNN-graph from chunk embeddings
- PageRank → top 20-30% "skeletal" chunks
- Skeletal: full LLM extraction (entities + relationships)
- Peripheral: cheap keyword-based links (no LLM)
- Result: 10x cost reduction, +32.4% generation quality

### Dual Node Builder (HippoRAG 2, ICML 2025)
- Phrase Nodes: entities for graph navigation
- Passage Nodes: full text for context preservation
- `(PhraseNode)-[:MENTIONED_IN]->(PassageNode)`
- Personalized PageRank (PPR) for balance
- F1 +7.1 on MuSiQue, 12x fewer tokens

### VectorCypher Retrieval
- Step 1: Vector search entry points (score > 0.7)
- Step 2: Cypher traversal (depth 2-3, apoc.path.subgraphAll)
- Step 3: Context assembly (triplets + passage nodes)

### Query Router
| Type | Strategy | Delta vs Vector |
|------|----------|----------------|
| simple | Vector only | -13.4% if graph |
| relation | VectorCypher depth=3 | +25-40% |
| multi_hop | Hybrid Pipeline | +4.5-20% |
| global | Community Search | +15-30% |
| temporal | Graphiti temporal | needs temporal KG |

### Self-Correction Loop
- Evaluate relevance (1-5 score)
- If < 3.0 → expand traversal depth, retry
- Check contradictions across graph paths
- Max 2 retries

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Embeddings | text-embedding-3-small (1536 dim) |
| Graph DB | Neo4j 5.x (temporal-kb-neo4j) |
| Temporal KG | graphiti-core >=0.26.0 |
| Doc parsing | Docling >=2.0.0 + GPU |
| UI | Streamlit >=1.30.0 |
| Config | Pydantic Settings v2 |
| Python | 3.12+ |

## Streamlit UI (6 tabs)

1. **Ingest** — upload, skeleton indexing progress, stats (chunks/entities/nodes)
2. **Search & Q&A** — mode selector (Vector/VectorCypher/Agent), confidence, sources
3. **Graph Explorer** — phrase + passage nodes visualization, relationship browsing
4. **Agent Trace** — routing decision, self-correction steps, tool calls
5. **Benchmark** — 5-mode comparison, PASS/FAIL table, delta metrics
6. **Settings** — config, cache stats, PageRank weights, clear DB

## Benchmark

5 modes on same data:
- Vector Only (baseline)
- VectorCypher (hybrid)
- Agentic (router + self-correction)
- Agentic + Skeleton (full pipeline)
- RAG 2.0 (comparison with existing)

Metrics: Accuracy, F1, Latency (P50/P95), Cost (tokens)

## Testing

- Unit tests: 80+ (rag-core modules + new components)
- Integration tests: 15+ (pipeline e2e, agent routing)
- Coverage target: 80%+
- CI: GitHub Actions (pytest + ruff, Python 3.12)

## Key Papers

| Paper | Venue | Contribution |
|-------|-------|-------------|
| KET-RAG | KDD 2025 | Cost-efficient indexing (10x) |
| HippoRAG 2 | ICML 2025 | Dual-node + PPR |
| GraphRAG-Bench | ICLR 2026 | When to use graph |
| T²RAG | ICLR 2026 | Triplet search without graph |
| Graphiti (Zep) | arXiv 2025 | Temporal KG + agent memory |
| DIGIMON | arXiv 2025 | Unified RAG methods |

## Source Modules (reuse map)

| Module | Source Project | Notes |
|--------|---------------|-------|
| config.py | Temporal-KB | Nested, async-ready |
| models.py | Combined | Chunk + Entity + SearchResult |
| loader.py | Temporal-KB | Docling full-featured |
| chunker.py | Temporal-KB | Table-aware, sanitize |
| enricher.py | RAG 2.0 | Contextual enrichment |
| embedder.py | RAG 2.0 | Batch embeddings |
| vector_store.py | RAG 2.0 | Neo4j Vector Index |
| kg_client.py | RAG-Temporal | Graphiti wrapper |
| query_expander.py | RAG 2.0 | Multi-query expansion |
| reranker.py | RAG 2.0 | Cosine reranking |
| generator.py | RAG 2.0 | LLM answer |
| reflector.py | RAG 2.0 | Relevance eval + retry |
| i18n.py | Temporal-KB | ~180 keys, RU/EN |
