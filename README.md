# Agentic Graph RAG

**Skeleton Indexing + VectorCypher + Agentic Router with Self-Correction + Mangle Reasoning + Typed API**

A production-ready Graph RAG system combining four cutting-edge techniques from recent research into a unified retrieval pipeline with declarative reasoning, full pipeline provenance, and a typed API contract (FastAPI REST + MCP).

## Benchmark Results (v5)

Evaluated on 15 bilingual questions (5 types) across 6 retrieval modes:

| Mode | Accuracy | Description |
|------|----------|-------------|
| **Agent (pattern)** | **12/15 (80%)** | Auto-routing via regex patterns + self-correction |
| **Agent (Mangle)** | **12/15 (80%)** | Declarative Datalog rule routing + self-correction |
| **Vector** | **11/15 (73%)** | Embedding similarity search |
| **Hybrid** | **11/15 (73%)** | Vector + Graph with RRF fusion |
| **Cypher** | **10/15 (67%)** | Graph traversal via VectorCypher (3-hop) |
| **Agent (LLM)** | **10/15 (67%)** | Auto-routing via GPT-4o-mini |
| **Overall** | **66/90 (73%)** | |

Accuracy by query type (best mode per type):

| Type | Best Mode | Accuracy |
|------|-----------|----------|
| simple | vector, hybrid, agent_pattern, agent_mangle | 100% |
| multi_hop | all modes | 100% |
| relation | vector, hybrid, agent_pattern, agent_llm, agent_mangle | 100% |
| temporal | agent_mangle | 100% |
| global | agent_pattern | 67% |

<details>
<summary>Benchmark history (v3 → v4 → v5)</summary>

| Version | Overall | Key Changes |
|---------|---------|-------------|
| v3 | 34/90 (38%) | Baseline (lang=en, pre-improvements) |
| v4 | 60/90 (67%) | lang=ru, cosine ranking, synthesis prompt, temporal boost |
| v5 | 66/90 (73%) | comprehensive_search, completeness check, retry query, max_hops=3 |

</details>

## Key Techniques

| Technique | Source | What It Does |
|-----------|--------|-------------|
| **Skeleton Indexing** | KET-RAG (KDD 2025) | KNN graph -> PageRank -> selective entity extraction (10x cost savings) |
| **Dual Node Structure** | HippoRAG 2 (ICML 2025) | Phrase nodes + passage nodes + Personalized PageRank |
| **VectorCypher** | Neo4j / GraphRAG | Vector entry points -> Cypher traversal -> context assembly |
| **Agentic Router** | Custom | Pattern/LLM/Mangle classification -> tool selection -> self-correction loop |
| **Mangle Reasoning** | PyMangle (Datalog) | Declarative routing rules with bilingual keyword matching (65 keywords) |

## Architecture

```
Ingestion:
  Document -> Docling -> Chunker -> Enricher -> Embedder
           -> Skeleton Indexer (PageRank top-B)
           -> Dual Node Builder (phrase + passage nodes)
           -> Neo4j (Vector Index + Knowledge Graph)

Retrieval:
  Query -> Router (simple/relation/multi_hop/global/temporal)
           Router cascade: Mangle -> LLM -> Pattern fallback
        -> Tool Selection (vector/cypher/hybrid/comprehensive/full_read/temporal)
        -> Self-Correction Loop (evaluate relevance -> escalate if <2.0)
        -> Graph Verifier (contradiction detection)
        -> Generator (GPT-4o synthesis + dynamic confidence + citations)
```

## Project Structure

```
agentic-graph-rag/
├── packages/rag-core/        # Shared pip package (models, config, ingestion, retrieval)
│   └── rag_core/
│       ├── models.py          # Chunk, Entity, SearchResult, QAResult, QueryType
│       ├── config.py          # Pydantic Settings (nested: Neo4j, OpenAI, Indexing, Agent)
│       ├── loader.py          # Docling: PDF/DOCX/PPTX + GPU
│       ├── chunker.py         # Table-aware chunking
│       ├── enricher.py        # Contextual enrichment (OpenAI)
│       ├── embedder.py        # text-embedding-3-small batch processing
│       ├── vector_store.py    # Neo4j Vector Index CRUD
│       ├── kg_client.py       # Graphiti wrapper + Cypher
│       ├── generator.py       # LLM answer synthesis + dynamic confidence
│       ├── reflector.py       # Relevance evaluation (1-5 scale) + completeness check
│       └── i18n.py            # RU/EN localization (~50 keys)
│
├── agentic_graph_rag/         # Graph RAG components
│   ├── indexing/
│   │   ├── skeleton.py        # KET-RAG: KNN -> PageRank -> skeletal extraction
│   │   └── dual_node.py       # HippoRAG 2: phrase + passage nodes + PPR
│   ├── retrieval/
│   │   └── vector_cypher.py   # Vector entry -> Cypher traversal -> context
│   ├── agent/
│   │   ├── router.py          # Query classifier (pattern + LLM + Mangle)
│   │   ├── retrieval_agent.py # Orchestrator + self-correction loop + provenance
│   │   └── tools.py           # 7 tools: vector, cypher, community, hybrid, temporal, full_read, comprehensive
│   ├── generation/
│   │   └── graph_verifier.py  # Contradiction detection + claim verification
│   ├── reasoning/
│   │   ├── reasoning_engine.py # PyMangle Datalog engine wrapper
│   │   └── rules/             # Mangle rules: routing.mg, access.mg, graph.mg
│   ├── optimization/
│   │   ├── cache.py           # LRU SubgraphCache + CommunityCache
│   │   └── monitor.py         # QueryMonitor + PageRank tuning suggestions
│   └── service.py             # PipelineService — typed internal contract
│
├── api/                       # v6: Typed API contract
│   ├── app.py                 # FastAPI factory with lifespan
│   ├── routes.py              # REST endpoints (query, trace, health, graph_stats)
│   ├── deps.py                # Dependency injection (PipelineService singleton)
│   └── mcp_server.py          # MCP tools (resolve_intent, search_graph, explain_trace)
│
├── pymangle/                  # PyMangle Datalog engine (~5K lines)
│
├── ui/
│   └── streamlit_app.py       # 7-tab Streamlit UI (port 8506, httpx thin client)
│
├── benchmark/
│   ├── questions.json         # 15 test questions (5 types, EN/RU)
│   ├── runner.py              # 6-mode benchmark runner
│   └── compare.py             # Comparison table generator
│
├── run_api.py                 # API launcher (uvicorn, port 8507)
└── tests/                     # 377 unit tests (269 core + 108 pymangle)
```

## Quick Start

### Prerequisites

- Python 3.12+
- Neo4j 5.x (Docker recommended)
- OpenAI API key

### Installation

```bash
# Clone
git clone https://github.com/vpakspace/agentic-graph-rag.git
cd agentic-graph-rag

# Install
pip install -e packages/rag-core --no-deps
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your OpenAI API key and Neo4j credentials
```

### Start Neo4j

```bash
docker run -d \
  --name agentic-graph-rag-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5
```

### Run Tests

```bash
PYTHONPATH=.:pymangle pytest tests/ pymangle/ -x -q  # 377 tests, ~4 seconds
```

### Run API Server (v6)

```bash
PYTHONPATH=.:pymangle python run_api.py  # http://localhost:8507
```

REST endpoints:
- `POST /api/v1/query` — Query the pipeline (returns answer + trace)
- `GET /api/v1/trace/{id}` — Retrieve a pipeline trace by ID
- `GET /api/v1/health` — Health check (Neo4j connectivity)
- `GET /api/v1/graph/stats` — Graph node/edge statistics

MCP tools (SSE at `/mcp`):
- `resolve_intent` — Classify query type and select tool
- `search_graph` — Execute full pipeline search
- `explain_trace` — Explain a pipeline trace

### Run Streamlit UI

```bash
# With API backend (recommended):
AGR_API_URL=http://localhost:8507 PYTHONPATH=.:pymangle streamlit run ui/streamlit_app.py --server.port 8506

# Standalone (direct Python, no API needed):
PYTHONPATH=.:pymangle streamlit run ui/streamlit_app.py --server.port 8506
```

### Run Benchmark

```python
from benchmark.runner import run_benchmark
from benchmark.compare import compare_modes

# Requires Neo4j running + documents ingested
results = run_benchmark(driver, openai_client)  # lang="ru" by default
print(compare_modes(results))
```

## Retrieval Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Vector** | Cosine similarity on embeddings | Simple factual queries |
| **Cypher** | Graph traversal via VectorCypher | Relationship queries |
| **Hybrid** | Vector + Graph with RRF fusion | Multi-hop queries |
| **Agent (pattern)** | Auto-routing via regex patterns | General use (fast) |
| **Agent (LLM)** | Auto-routing via GPT-4o-mini | General use (accurate) |
| **Agent (Mangle)** | Auto-routing via Datalog rules | General use (deterministic) |

## Pipeline Provenance (v6)

Every query produces a `PipelineTrace` — a structured record of the full pipeline execution:

```json
{
  "trace_id": "tr_abc123def456",
  "timestamp": "2026-02-17T12:00:00Z",
  "query": "Какие методы используются?",
  "router_step": {"method": "mangle", "decision": {"query_type": "simple", "suggested_tool": "vector_search"}},
  "tool_steps": [{"tool_name": "vector_search", "results_count": 10, "relevance_score": 3.2, "duration_ms": 150}],
  "escalation_steps": [],
  "generator_step": {"model": "gpt-4o-mini", "prompt_tokens": 1200, "completion_tokens": 350, "confidence": 0.82},
  "total_duration_ms": 1800
}
```

Traces are cached (LRU, 100 entries) and retrievable via `GET /api/v1/trace/{id}` or the MCP `explain_trace` tool.

## Self-Correction Loop

The agent evaluates retrieval quality (1-5 scale) and escalates through the tool chain when results are insufficient:

```
vector_search -> cypher_traverse -> hybrid_search -> comprehensive_search -> full_document_read
```

Each retry rephrases the query via LLM and uses a different tool with expanded search scope. For GLOBAL queries, a completeness check triggers additional retrieval if the answer is incomplete. Max 2 retries by default. Best results tracked across attempts.

## Mangle Reasoning

Declarative routing via PyMangle (Python Datalog engine):

- **65 bilingual keywords** (Russian + English) in `routing.mg`
- **Router cascade**: Mangle (confidence 0.7) -> LLM (0.85) -> Pattern (0.5)
- **UI**: "Mangle Router" checkbox in sidebar, Reasoning tab for rule testing
- **Rules**: `agentic_graph_rag/reasoning/rules/` — routing, access control, graph traversal

## Configuration

All settings via `.env` or environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `INDEXING_SKELETON_BETA` | `0.25` | Fraction of chunks for full extraction |
| `INDEXING_KNN_K` | `10` | KNN graph neighbors |
| `INDEXING_PAGERANK_DAMPING` | `0.85` | PageRank damping factor |
| `RETRIEVAL_TOP_K_VECTOR` | `10` | Vector search results count |
| `RETRIEVAL_TOP_K_FINAL` | `10` | Final results after fusion |
| `RETRIEVAL_VECTOR_THRESHOLD` | `0.5` | Minimum similarity score |
| `RETRIEVAL_MAX_HOPS` | `3` | Max graph traversal depth |
| `AGENT_MAX_RETRIES` | `2` | Self-correction retries |
| `AGENT_RELEVANCE_THRESHOLD` | `2.0` | Minimum relevance score (1-5) |

## Tech Stack

- **LLM**: OpenAI GPT-4o / GPT-4o-mini
- **Embeddings**: text-embedding-3-small (1536 dim)
- **Graph DB**: Neo4j 5.x (Vector Index + Cypher)
- **Reasoning**: PyMangle (Datalog engine)
- **Doc Parsing**: Docling (PDF/DOCX/PPTX + GPU)
- **Graph Algorithms**: NetworkX (PageRank, KNN, PPR)
- **API**: FastAPI (REST + MCP via FastMCP)
- **UI**: Streamlit (7 tabs, httpx thin client)
- **Testing**: pytest (377 tests) + ruff

## Streamlit UI Tabs

1. **Ingest** — Upload documents, skeleton indexing with progress
2. **Search & Q&A** — Mode selector (vector/hybrid/agent), confidence bar, sources
3. **Graph Explorer** — Phrase + passage node visualization (Graphviz)
4. **Agent Trace** — Routing decision, self-correction steps, raw trace
5. **Benchmark** — Run 6 modes, PASS/FAIL table, comparison metrics
6. **Reasoning** — Mangle rule testing, query classification preview
7. **Settings** — Config display, cache stats, monitor analytics, clear DB

## References

- [KET-RAG: Cost-Efficient Graph RAG](https://arxiv.org/abs/2502.09304) (KDD 2025)
- [HippoRAG 2: Agentic Retrieval](https://arxiv.org/abs/2502.14802) (ICML 2025)
- [VectorCypher: Neo4j Graph Retrieval](https://neo4j.com/docs/)
- [Agentic RAG: Self-Correcting Retrieval](https://arxiv.org/abs/2401.15884)
- [Mangle: Datalog Dialect](https://github.com/google/mangle) (Google Research)

## License

MIT
