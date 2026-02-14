# Agentic Graph RAG

**Skeleton Indexing + VectorCypher + Agentic Router with Self-Correction**

A production-ready Graph RAG system that combines four cutting-edge techniques from recent research papers into a unified retrieval pipeline.

## Key Techniques

| Technique | Source | What It Does |
|-----------|--------|-------------|
| **Skeleton Indexing** | KET-RAG (KDD 2025) | KNN graph → PageRank → selective entity extraction (10x cost savings) |
| **Dual Node Structure** | HippoRAG 2 (ICML 2025) | Phrase nodes + passage nodes + Personalized PageRank |
| **VectorCypher** | Neo4j / GraphRAG | Vector entry points → Cypher traversal → context assembly |
| **Agentic Router** | Custom | Pattern/LLM classification → tool selection → self-correction loop |

## Architecture

```
Ingestion:
  Document → Docling → Chunker → Enricher → Embedder
           → Skeleton Indexer (PageRank top-β)
           → Dual Node Builder (phrase + passage nodes)
           → Neo4j (Vector Index + Knowledge Graph)

Retrieval:
  Query → Router (simple/relation/multi_hop/global/temporal)
        → Tool Selection (vector/cypher/hybrid/full_read/temporal)
        → Self-Correction Loop (evaluate relevance → escalate if <3.0)
        → Graph Verifier (contradiction detection)
        → Generator (GPT-4o answer + citations)
```

## Project Structure

```
agentic-graph-rag/
├── packages/rag-core/        # Shared pip package (models, config, ingestion, retrieval)
│   └── rag_core/
│       ├── models.py          # Chunk, Entity, SearchResult, QAResult, QueryType, etc.
│       ├── config.py          # Pydantic Settings (nested: Neo4j, OpenAI, Indexing, Agent)
│       ├── loader.py          # Docling: PDF/DOCX/PPTX + GPU
│       ├── chunker.py         # Table-aware chunking
│       ├── enricher.py        # Contextual enrichment (OpenAI)
│       ├── embedder.py        # text-embedding-3-small batch processing
│       ├── vector_store.py    # Neo4j Vector Index CRUD
│       ├── kg_client.py       # Graphiti wrapper + Cypher
│       ├── generator.py       # LLM answer generation
│       ├── reflector.py       # Relevance evaluation (1-5 scale)
│       └── i18n.py            # RU/EN localization (~50 keys)
│
├── agentic_graph_rag/         # Graph RAG components (NEW)
│   ├── indexing/
│   │   ├── skeleton.py        # KET-RAG: KNN → PageRank → skeletal extraction
│   │   └── dual_node.py       # HippoRAG 2: phrase + passage nodes + PPR
│   ├── retrieval/
│   │   └── vector_cypher.py   # Vector entry → Cypher traversal → context
│   ├── agent/
│   │   ├── router.py          # Query classifier (pattern + LLM)
│   │   ├── retrieval_agent.py # Orchestrator + self-correction loop
│   │   └── tools.py           # 6 tools: vector, cypher, community, hybrid, temporal, full_read
│   ├── generation/
│   │   └── graph_verifier.py  # Contradiction detection + claim verification
│   └── optimization/
│       ├── cache.py           # LRU SubgraphCache + CommunityCache
│       └── monitor.py         # QueryMonitor + PageRank tuning suggestions
│
├── ui/
│   └── streamlit_app.py       # 6-tab Streamlit UI (port 8506)
│
├── benchmark/
│   ├── questions.json         # 15 test questions (5 types, EN/RU)
│   ├── runner.py              # 5-mode benchmark runner
│   └── compare.py             # Comparison table generator
│
└── tests/                     # 330+ unit tests
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
pytest -x -q  # 330+ tests, ~1 second
```

### Run Streamlit UI

```bash
streamlit run ui/streamlit_app.py --server.port 8506
```

### Run Benchmark

```python
from benchmark.runner import run_benchmark, load_questions
from benchmark.compare import compare_modes

# Requires Neo4j running + documents ingested
results = run_benchmark(driver, openai_client, modes=["vector", "hybrid", "agent_pattern"])
print(compare_modes(results))
```

## Retrieval Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Vector** | Embedding similarity search | Simple factual queries |
| **Cypher** | Graph traversal via VectorCypher | Relationship queries |
| **Hybrid** | Vector + Graph with RRF fusion | Multi-hop queries |
| **Agent (pattern)** | Auto-routing via regex patterns | General use (fast) |
| **Agent (LLM)** | Auto-routing via GPT-4o-mini | General use (accurate) |

## Self-Correction Loop

The agent evaluates retrieval quality (1-5 scale) and escalates through the tool chain when results are insufficient:

```
vector_search → cypher_traverse → hybrid_search → full_document_read
```

Each retry uses a different tool with expanded search scope. Max 2 retries by default.

## Configuration

All settings via `.env` or environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `INDEXING_SKELETON_BETA` | `0.25` | Fraction of chunks for full extraction |
| `INDEXING_KNN_K` | `10` | KNN graph neighbors |
| `INDEXING_PAGERANK_DAMPING` | `0.85` | PageRank damping factor |
| `RETRIEVAL_MAX_HOPS` | `2` | Max graph traversal depth |
| `AGENT_MAX_RETRIES` | `2` | Self-correction retries |
| `AGENT_RELEVANCE_THRESHOLD` | `3.0` | Minimum relevance score (1-5) |

## Tech Stack

- **LLM**: OpenAI GPT-4o / GPT-4o-mini
- **Embeddings**: text-embedding-3-small (1536 dim)
- **Graph DB**: Neo4j 5.x (Vector Index + Cypher)
- **KG Framework**: Graphiti
- **Doc Parsing**: Docling (PDF/DOCX/PPTX + GPU)
- **Graph Algorithms**: NetworkX (PageRank, KNN, PPR)
- **UI**: Streamlit
- **Testing**: pytest + ruff

## Streamlit UI Tabs

1. **Ingest** — Upload documents, skeleton indexing with progress
2. **Search & Q&A** — Mode selector (vector/hybrid/agent), confidence bar, sources
3. **Graph Explorer** — Phrase + passage node visualization (Graphviz)
4. **Agent Trace** — Routing decision, self-correction steps, raw trace
5. **Benchmark** — Run 5 modes, PASS/FAIL table, comparison metrics
6. **Settings** — Config display, cache stats, monitor analytics, clear DB

## References

- [KET-RAG: Cost-Efficient Graph RAG](https://arxiv.org/abs/2502.09304) (KDD 2025)
- [HippoRAG 2: Agentic Retrieval](https://arxiv.org/abs/2502.14802) (ICML 2025)
- [VectorCypher: Neo4j Graph Retrieval](https://neo4j.com/docs/)
- [Agentic RAG: Self-Correcting Retrieval](https://arxiv.org/abs/2401.15884)

## License

MIT
