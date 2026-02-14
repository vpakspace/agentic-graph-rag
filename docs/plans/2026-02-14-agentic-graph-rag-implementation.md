# Agentic Graph RAG — Implementation Plan

**Date**: 2026-02-14
**Design**: `2026-02-14-agentic-graph-rag-design.md`
**Estimated phases**: 10

---

## Phase 1: Project Scaffolding + rag-core Package (~30 min)

**Goal**: Monorepo structure, pyproject.toml, rag-core package with __init__.py

### Tasks:
1. Create full directory tree (agentic-graph-rag/)
2. Create `packages/rag-core/pyproject.toml` (name=rag-core, version=0.1.0, deps)
3. Create `packages/rag-core/rag_core/__init__.py` (exports)
4. Create root `pyproject.toml` (name=agentic-graph-rag, deps include rag-core)
5. Create `.env.example` (OPENAI_API_KEY, NEO4J_URI/USER/PASSWORD)
6. Create `requirements.txt` (pinned versions)
7. `pip install -e packages/rag-core` — verify import works
8. Git init + initial commit

**Verification**: `python -c "from rag_core import config; print('ok')"`

---

## Phase 2: rag-core — Config + Models (~20 min)

**Goal**: Pydantic Settings config + unified data models

### Tasks:
1. `rag_core/config.py` — copy from TKB, adapt: OpenAI, Neo4j, Indexing, Retrieval, Agent settings
2. `rag_core/models.py` — merge:
   - From RAG 2.0: Chunk, SearchResult, QAResult
   - From TKB: Entity, Relationship, TemporalEvent
   - New: PhraseNode, PassageNode, GraphContext, RouterDecision
3. Unit tests: `tests/test_config.py`, `tests/test_models.py`

**Verification**: Tests pass, models serialize/deserialize correctly

---

## Phase 3: rag-core — Ingestion Pipeline (~30 min)

**Goal**: Loader + Chunker + Enricher + Embedder from existing projects

### Tasks:
1. `rag_core/loader.py` — copy from TKB (Docling, GPU support, table metadata)
2. `rag_core/chunker.py` — copy from TKB (table-aware, sanitize_for_graphiti, split_large_content)
3. `rag_core/enricher.py` — copy from RAG 2.0 (contextual enrichment via OpenAI)
4. `rag_core/embedder.py` — copy from RAG 2.0 (text-embedding-3-small, batch)
5. Adapt all imports to use `rag_core.config` and `rag_core.models`
6. Unit tests: `tests/test_loader.py`, `tests/test_chunker.py`, `tests/test_enricher.py`, `tests/test_embedder.py`

**Verification**: Load a test PDF → chunk → enrich → embed → verify embeddings shape

---

## Phase 4: rag-core — Storage Layer (~30 min)

**Goal**: Vector store + KG client from existing projects

### Tasks:
1. `rag_core/vector_store.py` — copy from RAG 2.0 (Neo4j Vector Index, CRUD, search)
2. `rag_core/kg_client.py` — copy from rag-temporal (Graphiti wrapper, sanitize, episode splitting, SEARCH_RECIPES)
3. Adapt to `rag_core.config` and `rag_core.models`
4. Unit tests: `tests/test_vector_store.py`, `tests/test_kg_client.py` (mock Neo4j/Graphiti)

**Verification**: Tests pass with mocked Neo4j driver

---

## Phase 5: rag-core — Retrieval + Generation (~30 min)

**Goal**: Query expander, reranker, generator, reflector from existing projects

### Tasks:
1. `rag_core/query_expander.py` — copy from RAG 2.0
2. `rag_core/reranker.py` — copy from RAG 2.0
3. `rag_core/generator.py` — copy from RAG 2.0
4. `rag_core/reflector.py` — copy from RAG 2.0 (relevance eval 1-5, retry if < 3.0)
5. `rag_core/i18n.py` — copy from TKB (~180 keys)
6. Adapt all imports
7. Unit tests for each module

**Verification**: `from rag_core import generator, reflector, query_expander` works

---

## Phase 6: Skeleton Indexer (KET-RAG) (~45 min)

**Goal**: NEW — PageRank-based selective extraction

### Tasks:
1. `agentic_graph_rag/indexing/skeleton.py`:
   - `build_knn_graph(chunks, embeddings, k=10)` — cosine similarity KNN
   - `compute_pagerank(knn_graph, damping=0.85)` — networkx PageRank
   - `select_skeletal_chunks(chunks, pagerank_scores, beta=0.25)` — top-β selection
   - `extract_entities_full(skeletal_chunks)` — LLM extraction for top chunks
   - `link_peripheral_keywords(peripheral_chunks)` — keyword-based links (no LLM)
   - `build_skeleton_index(chunks, embeddings)` — orchestrator
2. `agentic_graph_rag/indexing/dual_node.py`:
   - `create_phrase_nodes(entities)` — entity nodes in Neo4j
   - `create_passage_nodes(chunks)` — full-text nodes in Neo4j
   - `link_phrase_to_passage(phrase_id, passage_id)` — MENTIONED_IN relationship
   - `compute_ppr(graph, query_nodes, alpha=0.15)` — Personalized PageRank
3. Unit tests: `tests/test_skeleton.py`, `tests/test_dual_node.py`
4. Dependencies: `networkx` (for PageRank)

**Verification**: Given 100 chunks → skeleton selects 25 → creates phrase+passage nodes in Neo4j

---

## Phase 7: VectorCypher Retrieval (~30 min)

**Goal**: NEW — Hybrid vector entry + Cypher traversal

### Tasks:
1. `agentic_graph_rag/retrieval/vector_cypher.py`:
   - `find_entry_points(query_embedding, top_k=5, threshold=0.7)` — Neo4j vector search
   - `traverse_graph(entry_ids, max_hops=2)` — Cypher apoc.path.subgraphAll
   - `collect_context(traversal_result)` — triplets + passage nodes text
   - `search(query, top_k=5, max_hops=2)` — full VectorCypher pipeline
2. Cypher queries:
   - Vector entry: `CALL db.index.vector.queryNodes(...) YIELD node, score WHERE score > $threshold`
   - Traversal: `CALL apoc.path.subgraphAll(start, {maxLevel: $max_hops})`
   - Context: `MATCH (p:PhraseNode)-[:MENTIONED_IN]->(pass:PassageNode) RETURN pass.text`
3. Unit tests: `tests/test_vector_cypher.py` (mock Neo4j)

**Verification**: Query → entry points → traversal → assembled context with triplets

---

## Phase 8: Agentic Router + Self-Correction (~45 min)

**Goal**: NEW — Query classifier, tool selection, self-correction loop

### Tasks:
1. `agentic_graph_rag/agent/router.py`:
   - `classify_query(query) -> RouterDecision` — LLM-based classification
   - Categories: simple, relation, multi_hop, global, temporal
   - Pattern matching: "связь", "сравнить", "цепочка" → relation; "покажи все" → global
2. `agentic_graph_rag/agent/tools.py`:
   - `vector_search(query, top_k)` — simple vector
   - `cypher_traverse(query, top_k, max_hops)` — VectorCypher
   - `community_search(query)` — Graphiti community search
   - `hybrid_search(query, top_k)` — vector + KG merge (RRF)
   - `temporal_query(query, valid_at)` — Graphiti temporal
   - `full_document_read(top_k)` — all chunks
3. `agentic_graph_rag/agent/retrieval_agent.py`:
   - `run(query) -> QAResult` — main entry point
   - `select_tool(decision) -> callable` — tool mapping
   - `self_correction_loop(query, results, max_retries=2)`:
     - Evaluate relevance (1-5)
     - If < 3.0 → expand depth, retry with different tool
     - Check contradictions across results
   - `generate_answer(query, context) -> QAResult`
4. `agentic_graph_rag/generation/graph_verifier.py`:
   - `check_contradictions(facts, graph_context)` — detect conflicting info
   - `verify_via_traversal(claim, start_entity)` — path-based verification
5. Unit tests: `tests/test_router.py`, `tests/test_tools.py`, `tests/test_retrieval_agent.py`, `tests/test_graph_verifier.py`

**Verification**: Different query types route to correct tools, self-correction triggers on low relevance

---

## Phase 9: Optimization (Cache + Monitor) (~20 min)

**Goal**: NEW — Performance optimization layer

### Tasks:
1. `agentic_graph_rag/optimization/cache.py`:
   - `SubgraphCache` — LRU cache for frequently queried subgraphs
   - `CommunityCache` — cache for community summaries
   - `cache_key(query_embedding)` — deterministic hash
2. `agentic_graph_rag/optimization/monitor.py`:
   - `QueryMonitor` — track query types, latency, tool usage
   - `suggest_pagerank_weights(stats)` — recommend indexing priorities
   - `get_stats() -> dict` — dashboard data
3. Unit tests: `tests/test_cache.py`, `tests/test_monitor.py`

**Verification**: Cache hit/miss ratio, monitor reports query distribution

---

## Phase 10: Streamlit UI + Benchmark + CI (~45 min)

**Goal**: 6-tab UI, benchmark runner, CI/CD

### Tasks:
1. `ui/streamlit_app.py` — 6 tabs:
   - Ingest: file upload, skeleton indexing progress, entity/node counts
   - Search & Q&A: mode selector, query input, confidence bar, sources
   - Graph Explorer: phrase+passage node visualization (st.graphviz or pyvis)
   - Agent Trace: routing decision display, self-correction steps, tool call log
   - Benchmark: run all modes, PASS/FAIL table, delta metrics
   - Settings: config display, cache stats, clear DB
2. `benchmark/questions.json` — 15+ questions (simple, relation, multi-hop, global, temporal)
3. `benchmark/runner.py` — run 5 modes, compute accuracy/F1/latency/cost
4. `benchmark/compare.py` — generate comparison table
5. `.github/workflows/ci.yml` — pytest + ruff, Python 3.12
6. `README.md` — installation, usage, architecture, benchmarks
7. Copy test PDFs from `~/pageindex/data/` to `data/`
8. Git commit + push to GitHub

**Verification**: Streamlit runs on port 8506, benchmark produces comparison table, CI green

---

## Summary

| Phase | Component | Time | New/Reuse |
|-------|-----------|------|-----------|
| 1 | Scaffolding + rag-core package | 30 min | Setup |
| 2 | Config + Models | 20 min | Reuse (TKB + RAG 2.0) |
| 3 | Ingestion Pipeline | 30 min | Reuse (TKB + RAG 2.0) |
| 4 | Storage Layer | 30 min | Reuse (RAG 2.0 + rag-temporal) |
| 5 | Retrieval + Generation | 30 min | Reuse (RAG 2.0 + TKB) |
| 6 | **Skeleton Indexer** | 45 min | **NEW** (KET-RAG) |
| 7 | **VectorCypher** | 30 min | **NEW** |
| 8 | **Agentic Router + Self-Correction** | 45 min | **NEW** |
| 9 | **Optimization** | 20 min | **NEW** |
| 10 | UI + Benchmark + CI | 45 min | Mixed |
| **Total** | | **~5.5 hours** | |

Phases 1-5: Reuse existing code (~2.5h)
Phases 6-9: New Graph RAG components (~2.5h)
Phase 10: UI + Polish (~45min)
