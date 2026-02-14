"""Agentic Graph RAG — Streamlit UI.

6 tabs: Ingest, Search & Q&A, Graph Explorer, Agent Trace, Benchmark, Settings.
Port: 8506
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from rag_core.config import get_settings
from rag_core.i18n import get_translator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic Graph RAG",
    page_icon="🔗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

lang = st.sidebar.radio("Language / Язык", ["en", "ru"], index=0)
t = get_translator(lang)

st.sidebar.title(t("app_title"))
st.sidebar.caption(t("app_subtitle"))

use_gpu = st.sidebar.checkbox(t("ingest_gpu"), value=False)
use_llm_router = st.sidebar.checkbox(
    "LLM Router" if lang == "en" else "LLM Роутер",
    value=False,
)


# ---------------------------------------------------------------------------
# Lazy-loaded resources
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_neo4j_driver():
    from neo4j import GraphDatabase
    cfg = get_settings()
    return GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))


@st.cache_resource
def _get_openai_client():
    from openai import OpenAI
    cfg = get_settings()
    return OpenAI(api_key=cfg.openai.api_key)


@st.cache_resource
def _get_vector_store():
    from rag_core.vector_store import VectorStore
    cfg = get_settings()
    return VectorStore(
        uri=cfg.neo4j.uri,
        user=cfg.neo4j.user,
        password=cfg.neo4j.password,
    )


@st.cache_resource
def _get_cache():
    from agentic_graph_rag.optimization.cache import SubgraphCache
    return SubgraphCache()


@st.cache_resource
def _get_monitor():
    from agentic_graph_rag.optimization.monitor import QueryMonitor
    return QueryMonitor()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "last_qa" not in st.session_state:
    st.session_state.last_qa = None
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ingest, tab_search, tab_graph, tab_trace, tab_bench, tab_settings = st.tabs([
    t("tab_ingest"),
    t("tab_search"),
    t("tab_graph_explorer"),
    t("tab_agent_trace"),
    t("tab_benchmark"),
    t("tab_settings"),
])


# ===================== TAB 1: INGEST ======================================

with tab_ingest:
    st.header(t("ingest_header"))
    st.caption(t("ingest_supported"))

    source = st.radio(
        t("ingest_upload"),
        [t("ingest_source_upload"), t("ingest_source_path")],
        horizontal=True,
    )

    file_path: str | None = None
    if source == t("ingest_source_upload"):
        uploaded = st.file_uploader(
            t("ingest_upload"),
            type=["txt", "pdf", "docx", "pptx", "xlsx", "html"],
            label_visibility="collapsed",
        )
        if uploaded:
            suffix = Path(uploaded.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.flush()
            file_path = tmp.name
    else:
        file_path = st.text_input(
            t("ingest_path_input"),
            placeholder=t("ingest_path_placeholder"),
        )
        if file_path and not Path(file_path).exists():
            st.warning(t("ingest_path_not_found", path=file_path))
            file_path = None

    skip_enrichment = st.checkbox(t("ingest_skip_enrichment"), value=False)

    if st.button(t("ingest_button"), disabled=not file_path):
        try:
            driver = _get_neo4j_driver()
            client = _get_openai_client()
            store = _get_vector_store()

            from rag_core.chunker import chunk_text
            from rag_core.embedder import embed_chunks
            from rag_core.enricher import enrich_chunks
            from rag_core.loader import load_file

            progress = st.progress(0, text=t("ingest_loading"))
            text = load_file(file_path, use_gpu=use_gpu)
            st.info(t("ingest_chars_loaded", chars=len(text)))
            progress.progress(20, text=t("ingest_chunking"))

            cfg = get_settings()
            chunks = chunk_text(text, cfg.indexing.chunk_size, cfg.indexing.chunk_overlap)
            st.info(t("ingest_chunks_created", count=len(chunks)))
            progress.progress(40, text=t("ingest_enriching"))

            if not skip_enrichment:
                chunks = enrich_chunks(chunks, text)
            progress.progress(60, text=t("ingest_embedding"))

            chunks = embed_chunks(chunks)
            progress.progress(80, text=t("ingest_storing"))

            store.add_chunks(chunks)
            total = store.count()
            progress.progress(100, text="Done")

            st.success(t("ingest_success", chunks=len(chunks), total=total))
        except Exception as e:
            st.error(t("error", msg=str(e)))


# ===================== TAB 2: SEARCH & Q&A ================================

with tab_search:
    st.header(t("search_header"))

    mode = st.radio(
        t("search_mode"),
        [
            t("search_mode_vector"),
            t("search_mode_hybrid"),
            t("search_mode_agent"),
        ],
        horizontal=True,
    )

    query = st.text_input(t("search_input"), placeholder=t("search_placeholder"))

    if st.button(t("search_button"), disabled=not query):
        try:
            driver = _get_neo4j_driver()
            client = _get_openai_client()
            monitor = _get_monitor()

            with st.spinner(t("search_thinking")):
                if mode == t("search_mode_agent"):
                    from agentic_graph_rag.agent.retrieval_agent import run as agent_run
                    with monitor.track("agent", "agent_router"):
                        qa = agent_run(query, driver, openai_client=client, use_llm_router=use_llm_router)
                elif mode == t("search_mode_hybrid"):
                    from rag_core.generator import generate_answer

                    from agentic_graph_rag.agent.tools import hybrid_search
                    with monitor.track("hybrid", "hybrid_search"):
                        results = hybrid_search(query, driver, client)
                        qa = generate_answer(query, results, client)
                else:
                    from rag_core.generator import generate_answer

                    from agentic_graph_rag.agent.tools import vector_search
                    with monitor.track("simple", "vector_search"):
                        results = vector_search(query, driver, client)
                        qa = generate_answer(query, results, client)

            st.session_state.last_qa = qa

            # Build trace info
            trace: dict[str, Any] = {"query": query, "mode": mode}
            if qa.router_decision:
                trace["router"] = {
                    "query_type": qa.router_decision.query_type.value,
                    "confidence": qa.router_decision.confidence,
                    "reasoning": qa.router_decision.reasoning,
                    "tool": qa.router_decision.suggested_tool,
                }
            trace["retries"] = qa.retries
            trace["confidence"] = qa.confidence
            trace["sources_count"] = len(qa.sources)
            st.session_state.last_trace = trace

            # Display answer
            st.subheader(t("search_answer"))
            st.write(qa.answer)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(t("search_confidence"), f"{qa.confidence:.0%}")
            with col2:
                st.metric(t("search_retries", count=qa.retries), qa.retries)

            if qa.router_decision:
                st.caption(t("search_query_type", qtype=qa.router_decision.query_type.value))
                st.caption(t("search_router_confidence", conf=qa.router_decision.confidence))

            if qa.sources:
                with st.expander(t("search_sources", count=len(qa.sources))):
                    for i, src in enumerate(qa.sources, 1):
                        st.markdown(f"**{i}.** {src.chunk.content[:200]}...")
                        st.caption(t("search_source_score", score=src.score))
        except Exception as e:
            st.error(t("error", msg=str(e)))


# ===================== TAB 3: GRAPH EXPLORER ==============================

with tab_graph:
    st.header(t("graph_header"))

    max_nodes = st.slider(t("graph_max_nodes"), 10, 200, 50)

    try:
        driver = _get_neo4j_driver()
        with driver.session() as session:
            # Count phrase nodes
            phrase_count = session.run(
                "MATCH (n:PhraseNode) RETURN count(n) AS cnt"
            ).single()["cnt"]

            # Count passage nodes
            passage_count = session.run(
                "MATCH (n:PassageNode) RETURN count(n) AS cnt"
            ).single()["cnt"]

            st.metric(t("graph_phrase_nodes"), phrase_count)
            st.metric(t("graph_passage_nodes"), passage_count)

            if phrase_count == 0 and passage_count == 0:
                st.info(t("graph_no_data"))
            else:
                # Build graphviz dot string
                result = session.run(
                    """MATCH (a:PhraseNode)-[r]->(b)
                    RETURN a.name AS src, type(r) AS rel, b.name AS tgt
                    LIMIT $limit""",
                    limit=max_nodes,
                )
                edges = [(rec["src"], rec["rel"], rec["tgt"]) for rec in result]

                if edges:
                    dot_lines = [
                        "digraph G {",
                        "  rankdir=LR;",
                        '  node [shape=box, style=filled, fillcolor="#E8F4FD"];',
                    ]
                    for src, rel, tgt in edges:
                        safe_src = (src or "?").replace('"', '\\"')
                        safe_tgt = (tgt or "?").replace('"', '\\"')
                        safe_rel = (rel or "").replace('"', '\\"')
                        dot_lines.append(f'  "{safe_src}" -> "{safe_tgt}" [label="{safe_rel}"];')
                    dot_lines.append("}")
                    st.graphviz_chart("\n".join(dot_lines))
                else:
                    st.info(t("graph_no_data"))

    except Exception as e:
        st.warning(t("error", msg=str(e)))


# ===================== TAB 4: AGENT TRACE =================================

with tab_trace:
    st.header(t("trace_header"))

    trace_data = st.session_state.get("last_trace")
    if trace_data is None:
        st.info(t("trace_no_data"))
    else:
        st.subheader(t("trace_routing"))

        if "router" in trace_data:
            r = trace_data["router"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t("trace_query_type"), r["query_type"])
            with col2:
                st.metric(t("trace_confidence"), f"{r['confidence']:.0%}")
            with col3:
                st.metric(t("trace_tool"), r["tool"])
            st.caption(f"{t('trace_reasoning')}: {r['reasoning']}")

        st.divider()

        st.subheader(t("trace_correction"))
        st.metric(t("trace_retries", count=trace_data.get("retries", 0)), trace_data.get("retries", 0))

        st.divider()

        st.subheader("Raw Trace")
        st.json(trace_data)


# ===================== TAB 5: BENCHMARK ===================================

with tab_bench:
    st.header(t("bench_header"))

    bench_modes = st.multiselect(
        t("bench_mode"),
        ["vector", "cypher", "hybrid", "agent_pattern", "agent_llm"],
        default=["vector", "hybrid", "agent_pattern"],
    )

    if st.button(t("bench_run"), disabled=not bench_modes):
        try:
            driver = _get_neo4j_driver()
            client = _get_openai_client()

            from benchmark.compare import compare_modes, compute_metrics
            from benchmark.runner import load_questions, run_benchmark

            questions = load_questions()

            with st.spinner(t("bench_running", current=0, total=len(questions))):
                all_results = run_benchmark(
                    driver, client, modes=bench_modes, questions=questions, lang=lang,
                )

            # Comparison table
            st.subheader(t("bench_results"))
            comparison = compare_modes(all_results)
            st.dataframe(comparison, use_container_width=True)

            # Per-mode details
            for mode_name, results in all_results.items():
                m = compute_metrics(results)
                with st.expander(f"{mode_name}: {m['correct']}/{m['total']} ({m['accuracy']:.0%})"):
                    rows = []
                    for r in results:
                        rows.append({
                            t("bench_col_q"): r["id"],
                            t("bench_col_status"): "PASS" if r["passed"] else "FAIL",
                            t("bench_col_confidence"): f"{r['confidence']:.2f}",
                            t("bench_col_retries"): r["retries"],
                            t("bench_col_question"): r["question"][:80],
                        })
                    st.dataframe(rows, use_container_width=True)

        except Exception as e:
            st.error(t("error", msg=str(e)))


# ===================== TAB 6: SETTINGS ====================================

with tab_settings:
    st.header(t("settings_header"))

    # Current config
    st.subheader(t("settings_current"))
    cfg = get_settings()
    config_dict = {
        "Neo4j URI": cfg.neo4j.uri,
        "Embedding Model": cfg.openai.embedding_model,
        "LLM Model": cfg.openai.llm_model,
        "Chunk Size": cfg.indexing.chunk_size,
        "Skeleton Beta": cfg.indexing.skeleton_beta,
        "KNN K": cfg.indexing.knn_k,
        "PageRank Damping": cfg.indexing.pagerank_damping,
        "Top K Vector": cfg.retrieval.top_k_vector,
        "Top K Final": cfg.retrieval.top_k_final,
        "Max Hops": cfg.retrieval.max_hops,
        "Max Retries": cfg.agent.max_retries,
        "Relevance Threshold": cfg.agent.relevance_threshold,
    }
    st.json(config_dict)

    # Vector store stats
    st.subheader(t("settings_store_stats"))
    try:
        store = _get_vector_store()
        st.write(t("settings_total_chunks", count=store.count()))
    except Exception as e:
        st.caption(t("error", msg=str(e)))

    # Cache stats
    st.subheader(t("settings_cache_header"))
    cache = _get_cache()
    cs = cache.stats()
    st.write(t("settings_cache_size", size=cs["size"], max_size=cs["max_size"]))
    st.write(t("settings_cache_hit_rate", rate=cs["hit_rate"]))

    # Monitor stats
    st.subheader(t("settings_monitor_header"))
    monitor = _get_monitor()
    ms = monitor.get_stats()
    st.write(t("settings_monitor_total", count=ms["total_queries"]))
    if ms["total_queries"] > 0:
        st.json(ms)
        suggestions = monitor.suggest_pagerank_weights()
        if suggestions.get("adjustments"):
            st.subheader(t("settings_suggestions"))
            st.json(suggestions)

    # Clear DB
    st.subheader(t("settings_clear_db"))
    confirm = st.text_input(t("settings_clear_confirm"), key="clear_confirm")
    if st.button(t("settings_clear_button"), disabled=confirm != "DELETE"):
        try:
            store = _get_vector_store()
            count = store.count()
            store.clear()
            st.success(t("settings_cleared", count=count))
        except Exception as e:
            st.error(t("error", msg=str(e)))
