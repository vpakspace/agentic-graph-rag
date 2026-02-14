"""rag-core — shared RAG library for Agentic Graph RAG."""

__version__ = "0.1.0"


# Lazy imports to avoid heavy deps at import time
def __getattr__(name: str):
    _module_map = {
        "Settings": "rag_core.config",
        "Chunk": "rag_core.models",
        "SearchResult": "rag_core.models",
        "QAResult": "rag_core.models",
        "Entity": "rag_core.models",
        "Relationship": "rag_core.models",
        "PhraseNode": "rag_core.models",
        "PassageNode": "rag_core.models",
        "load_file": "rag_core.loader",
        "chunk_text": "rag_core.chunker",
        "enrich_chunks": "rag_core.enricher",
        "embed_chunks": "rag_core.embedder",
        "VectorStore": "rag_core.vector_store",
        "KGClient": "rag_core.kg_client",
        "expand_query": "rag_core.query_expander",
        "rerank": "rag_core.reranker",
        "generate_answer": "rag_core.generator",
        "evaluate_relevance": "rag_core.reflector",
    }
    if name in _module_map:
        import importlib

        mod = importlib.import_module(_module_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'rag_core' has no attribute {name!r}")
