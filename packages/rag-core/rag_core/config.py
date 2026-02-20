"""Agentic Graph RAG configuration via Pydantic Settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Neo4jSettings(BaseSettings):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"

    model_config = {"env_prefix": "NEO4J_"}


class OpenAISettings(BaseSettings):
    api_key: str = ""
    base_url: str = ""  # LiteLLM proxy: e.g. "http://localhost:4000/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o"
    llm_model_mini: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    model_config = {"env_prefix": "OPENAI_"}


class IndexingSettings(BaseSettings):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    skeleton_beta: float = 0.25
    knn_k: int = 10
    pagerank_damping: float = 0.85

    model_config = {"env_prefix": "INDEXING_"}


class RetrievalSettings(BaseSettings):
    top_k_vector: int = 10
    top_k_final: int = 10
    vector_threshold: float = 0.5
    max_hops: int = 3
    ppr_alpha: float = 0.15

    model_config = {"env_prefix": "RETRIEVAL_"}


class AgentSettings(BaseSettings):
    max_retries: int = 2
    relevance_threshold: float = 2.0

    model_config = {"env_prefix": "AGENT_"}


class Settings(BaseSettings):
    neo4j: Neo4jSettings = Neo4jSettings()
    openai: OpenAISettings = OpenAISettings()
    indexing: IndexingSettings = IndexingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    agent: AgentSettings = AgentSettings()

    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> Settings:
    """Create settings instance loading from environment."""
    return Settings()


def make_openai_client(settings: Settings | None = None):
    """Create OpenAI client with optional LiteLLM proxy support.

    If OPENAI_BASE_URL is set, uses it as base_url (e.g. LiteLLM proxy).
    If api_key is empty and base_url is set, uses "none" as placeholder.
    """
    from openai import OpenAI

    cfg = settings or get_settings()
    kwargs: dict[str, str] = {}
    if cfg.openai.api_key:
        kwargs["api_key"] = cfg.openai.api_key
    elif cfg.openai.base_url:
        kwargs["api_key"] = "none"  # LiteLLM proxy doesn't need real key
    if cfg.openai.base_url:
        kwargs["base_url"] = cfg.openai.base_url
    return OpenAI(**kwargs)
