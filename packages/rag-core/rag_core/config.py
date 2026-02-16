"""Agentic Graph RAG configuration via Pydantic Settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Neo4jSettings(BaseSettings):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "temporal_kb_2026"

    model_config = {"env_prefix": "NEO4J_"}


class OpenAISettings(BaseSettings):
    api_key: str = ""
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
