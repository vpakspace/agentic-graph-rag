"""Tests for rag_core.config."""



from rag_core.config import (
    AgentSettings,
    IndexingSettings,
    Neo4jSettings,
    OpenAISettings,
    RetrievalSettings,
    Settings,
    get_settings,
)


class TestNeo4jSettings:
    def test_defaults(self):
        s = Neo4jSettings()
        assert s.uri == "bolt://localhost:7687"
        assert s.user == "neo4j"
        assert s.password == "neo4j"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "bolt://custom:7688")
        s = Neo4jSettings()
        assert s.uri == "bolt://custom:7688"


class TestOpenAISettings:
    def test_defaults(self):
        s = OpenAISettings()
        assert s.embedding_model == "text-embedding-3-small"
        assert s.embedding_dimensions == 1536
        assert s.llm_model == "gpt-4o"
        assert s.llm_model_mini == "gpt-4o-mini"
        assert s.llm_temperature == 0.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        s = OpenAISettings()
        assert s.api_key == "sk-test-123"


class TestIndexingSettings:
    def test_defaults(self):
        s = IndexingSettings()
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.skeleton_beta == 0.25
        assert s.knn_k == 10
        assert s.pagerank_damping == 0.85

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("INDEXING_SKELETON_BETA", "0.3")
        s = IndexingSettings()
        assert s.skeleton_beta == 0.3


class TestRetrievalSettings:
    def test_defaults(self):
        s = RetrievalSettings()
        assert s.top_k_vector == 10
        assert s.top_k_final == 10
        assert s.vector_threshold == 0.5
        assert s.max_hops == 3
        assert s.ppr_alpha == 0.15


class TestAgentSettings:
    def test_defaults(self):
        s = AgentSettings()
        assert s.max_retries == 2
        assert s.relevance_threshold == 2.0


class TestSettings:
    def test_nested_settings(self):
        s = Settings()
        assert isinstance(s.neo4j, Neo4jSettings)
        assert isinstance(s.openai, OpenAISettings)
        assert isinstance(s.indexing, IndexingSettings)
        assert isinstance(s.retrieval, RetrievalSettings)
        assert isinstance(s.agent, AgentSettings)
        assert s.log_level == "INFO"

    def test_get_settings_returns_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)
