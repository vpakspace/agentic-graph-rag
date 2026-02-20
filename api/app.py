"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI

from api.deps import set_service
from api.routes import router

if TYPE_CHECKING:
    from agentic_graph_rag.service import PipelineService


def create_app(service: PipelineService | None = None) -> FastAPI:
    """Create FastAPI app. If service is provided, use it (for testing).
    Otherwise, create from config at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            # Production: create service from config
            from neo4j import GraphDatabase
            from rag_core.config import get_settings, make_openai_client

            cfg = get_settings()
            driver = GraphDatabase.driver(
                cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
            )
            client = make_openai_client(cfg)

            reasoning = None
            try:
                from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine
                rules_dir = str(Path(__file__).resolve().parent.parent / "agentic_graph_rag" / "reasoning" / "rules")
                reasoning = ReasoningEngine(rules_dir)
            except Exception:
                pass

            from agentic_graph_rag.service import PipelineService
            svc = PipelineService(driver, client, reasoning)
            set_service(svc)
            try:
                from api.mcp_server import mount_mcp
                mount_mcp(app, svc)
            except Exception:
                pass
            yield
            driver.close()
        else:
            # Testing: use provided service
            set_service(service)
            try:
                from api.mcp_server import mount_mcp
                mount_mcp(app, service)
            except Exception:
                pass
            yield

    app = FastAPI(
        title="Agentic Graph RAG API",
        version="0.6.0",
        description="Typed API contract with full pipeline provenance",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
