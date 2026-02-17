"""Dependency injection for FastAPI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_graph_rag.service import PipelineService

_service: PipelineService | None = None


def get_service() -> PipelineService:
    """Get the PipelineService singleton."""
    if _service is None:
        raise RuntimeError("PipelineService not initialized")
    return _service


def set_service(service: PipelineService) -> None:
    """Set the PipelineService singleton (called at startup)."""
    global _service  # noqa: PLW0603
    _service = service
