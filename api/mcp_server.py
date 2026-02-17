"""MCP server tools for Agentic Graph RAG.

Provides 3 tools: resolve_intent, search_graph, explain_trace.
Tools are functions that can be registered with FastMCP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_graph_rag.service import PipelineService


def create_mcp_tools(service: PipelineService) -> dict:
    """Create MCP tool functions bound to a PipelineService instance.

    Returns a dict of {tool_name: callable} for registration with FastMCP
    or for direct testing.
    """

    def resolve_intent(query: str, mode: str = "agent_pattern") -> dict:
        """Resolve user query via Agentic Graph RAG pipeline."""
        qa = service.query(query, mode=mode)
        return qa.model_dump()

    def search_graph(query: str, tool: str = "vector_search") -> dict:
        """Search the knowledge graph without answer generation."""
        valid_tools = (
            "vector_search", "cypher_traverse", "hybrid_search",
            "comprehensive_search", "temporal_query", "full_document_read",
        )
        if tool not in valid_tools:
            return {"error": f"Unknown tool: {tool}"}

        qa = service.query(query, mode="agent_pattern")
        return {
            "results": [r.model_dump() for r in qa.sources],
            "trace": qa.trace.model_dump() if qa.trace else None,
        }

    def explain_trace(trace_id: str) -> dict:
        """Get provenance trace by ID."""
        trace = service.get_trace(trace_id)
        if trace is None:
            return {"error": f"Trace {trace_id} not found"}
        return trace.model_dump()

    return {
        "resolve_intent": resolve_intent,
        "search_graph": search_graph,
        "explain_trace": explain_trace,
    }


def mount_mcp(app, service: PipelineService):
    """Mount FastMCP server on a FastAPI/Starlette app.

    Uses SSE transport at /mcp/sse.
    """
    try:
        from fastmcp import FastMCP

        mcp = FastMCP("Agentic Graph RAG")
        tools = create_mcp_tools(service)

        @mcp.tool()
        def resolve_intent(query: str, mode: str = "agent_pattern") -> dict:
            """Resolve user query via Agentic Graph RAG pipeline.
            Returns answer with full provenance trace."""
            return tools["resolve_intent"](query, mode)

        @mcp.tool()
        def search_graph(query: str, tool: str = "vector_search") -> dict:
            """Search the knowledge graph. Tools: vector_search, cypher_traverse,
            hybrid_search, comprehensive_search, temporal_query, full_document_read."""
            return tools["search_graph"](query, tool)

        @mcp.tool()
        def explain_trace(trace_id: str) -> dict:
            """Get full provenance trace by trace ID."""
            return tools["explain_trace"](trace_id)

        mcp.mount(app, path="/mcp")

    except ImportError:
        import logging
        logging.getLogger(__name__).warning("fastmcp not installed — MCP server disabled")
