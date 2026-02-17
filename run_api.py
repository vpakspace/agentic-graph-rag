#!/usr/bin/env python3
"""Launch the Agentic Graph RAG API server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8507,
        reload=False,
        log_level="info",
    )
