#!/usr/bin/env python3
"""Launch the Agentic Graph RAG API server."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pymangle"))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8507,
        reload=False,
        log_level="info",
    )
