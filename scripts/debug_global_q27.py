#!/usr/bin/env python3
"""Debug: what does comprehensive_search return for Q27?"""
import os, sys, logging
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pymangle"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from neo4j import GraphDatabase
from openai import OpenAI
from rag_core.config import get_settings
from agentic_graph_rag.agent.tools import comprehensive_search, full_document_read, vector_search

cfg = get_settings()
driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
client = OpenAI(api_key=cfg.openai.api_key)

queries = {
    "Q27": "Перечисли все семь архитектурных решений, описанных в дизайне Semantic Companion Layer",
    "Q8": "Перечисли все компоненты архитектуры графа знаний",
    "Q19": "Опиши все компоненты и слои архитектуры MeaningHub",
}

for qid, query in queries.items():
    print(f"\n{'='*60}")
    print(f"{qid}: {query}")
    print(f"{'='*60}")

    # What does full_document_read return with large top_k?
    results = full_document_read(query, driver, client, top_k=30)
    print(f"\nfull_document_read (top_k=30): {len(results)} results")
    for i, r in enumerate(results[:5]):
        text = r.chunk.content[:150].replace('\n', ' ')
        print(f"  [{i+1}] score={r.score:.3f} | {text}...")

    # Check keywords in retrieved text
    all_text = " ".join(r.chunk.content for r in results)
    keywords_q27 = ["GraphQL", "multi-backend", "ConstraintSet", "MCP", "packs", "Strawberry", "SDL"]
    keywords_q8 = ["ontology", "OWL", "extraction", "Neo4j", "embedding", "skeleton"]
    keywords_q19 = ["northbound", "southbound", "semantic core", "packs", "GraphQL"]

    kw = keywords_q27 if "семь" in query else (keywords_q8 if "компоненты архитектуры графа" in query else keywords_q19)
    found = [k for k in kw if k.lower() in all_text.lower()]
    missing = [k for k in kw if k.lower() not in all_text.lower()]
    print(f"\n  Keywords FOUND in top-30: {found}")
    print(f"  Keywords MISSING: {missing}")

driver.close()
