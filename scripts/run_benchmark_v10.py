#!/usr/bin/env python3
"""Run benchmark v10 after co-occurrence expansion restore."""

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pymangle"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from neo4j import GraphDatabase
from openai import OpenAI

from rag_core.config import get_settings
from benchmark.runner import run_benchmark, load_questions

cfg = get_settings()
driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
openai_client = OpenAI(api_key=cfg.openai.api_key)

questions = load_questions()
print(f"Loaded {len(questions)} questions")

modes = ["vector", "cypher", "hybrid", "agent_pattern", "agent_llm", "agent_mangle"]
results = run_benchmark(driver, openai_client, modes=modes, questions=questions, lang="ru")

# Print results
print("\n" + "=" * 60)
print("BENCHMARK v10 RESULTS (co-occurrence expansion restored)")
print("=" * 60)

# Load v7 for comparison
v7_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "results_v7_dual.json")
v7 = {}
if os.path.exists(v7_path):
    with open(v7_path) as f:
        v7 = json.load(f)

for mode, mode_results in results.items():
    passed = sum(1 for r in mode_results if r["passed"])
    total = len(mode_results)
    doc1_results = [r for r in mode_results if r["id"] <= 15]
    doc2_results = [r for r in mode_results if r["id"] > 15]
    doc1_pass = sum(1 for r in doc1_results if r["passed"])
    doc2_pass = sum(1 for r in doc2_results if r["passed"])

    # v7 comparison
    v7_pass = sum(1 for r in v7.get(mode, []) if r["passed"]) if v7 else "?"
    delta = f"({'+' if passed >= v7_pass else ''}{passed - v7_pass} vs v7)" if isinstance(v7_pass, int) else ""

    print(f"  {mode:20s}: {passed}/{total} ({100*passed//total}%) | Doc1: {doc1_pass}/15 | Doc2: {doc2_pass}/5 {delta}")

total_passed = sum(sum(1 for r in mr if r["passed"]) for mr in results.values())
total_all = sum(len(mr) for mr in results.values())
v7_total = sum(sum(1 for r in mr if r["passed"]) for mr in v7.values()) if v7 else "?"
delta_total = f"({'+' if total_passed >= v7_total else ''}{total_passed - v7_total} vs v7)" if isinstance(v7_total, int) else ""
print(f"\n  {'OVERALL':20s}: {total_passed}/{total_all} ({100*total_passed//total_all}%) {delta_total}")

# Save results
out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "results_v10_cooccurrence.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")

driver.close()
