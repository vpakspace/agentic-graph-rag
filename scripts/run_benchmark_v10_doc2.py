#!/usr/bin/env python3
"""Run benchmark v10 on Doc2 (English SCL) only — questions 16-20."""

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

# Only Doc2 questions (id 16-20)
all_questions = load_questions()
doc2_questions = [q for q in all_questions if q["id"] > 15]
print(f"Running Doc2 only: {len(doc2_questions)} questions")

modes = ["vector", "cypher", "hybrid", "agent_pattern", "agent_llm", "agent_mangle"]
results = run_benchmark(driver, openai_client, modes=modes, questions=doc2_questions, lang="ru")

# Load v7 for comparison
v7_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "results_v7_dual.json")
v7 = {}
if os.path.exists(v7_path):
    with open(v7_path) as f:
        v7 = json.load(f)

print("\n" + "=" * 60)
print("BENCHMARK v10 — Doc2 (English SCL) ONLY")
print("=" * 60)

for mode, mode_results in results.items():
    passed = sum(1 for r in mode_results if r["passed"])
    total = len(mode_results)

    v7_doc2 = [r for r in v7.get(mode, []) if r["id"] > 15]
    v7_pass = sum(1 for r in v7_doc2 if r["passed"])
    delta = f"(v7: {v7_pass}/5)"

    # Show per-question
    details = " ".join(
        f"Q{r['id']}:{'PASS' if r['passed'] else 'FAIL'}"
        for r in sorted(mode_results, key=lambda x: x["id"])
    )
    print(f"  {mode:20s}: {passed}/{total} {delta}  | {details}")

total_passed = sum(sum(1 for r in mr if r["passed"]) for mr in results.values())
total_all = sum(len(mr) for mr in results.values())
v7_doc2_total = sum(sum(1 for r in v7.get(m, []) if r["id"] > 15 and r["passed"]) for m in modes)
print(f"\n  {'OVERALL':20s}: {total_passed}/{total_all} (v7: {v7_doc2_total}/30)")

out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "results_v10_doc2.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")

driver.close()
