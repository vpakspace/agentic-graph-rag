#!/usr/bin/env python3
"""Run benchmark v11 — global questions improvement test.

Tests only global questions (Q8, Q9, Q10, Q19, Q27, Q28) across all 6 modes
to measure the impact of comprehensive_search improvements.
"""

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
from rag_core.config import get_settings, make_openai_client

from benchmark.runner import load_questions, run_benchmark

cfg = get_settings()
driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
openai_client = make_openai_client(cfg)

# Only global questions
all_questions = load_questions()
global_questions = [q for q in all_questions if q["type"] == "global"]
print(f"Running global questions only: {len(global_questions)} questions")
for q in global_questions:
    print(f"  Q{q['id']}: {q.get('question_ru', q['question'])[:60]}...")

modes = ["vector", "cypher", "hybrid", "agent_pattern", "agent_llm", "agent_mangle"]
results = run_benchmark(driver, openai_client, modes=modes, questions=global_questions, lang="ru")

# Load v10 for comparison
v10_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "results_v10_full.json")
v10 = {}
if os.path.exists(v10_path):
    with open(v10_path) as f:
        v10 = json.load(f)

print("\n" + "=" * 70)
print("BENCHMARK v11 — GLOBAL QUESTIONS ONLY")
print("=" * 70)

for mode, mode_results in results.items():
    passed = sum(1 for r in mode_results if r["passed"])
    total = len(mode_results)

    # v10 global comparison
    v10_global = [r for r in v10.get(mode, []) if r.get("type") == "global"]
    v10_pass = sum(1 for r in v10_global if r["passed"])
    delta = f"(v10: {v10_pass}/6)"

    details = " ".join(
        f"Q{r['id']}:{'PASS' if r['passed'] else 'FAIL'}"
        for r in sorted(mode_results, key=lambda x: x["id"])
    )
    print(f"  {mode:20s}: {passed}/{total} {delta}  | {details}")

total_passed = sum(sum(1 for r in mr if r["passed"]) for mr in results.values())
total_all = sum(len(mr) for mr in results.values())
v10_global_total = sum(
    sum(1 for r in v10.get(m, []) if r.get("type") == "global" and r["passed"])
    for m in modes
)
print(f"\n  {'OVERALL':20s}: {total_passed}/{total_all} (v10: {v10_global_total}/36)")
print(f"  Improvement: {total_passed - v10_global_total:+d} ({(total_passed - v10_global_total)*100//36:+d}pp)")

driver.close()
