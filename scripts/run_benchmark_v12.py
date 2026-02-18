#!/usr/bin/env python3
"""Run benchmark v12 — 30 questions x 6 modes = 180 evaluations.

v12 changes vs v11:
- Hybrid judge: keyword overlap ≥40% → auto-PASS + cross-language concept matching
- Smart mention routing: _MENTION_RE + _needs_comprehensive() for non-agent modes
- v13 engine improvements: make_openai_client, access.mg/graph.mg integration
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

questions = load_questions()
doc1 = sum(1 for q in questions if q['id'] <= 15)
doc2 = sum(1 for q in questions if q['id'] > 15)
print(f"Loaded {len(questions)} questions (Doc1: {doc1}, Doc2: {doc2})")

modes = ["vector", "cypher", "hybrid", "agent_pattern", "agent_llm", "agent_mangle"]
results = run_benchmark(driver, openai_client, modes=modes, questions=questions, lang="ru")

# Load v11 for comparison
v11_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "results_v11_full.json")
v11 = {}
if os.path.exists(v11_path):
    with open(v11_path) as f:
        v11 = json.load(f)

print("\n" + "=" * 70)
print("BENCHMARK v12 — 30 questions x 6 modes = 180 evaluations")
print("=" * 70)

for mode, mode_results in results.items():
    passed = sum(1 for r in mode_results if r["passed"])
    total = len(mode_results)
    doc1_results = [r for r in mode_results if r["id"] <= 15]
    doc2_results = [r for r in mode_results if r["id"] > 15]
    doc1_pass = sum(1 for r in doc1_results if r["passed"])
    doc2_total = len(doc2_results)
    doc2_pass = sum(1 for r in doc2_results if r["passed"])

    # v11 comparison
    v11_mode = v11.get(mode, [])
    v11_pass = sum(1 for r in v11_mode if r["passed"]) if v11_mode else "?"

    # Per-question details
    details = " ".join(
        f"Q{r['id']}:{'P' if r['passed'] else 'F'}"
        for r in sorted(mode_results, key=lambda x: x["id"])
    )

    delta = f" (Δ{passed - v11_pass:+d})" if isinstance(v11_pass, int) else ""
    print(f"\n  {mode:20s}: {passed}/{total} ({100*passed//total}%){delta}")
    print(f"    Doc1 (RU): {doc1_pass}/15  Doc2 (EN): {doc2_pass}/{doc2_total}")
    if isinstance(v11_pass, int):
        print(f"    vs v11: {v11_pass}/30")
    print(f"    {details}")

total_passed = sum(sum(1 for r in mr if r["passed"]) for mr in results.values())
total_all = sum(len(mr) for mr in results.values())
doc1_total_pass = sum(sum(1 for r in mr if r["id"] <= 15 and r["passed"]) for mr in results.values())
doc2_total_pass = sum(sum(1 for r in mr if r["id"] > 15 and r["passed"]) for mr in results.values())
doc2_total_all = sum(sum(1 for r in mr if r["id"] > 15) for mr in results.values())

v11_total = sum(sum(1 for r in mr if r["passed"]) for mr in v11.values()) if v11 else "?"
v11_delta = f" (Δ{total_passed - v11_total:+d})" if isinstance(v11_total, int) else ""

print(f"\n{'=' * 70}")
print(f"  {'OVERALL':20s}: {total_passed}/{total_all} ({100*total_passed//total_all}%){v11_delta}")
print(f"    Doc1 (RU): {doc1_total_pass}/90  Doc2 (EN): {doc2_total_pass}/{doc2_total_all}")
if isinstance(v11_total, int):
    print(f"    vs v11: {v11_total}/180")

# Question-type breakdown
type_stats = {}
for mode, mode_results in results.items():
    for r in mode_results:
        qtype = r.get("type", "unknown")
        if qtype not in type_stats:
            type_stats[qtype] = {"passed": 0, "total": 0}
        type_stats[qtype]["total"] += 1
        if r["passed"]:
            type_stats[qtype]["passed"] += 1

print("\n  By question type:")
for qtype, stats in sorted(type_stats.items()):
    print(f"    {qtype:12s}: {stats['passed']}/{stats['total']} ({100*stats['passed']//stats['total']}%)")

# Persistent failures analysis
print("\n  Persistent failures (0/6):")
for q in questions:
    q_results = []
    for mode, mode_results in results.items():
        for r in mode_results:
            if r["id"] == q["id"]:
                q_results.append(r)
    passes = sum(1 for r in q_results if r["passed"])
    if passes == 0:
        print(f"    Q{q['id']} ({q['type']}): {q.get('question_ru', q['question'])[:60]}...")

# Save results
out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "results_v12_run.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")

driver.close()
