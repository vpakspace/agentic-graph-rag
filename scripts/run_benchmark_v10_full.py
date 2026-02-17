#!/usr/bin/env python3
"""Run benchmark v10 full — 30 questions (15 Doc1 RU + 15 Doc2 EN) x 6 modes = 180 evaluations."""

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

from benchmark.runner import load_questions, run_benchmark

cfg = get_settings()
driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
openai_client = OpenAI(api_key=cfg.openai.api_key)

questions = load_questions()
doc1 = sum(1 for q in questions if q['id'] <= 15)
doc2 = sum(1 for q in questions if q['id'] > 15)
print(f"Loaded {len(questions)} questions (Doc1: {doc1}, Doc2: {doc2})")

modes = ["vector", "cypher", "hybrid", "agent_pattern", "agent_llm", "agent_mangle"]
results = run_benchmark(driver, openai_client, modes=modes, questions=questions, lang="ru")

# Load v7 for comparison
v7_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "results_v7_dual.json")
v7 = {}
if os.path.exists(v7_path):
    with open(v7_path) as f:
        v7 = json.load(f)

print("\n" + "=" * 70)
print("BENCHMARK v10 FULL — 30 questions x 6 modes = 180 evaluations")
print("=" * 70)

for mode, mode_results in results.items():
    passed = sum(1 for r in mode_results if r["passed"])
    total = len(mode_results)
    doc1_results = [r for r in mode_results if r["id"] <= 15]
    doc2_results = [r for r in mode_results if r["id"] > 15]
    doc1_pass = sum(1 for r in doc1_results if r["passed"])
    doc2_total = len(doc2_results)
    doc2_pass = sum(1 for r in doc2_results if r["passed"])

    # v7 comparison (v7 had 20 questions)
    v7_mode = v7.get(mode, [])
    v7_pass = sum(1 for r in v7_mode if r["passed"]) if v7_mode else "?"
    v7_doc1 = sum(1 for r in v7_mode if r["id"] <= 15 and r["passed"]) if v7_mode else "?"
    v7_doc2 = sum(1 for r in v7_mode if r["id"] > 15 and r["passed"]) if v7_mode else "?"

    # Per-question details
    details = " ".join(
        f"Q{r['id']}:{'P' if r['passed'] else 'F'}"
        for r in sorted(mode_results, key=lambda x: x["id"])
    )

    print(f"\n  {mode:20s}: {passed}/{total} ({100*passed//total}%)")
    print(f"    Doc1 (RU): {doc1_pass}/15  Doc2 (EN): {doc2_pass}/{doc2_total}")
    if isinstance(v7_pass, int):
        print(f"    vs v7: {v7_pass}/20 (Doc1: {v7_doc1}/15, Doc2: {v7_doc2}/5)")
    print(f"    {details}")

total_passed = sum(sum(1 for r in mr if r["passed"]) for mr in results.values())
total_all = sum(len(mr) for mr in results.values())
doc1_total_pass = sum(sum(1 for r in mr if r["id"] <= 15 and r["passed"]) for mr in results.values())
doc2_total_pass = sum(sum(1 for r in mr if r["id"] > 15 and r["passed"]) for mr in results.values())
doc2_total_all = sum(sum(1 for r in mr if r["id"] > 15) for mr in results.values())

v7_total = sum(sum(1 for r in mr if r["passed"]) for mr in v7.values()) if v7 else "?"

print(f"\n{'=' * 70}")
print(f"  {'OVERALL':20s}: {total_passed}/{total_all} ({100*total_passed//total_all}%)")
print(f"    Doc1 (RU): {doc1_total_pass}/90  Doc2 (EN): {doc2_total_pass}/{doc2_total_all}")
if isinstance(v7_total, int):
    print(f"    vs v7 (20q): {v7_total}/120")

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

# Save results
out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "results_v10_full.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")

driver.close()
