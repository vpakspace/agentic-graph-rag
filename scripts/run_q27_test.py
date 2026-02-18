#!/usr/bin/env python3
"""Run Q27 test — semantic judge evaluation.

Filters questions to Q27 only, runs all 6 modes, prints results.
Validates that the semantic judge (embedding similarity + reference answer)
correctly evaluates enumeration answers.
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

# Filter to Q27 only
questions = [q for q in load_questions() if q["id"] == 27]
assert len(questions) == 1, f"Expected 1 question, got {len(questions)}"
print(f"Q27: {questions[0]['question_ru']}")
print(f"Reference answer: {questions[0].get('reference_answer', 'NONE')[:80]}...")
print()

modes = ["vector", "cypher", "hybrid", "agent_pattern", "agent_llm", "agent_mangle"]
results = run_benchmark(driver, openai_client, modes=modes, questions=questions, lang="ru")

print("\n" + "=" * 70)
print("Q27 TEST — Semantic Judge (embedding similarity + reference answer)")
print("=" * 70)

total_passed = 0
total_modes = 0
for mode, mode_results in results.items():
    r = mode_results[0]
    status = "PASS" if r["passed"] else "FAIL"
    total_passed += int(r["passed"])
    total_modes += 1
    print(f"  {mode:20s}: {status}  (conf={r['confidence']:.2f}, latency={r['latency']:.1f}s)")
    # Print first 200 chars of answer for debugging
    answer_preview = r["answer"][:200].replace("\n", " ")
    print(f"    Answer: {answer_preview}...")

print(f"\n  OVERALL: {total_passed}/{total_modes}")
print("  Target: 4-6/6")

# Save results
out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "results_q27_test.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n  Saved to {out_path}")

driver.close()
