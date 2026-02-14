"""Benchmark comparison — generate comparison tables from results."""

from __future__ import annotations

from typing import Any


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics for a single mode's results."""
    total = len(results)
    if total == 0:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "avg_confidence": 0.0,
            "avg_latency": 0.0,
            "avg_retries": 0.0,
            "total_retries": 0,
        }

    correct = sum(1 for r in results if r["passed"])
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "avg_confidence": round(
            sum(r["confidence"] for r in results) / total, 3,
        ),
        "avg_latency": round(
            sum(r["latency"] for r in results) / total, 3,
        ),
        "avg_retries": round(
            sum(r["retries"] for r in results) / total, 2,
        ),
        "total_retries": sum(r["retries"] for r in results),
    }


def compare_modes(
    all_results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Generate a comparison table across all modes.

    Returns list of dicts suitable for Streamlit dataframe display.
    """
    rows: list[dict[str, Any]] = []
    for mode_name, results in all_results.items():
        m = compute_metrics(results)
        rows.append({
            "Mode": mode_name,
            "Accuracy": f"{m['correct']}/{m['total']} ({m['accuracy']:.0%})",
            "Avg Confidence": m["avg_confidence"],
            "Avg Latency (s)": m["avg_latency"],
            "Avg Retries": m["avg_retries"],
        })
    return rows


def accuracy_by_type(
    all_results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Compute per-query-type accuracy for each mode.

    Returns dict[mode → dict[query_type → accuracy]].
    """
    breakdown: dict[str, dict[str, float]] = {}

    for mode_name, results in all_results.items():
        type_groups: dict[str, list[bool]] = {}
        for r in results:
            qtype = r.get("type", "unknown")
            type_groups.setdefault(qtype, []).append(r["passed"])

        breakdown[mode_name] = {
            qtype: sum(passed_list) / len(passed_list) if passed_list else 0.0
            for qtype, passed_list in type_groups.items()
        }

    return breakdown
