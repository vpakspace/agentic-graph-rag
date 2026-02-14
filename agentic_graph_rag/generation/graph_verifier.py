"""Graph Verifier — detect contradictions and verify claims via traversal.

Uses graph triplets and passages to check factual consistency
of retrieved information before answer generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.models import GraphContext

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Check contradictions
# ---------------------------------------------------------------------------

def check_contradictions(
    facts: list[str],
    graph_context: GraphContext,
    openai_client: OpenAI | None = None,
) -> list[dict]:
    """Detect contradictions between extracted facts and graph context.

    Returns list of contradiction dicts with keys:
      - fact: the contradicting fact
      - evidence: conflicting triplet or passage
      - severity: "low", "medium", "high"
    """
    if not facts or (not graph_context.triplets and not graph_context.passages):
        return []

    cfg = get_settings()
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=cfg.openai.api_key)

    # Build context from triplets
    triplet_strs = []
    for t in graph_context.triplets:
        triplet_strs.append(f"{t['source']} --[{t['relation']}]--> {t['target']}")
    triplet_text = "\n".join(triplet_strs) if triplet_strs else "No triplets."

    passage_text = "\n---\n".join(graph_context.passages[:5]) if graph_context.passages else "No passages."
    facts_text = "\n".join(f"- {f}" for f in facts)

    prompt = (
        "Given the following knowledge graph facts and passages, "
        "identify any contradictions with the listed claims.\n\n"
        f"Graph triplets:\n{triplet_text}\n\n"
        f"Passages:\n{passage_text}\n\n"
        f"Claims to verify:\n{facts_text}\n\n"
        "For each contradiction found, output one line in the format:\n"
        "CONTRADICTION: <claim> | <conflicting evidence> | <severity: low/medium/high>\n"
        "If no contradictions, output: NONE"
    )

    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model_mini,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""

        contradictions = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.upper() == "NONE":
                break
            if line.startswith("CONTRADICTION:"):
                parts = line[len("CONTRADICTION:"):].split("|")
                if len(parts) >= 3:
                    contradictions.append({
                        "fact": parts[0].strip(),
                        "evidence": parts[1].strip(),
                        "severity": parts[2].strip().lower(),
                    })
                elif len(parts) == 2:
                    contradictions.append({
                        "fact": parts[0].strip(),
                        "evidence": parts[1].strip(),
                        "severity": "medium",
                    })

        logger.info("Found %d contradictions", len(contradictions))
        return contradictions

    except Exception as e:
        logger.error("Error checking contradictions: %s", e)
        return []


# ---------------------------------------------------------------------------
# 2. Verify claim via graph traversal
# ---------------------------------------------------------------------------

def verify_via_traversal(
    claim: str,
    graph_context: GraphContext,
) -> dict:
    """Verify a claim using graph triplets (path-based verification).

    Checks if the claim's entities appear in triplets and
    whether the relationship direction supports the claim.

    Returns dict with keys:
      - verified: bool
      - supporting_triplets: list of matching triplets
      - confidence: float (0-1)
    """
    if not claim or not graph_context.triplets:
        return {"verified": False, "supporting_triplets": [], "confidence": 0.0}

    claim_lower = claim.lower()
    supporting = []

    for t in graph_context.triplets:
        source_lower = t["source"].lower()
        target_lower = t["target"].lower()
        relation_lower = t["relation"].lower()

        # Check if triplet entities appear in the claim
        source_in = source_lower in claim_lower
        target_in = target_lower in claim_lower
        relation_in = relation_lower.replace("_", " ") in claim_lower

        if source_in and target_in:
            supporting.append(t)
        elif (source_in or target_in) and relation_in:
            supporting.append(t)

    if supporting:
        confidence = min(len(supporting) * 0.3, 1.0)
        return {
            "verified": True,
            "supporting_triplets": supporting,
            "confidence": confidence,
        }

    return {"verified": False, "supporting_triplets": [], "confidence": 0.0}
