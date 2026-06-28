"""
Explainability Engine — Reasoning Generation
Team Hyperion | India Runs Hackathon | Track 1: Data & AI Challenge
Owner: Nitanshu Tak

PURPOSE
-------
Generate the `reasoning` column for each of the top 100 candidates,
written directly from the same computed sub-scores that produced their
rank — never templated boilerplate, never a claim that doesn't trace back
to an actual field in that candidate's record.

This matters specifically because of Stage 4 (manual review). Per
submission_spec.docx, reviewers sample 10 rows by hand and check
reasoning for:
  - hallucination (claims about skills/employers not actually present)
  - rank-consistency (does the tone match the rank — a rank-5 reasoning
    string should not read as lukewarm, a rank-95 string should not read
    as glowing)
  - genuine substance vs. empty/repeated boilerplate

WHY NO LLM CALL HERE
----------------------
Two independent reasons, not one:
1. Network is disabled at ranking time (hard constraint, all layers).
2. Even if it weren't: generating reasoning via a second LLM pass creates
   exactly the hallucination risk Stage 4 is designed to catch — an LLM
   asked to "explain why this candidate ranks well" can easily invent a
   plausible-sounding but unverified detail. Building the sentence
   directly from verified fields makes hallucination structurally
   impossible, not just unlikely.

HOW THE SENTENCE IS BUILT
----------------------------
Each reasoning string pulls from, in priority order:
  1. years_of_experience + current_title (always available, grounds the
     sentence in a concrete, verifiable fact)
  2. matched_requirements from Layer 2 (semantic_fit_score's component
     breakdown) — names the SPECIFIC JD requirement(s) this candidate's
     career history matches, using Layer 1's must_have_detail/
     nice_to_have_detail human-readable text, not the raw code
  3. trust signal context from Layer 4, only when notably high or low
     (recency_score and response_component) — keeps the sentence honest
     about availability, not just skill
  4. disqualifier context from Layer 3, ONLY if disqualifier_penalty > 0
     — for a candidate who still made the top 100 despite a soft flag,
     naming it directly is more defensible at Stage 4 than silently
     omitting it
  5. a final integrative clause for candidates with no specific matched
     requirement (low semantic_fit_score) — describes the position in
     the ranking honestly (e.g., "weaker overall fit") rather than
     fabricating a positive claim with no support

Every numeric value quoted in the sentence (years of experience,
response rate, recency) is read directly from the candidate's own record
or from this same pipeline's own computed scores — never invented.
"""

from __future__ import annotations
from typing import Dict, List

import pandas as pd


def _format_requirement_names(codes_str: str, jd_requirements: dict) -> List[str]:
    """Turn a comma-joined string of requirement codes into short human-readable phrases."""
    if not codes_str:
        return []
    detail_map: Dict[str, str] = {}
    detail_map.update(jd_requirements.get("must_have_detail", {}))
    detail_map.update(jd_requirements.get("nice_to_have_detail", {}))

    # Short human phrase per code — first clause of the full detail text,
    # kept brief so the reasoning sentence doesn't balloon when several
    # requirements match at once.
    short_phrases = {
        "production_embeddings_retrieval": "production embeddings-based retrieval",
        "production_vector_db_or_hybrid_search": "production vector DB / hybrid search experience",
        "strong_python_demonstrated_in_systems": "demonstrated Python systems experience",
        "ranking_evaluation_framework_experience": "ranking evaluation framework experience",
        "llm_fine_tuning_lora_qlora_peft": "LLM fine-tuning (LoRA/QLoRA/PEFT)",
        "learning_to_rank_xgboost_or_neural": "learning-to-rank modeling",
        "hr_tech_recruiting_marketplace_background": "HR-tech/marketplace background",
        "distributed_systems_or_inference_optimization": "distributed systems / inference optimization",
        "open_source_ai_ml_contributions": "open-source AI/ML contributions",
    }
    codes = codes_str.split(",")
    return [short_phrases.get(code, code.replace("_", " ")) for code in codes if code]


def generate_reasoning(row: pd.Series, candidate_lookup: Dict[str, dict], jd_requirements: dict) -> str:
    """
    Build one reasoning string for a single ranked candidate row.

    Parameters
    ----------
    row : pd.Series
        One row from the fused ranking DataFrame (score_fusion.fuse_and_rank
        output) — must include candidate_id, semantic_fit_score,
        trust_multiplier, disqualifier_penalty, matched_requirements,
        disqualifier_vector.
    candidate_lookup : dict
        candidate_id -> full candidate record (candidate_schema.json
        shape), for reading years_of_experience / current_title /
        current_company directly rather than re-deriving them.
    jd_requirements : dict
        Layer 1's structured JD object, for translating requirement codes
        and disqualifier codes into human-readable phrases.

    Returns
    -------
    str
        A 1-2 sentence, fully grounded reasoning string.
    """
    candidate = candidate_lookup[row["candidate_id"]]
    profile = candidate["profile"]
    years = profile["years_of_experience"]
    title = profile["current_title"]
    company = profile["current_company"]

    clauses = [f"{years} years of experience, currently {title} at {company}."]

    matched = _format_requirement_names(row.get("matched_requirements", ""), jd_requirements)
    if matched:
        if len(matched) == 1:
            clauses.append(f"Career history shows direct evidence of {matched[0]}.")
        else:
            joined = ", ".join(matched[:-1]) + f", and {matched[-1]}"
            clauses.append(f"Career history shows direct evidence of {joined}.")
    else:
        clauses.append(
            "No strong direct evidence of the JD's specific must-have requirements was "
            "found in career history text; ranked primarily on other signals below."
        )

    trust_mult = row["trust_multiplier"]
    if trust_mult >= 0.85:
        clauses.append("Behavioral signals (response rate, recent activity) indicate a highly engaged, available candidate.")
    elif trust_mult <= 0.45:
        clauses.append(
            "Caution: behavioral signals (low recent activity and/or response rate) suggest this "
            "candidate may not currently be responsive to outreach despite the skill match."
        )

    penalty = row["disqualifier_penalty"]
    if penalty > 0:
        fired_codes = row.get("disqualifier_vector", "")
        detail_map = jd_requirements.get("hard_disqualifiers_detail", {})
        descriptions = [detail_map.get(code, code) for code in fired_codes.split(",") if code]
        if descriptions:
            clauses.append(
                f"Note: this profile triggered a structural concern (penalty applied: "
                f"{descriptions[0]})."
            )

    return " ".join(clauses)


def generate_reasoning_for_ranking(
    ranked_df: pd.DataFrame, candidates: List[dict], jd_requirements: dict
) -> pd.DataFrame:
    """
    Apply generate_reasoning() across every row of a ranked DataFrame.

    Parameters
    ----------
    ranked_df : pd.DataFrame
        Output of score_fusion.fuse_and_rank() (or assign_ranks()).
    candidates : list of dict
        Full candidate records, used to build the candidate_id lookup.
    jd_requirements : dict
        Layer 1's structured JD object.

    Returns
    -------
    pd.DataFrame
        `ranked_df` with an added `reasoning` column.
    """
    lookup = {c["candidate_id"]: c for c in candidates}
    result = ranked_df.copy()
    result["reasoning"] = result.apply(
        lambda row: generate_reasoning(row, lookup, jd_requirements), axis=1
    )
    return result


if __name__ == "__main__":
    import json
    import os
    import sys
    import time

    base = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(base, "..", "layer1"))
    sys.path.insert(0, os.path.join(base, "..", "layer2"))
    sys.path.insert(0, os.path.join(base, "..", "layer3"))
    sys.path.insert(0, os.path.join(base, "..", "layer4"))
    sys.path.insert(0, os.path.join(base, "..", "layer5"))

    from jd_requirements import get_jd_requirements  # noqa: E402
    from semantic_scorer import run_semantic_scoring  # noqa: E402
    from disqualifiers import run_structural_disqualifiers  # noqa: E402
    from trust_multiplier import run_trust_multiplier  # noqa: E402
    from honeypot_detection import run_honeypot_detection  # noqa: E402
    from score_fusion import fuse_and_rank  # noqa: E402

    sample_path = os.path.join(base, "..", "layer3", "sample_candidates.json")
    with open(sample_path) as f:
        sample = json.load(f)

    jd_reqs = get_jd_requirements()

    start = time.perf_counter()
    l2 = run_semantic_scoring(sample, jd_requirements=jd_reqs)
    l3 = run_structural_disqualifiers(sample, jd_requirements=jd_reqs)
    l4 = run_trust_multiplier(sample)
    l5 = run_honeypot_detection(sample)
    final = fuse_and_rank(l2, l3, l4, l5, top_n=5)
    final_with_reasoning = generate_reasoning_for_ranking(final, sample, jd_reqs)
    elapsed = time.perf_counter() - start

    print(f"Ran full pipeline + reasoning on {len(sample)} sample candidates in {elapsed:.4f}s\n")
    for _, row in final_with_reasoning.iterrows():
        print(f"Rank {row['rank']} | {row['candidate_id']} | score={row['score']}")
        print(f"  {row['reasoning']}\n")
