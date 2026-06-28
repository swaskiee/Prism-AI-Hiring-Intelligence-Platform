"""
Score Fusion & Ranking
Team Hyperion | India Runs Hackathon | Track 1: Data & AI Challenge
Owner: Nitanshu Tak

PURPOSE
-------
Combine the outputs of all five layers into one final score per
candidate, apply the Layer 5 honeypot hard gate, and produce the final
top-100 ranking in the EXACT format the official validator
(validate_submission.py) checks.

INPUTS (one DataFrame per layer, all keyed by candidate_id)
-------------------------------------------------------------
- layer2_df  : candidate_id, semantic_fit_score, matched_requirements
               (semantic_scorer.run_semantic_scoring)
- layer3_df  : candidate_id, ..., disqualifier_vector, disqualifier_penalty
               (disqualifiers.run_structural_disqualifiers)
- layer4_df  : candidate_id, trust_multiplier, recency_score,
               response_component, interview_component,
               open_to_work_component (trust_multiplier.run_trust_multiplier)
- layer5_df  : candidate_id, ..., honeypot_flag, honeypot_evidence
               (honeypot_detection.run_honeypot_detection)

FUSION FORMULA
--------------
For every candidate NOT flagged as a honeypot:

    final_score = semantic_fit_score
                  * trust_multiplier
                  * (1 - disqualifier_penalty)

Rationale for this exact shape, in order:

1. semantic_fit_score is the base — "does this person's actual
   experience match what the JD needs." Everything else adjusts this
   base; nothing else can manufacture a high score on its own. A
   candidate with zero skill relevance cannot be pulled to the top by a
   great trust multiplier or a clean disqualifier record — multiplying
   by near-zero keeps the result near-zero, which is the correct
   behavior (a perfectly "available," "clean" candidate who simply isn't
   a skills match should not rank highly).

2. trust_multiplier (Layer 4) is applied MULTIPLICATIVELY, not added,
   specifically per the handoff doc and Redrob's own framing: a
   skill-perfect but unavailable candidate should be pulled DOWN
   proportionally to how unavailable they are, not just nudged by a flat
   penalty regardless of how good their skill match was. Bounded at
   MIN_MULTIPLIER (0.25, see trust_multiplier.py) rather than 0, so an
   excellent but currently-quiet candidate can still surface at a lower
   rank rather than being treated as disqualified — that judgment
   (disqualify vs. de-prioritize) is deliberately reserved for Layers 3
   and 5, not Layer 4.

3. (1 - disqualifier_penalty) converts Layer 3's soft 0.0-1.0 penalty
   into a multiplicative down-weight, consistent with the same
   "multiply, don't subtract" philosophy — and consistent with the
   handoff doc's explicit instruction that Layer 3's signal should be "a
   down-weighting multiplier... not a hard binary cutoff in isolation."
   A candidate who trips one soft rule (e.g., disqualifier_penalty=0.05,
   the lowest single-rule weight — stale_architect) loses only 5% of
   their score; a candidate tripping multiple rules loses proportionally
   more, exactly matching the handoff doc's requirement that "a candidate
   who trips one soft rule shouldn't be treated the same as someone who
   trips three hard ones."

HONEYPOT HARD GATE
-------------------
Per Layer 5's own explicit recommendation (honeypot_dev_notes.md,
"Recommendation to Nitanshu"): any candidate with honeypot_flag == True
is given final_score = 0.0 and is excluded from ranking consideration
entirely — this is NOT folded into the multiplicative formula above, it
is a hard pre-filter applied before fusion, because honeypots are forced
to relevance TIER 0 in the hidden ground truth and the >10%-in-top-100
threshold is a disqualification gate, not a scoring nuance.

TIE-BREAKING — REQUIRED BY THE OFFICIAL VALIDATOR
----------------------------------------------------
validate_submission.py enforces, verbatim: when two candidates have an
EQUAL score, the one with the LEXICOGRAPHICALLY SMALLER candidate_id must
receive the better (lower-numbered) rank. This is not a stylistic choice
— submitting a tied-score ranking in the wrong order fails the official
validator outright. assign_ranks() below implements this exactly by
sorting on (-final_score, candidate_id) ascending.

OUTPUT
------
get_final_ranking() returns a pandas DataFrame, exactly the top 100
candidates, with columns:

    candidate_id    str
    rank            int, 1-100, each used exactly once
    score           float, non-increasing as rank increases
    semantic_fit_score, trust_multiplier, disqualifier_penalty,
    matched_requirements, disqualifier_vector
                    (kept for the Explainability Engine and for our own
                    Stage-5 debugging — NOT written to the final
                    submission CSV, which per submission_spec.docx must
                    contain exactly candidate_id, rank, score, reasoning
                    and no other columns)
"""

from __future__ import annotations

import pandas as pd


def fuse_and_rank(
    layer2_df: pd.DataFrame,
    layer3_df: pd.DataFrame,
    layer4_df: pd.DataFrame,
    layer5_df: pd.DataFrame,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Merge all four layer outputs on candidate_id, compute final_score per
    the fusion formula, hard-exclude honeypots, and return the top_n
    ranked candidates with full component breakdown (for explainability
    and debugging — trimmed to the official 4 columns only when writing
    the submission CSV, see rank.py).

    Parameters
    ----------
    layer2_df, layer3_df, layer4_df, layer5_df : pd.DataFrame
        Outputs of the four upstream layers, as documented in the module
        docstring above. Each must contain a candidate_id column.
    top_n : int
        How many top-ranked candidates to return (100, per the official
        submission spec).

    Returns
    -------
    pd.DataFrame
        Exactly top_n rows, columns: candidate_id, rank, score, plus
        component columns for explainability (semantic_fit_score,
        trust_multiplier, disqualifier_penalty, matched_requirements,
        disqualifier_vector, honeypot_flag).
    """
    merged = (
        layer2_df[["candidate_id", "semantic_fit_score", "matched_requirements"]]
        .merge(
            layer3_df[["candidate_id", "disqualifier_vector", "disqualifier_penalty"]],
            on="candidate_id", how="inner",
        )
        .merge(
            layer4_df[["candidate_id", "trust_multiplier"]],
            on="candidate_id", how="inner",
        )
        .merge(
            layer5_df[["candidate_id", "honeypot_flag", "honeypot_evidence"]],
            on="candidate_id", how="inner",
        )
    )

    expected = len(layer2_df)
    if len(merged) != expected:
        raise ValueError(
            f"Merge produced {len(merged)} rows but expected {expected} — "
            f"candidate_id mismatch between layers. Check that all four "
            f"layer DataFrames were run on the exact same candidate set."
        )

    # --- Fusion formula (see module docstring) ---
    merged["final_score"] = (
        merged["semantic_fit_score"]
        * merged["trust_multiplier"]
        * (1.0 - merged["disqualifier_penalty"])
    )

    # --- Honeypot hard gate: zero out and exclude entirely (Layer 5's own
    #     explicit recommendation) ---
    merged.loc[merged["honeypot_flag"], "final_score"] = 0.0
    eligible = merged[~merged["honeypot_flag"]].copy()

    ranked = assign_ranks(eligible, score_col="final_score", top_n=top_n)
    return ranked


def assign_ranks(df: pd.DataFrame, score_col: str = "final_score", top_n: int = 100) -> pd.DataFrame:
    """
    Sort by score descending with the official tie-break rule (equal
    scores -> smaller candidate_id ranks better), then assign rank 1..N
    and return exactly the top_n rows.

    This function exists standalone (not just inlined into fuse_and_rank)
    so the EXACT same tie-break logic can also be applied as a final
    sanity pass in rank.py right before writing the submission CSV — see
    that file's closing validation step.
    """
    sorted_df = df.sort_values(
        by=[score_col, "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    sorted_df = sorted_df.head(top_n).copy()
    sorted_df["rank"] = range(1, len(sorted_df) + 1)
    sorted_df["score"] = sorted_df[score_col].round(6)
    return sorted_df


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

    sample_path = os.path.join(base, "..", "layer3", "sample_candidates.json")
    with open(sample_path) as f:
        sample = json.load(f)

    jd_reqs = get_jd_requirements()

    start = time.perf_counter()
    l2 = run_semantic_scoring(sample, jd_requirements=jd_reqs)
    l3 = run_structural_disqualifiers(sample, jd_requirements=jd_reqs)
    l4 = run_trust_multiplier(sample)
    l5 = run_honeypot_detection(sample)
    final = fuse_and_rank(l2, l3, l4, l5, top_n=10)
    elapsed = time.perf_counter() - start

    print(f"Ran full fusion pipeline on {len(sample)} sample candidates in {elapsed:.4f}s\n")
    print(final[[
        "rank", "candidate_id", "score", "semantic_fit_score",
        "trust_multiplier", "disqualifier_penalty", "matched_requirements",
    ]].to_string(index=False))
