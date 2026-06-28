"""
Layer 4 — Behavioral Trust Multiplier
Team Hyperion | India Runs Hackathon | Track 1: Data & AI Challenge
Owner: Nitanshu Tak

PURPOSE
-------
Apply Redrob's own stated insight as a real, computed signal rather than
an afterthought:

    "A perfect-on-paper candidate who hasn't logged in for 6 months and
    has a 5% response rate is, for hiring purposes, not actually
    available." (redrob_signals_doc — paraphrased framing used throughout
    the team's planning docs)

A candidate with a perfect semantic_fit_score (Layer 2) and a clean
disqualifier_penalty (Layer 3) who has gone quiet for six months is not a
top-10 recommendation for an active hiring need. This layer encodes that
as a MULTIPLIER applied to the skill-fit score in Score Fusion — not as
an additive bonus/penalty — specifically so that a low trust score pulls
a high skill score DOWN proportionally, rather than just nudging rank
order at the margins.

WHICH OF THE 23 redrob_signals FIELDS ARE USED, AND WHY
----------------------------------------------------------
Four fields, deliberately — chosen because they answer one specific
question this layer is responsible for: "if Redrob reached out to this
candidate today, would they actually respond and engage?" Fields that
answer a DIFFERENT question (e.g. expected_salary_range_inr_lpa, which is
a negotiation/fit question, not an availability question) are deliberately
left for a future layer or out of scope, rather than folded in here just
because they exist in the schema.

1. recruiter_response_rate (0.0-1.0)
   The single most direct measure of "will this person respond when
   contacted." Used near-linearly.

2. last_active_date -> recency_score
   Converted to a recency decay: very recent login keeps full weight; a
   candidate inactive for many months decays toward a floor. Mirrors
   Redrob's own "6 months inactive" framing directly.

3. interview_completion_rate (0.0-1.0)
   Distinguishes "responds to messages" from "actually shows up and
   follows through" — a candidate could have a fine response rate but a
   history of no-showing scheduled interviews, which is a real and
   different trust signal.

4. open_to_work_flag (bool)
   A direct, explicit self-declaration. Given real weight as a binary
   gate-like factor rather than ignored just because it's coarse — a
   candidate who has explicitly marked themselves NOT open to work is a
   meaningfully different case from one who simply has a so-so response
   rate, and the multiplier reflects that distinction.

NOT used: profile_views_received_30d, search_appearance_30d,
saved_by_recruiters_30d (these measure how OTHER recruiters already
perceive the candidate, which would let this layer's score be influenced
by other companies' hiring activity rather than this candidate's own
behavior — a confound, not a signal we want); expected_salary_range,
preferred_work_mode, willing_to_relocate (negotiation/logistics fit,
arguably belongs in Layer 3 or a future layer, not "trust"); verified_*
flags and connection_count/endorsements_received (platform-trust/vanity
metrics, weakly related to availability specifically).

OUTPUT
------
A pandas DataFrame, one row per candidate_id, with columns:

    candidate_id            str
    trust_multiplier         float, MIN_MULTIPLIER-1.0
    recency_score            float, 0.0-1.0 (component, kept for
                              explainability/debugging)
    response_component       float, 0.0-1.0 (component)
    interview_component      float, 0.0-1.0 (component)
    open_to_work_component   float, 0.0-1.0 (component)

trust_multiplier is meant to MULTIPLY semantic_fit_score in Score Fusion,
not be summed with it — see score_fusion.py.

PERFORMANCE
-----------
Fully vectorizable with pandas/numpy (no nested-structure dependency like
Layers 3/5 have, since redrob_signals is a flat dict per candidate) — runs
in well under a second on the full 100,000 candidates. See benchmark.py.
"""

from __future__ import annotations
from datetime import date, datetime
from typing import List

import numpy as np
import pandas as pd

# Same reference-date convention as Layer 5 (honeypot_detection.py's
# DEFAULT_REFERENCE_DATE), kept consistent across the pipeline rather than
# each layer picking its own "today."
DEFAULT_REFERENCE_DATE = date(2026, 6, 28)

# Recency decay: a candidate active within RECENCY_FULL_CREDIT_DAYS gets
# full credit (1.0). Past that, recency_score decays linearly down to
# RECENCY_FLOOR at RECENCY_ZERO_CREDIT_DAYS and beyond. Chosen to mirror
# Redrob's own "6 months inactive" framing as the point where a candidate
# is clearly no longer "actually available" — 180 days is given partial
# credit on the way down, not a sudden cliff, since recency is a gradient,
# not a binary.
RECENCY_FULL_CREDIT_DAYS = 30
RECENCY_ZERO_CREDIT_DAYS = 180
RECENCY_FLOOR = 0.05

# The overall trust_multiplier is bounded below by this floor rather than
# allowed to reach 0.0 — even a candidate with a poor behavioral profile
# might still be worth surfacing at a much lower rank if their skill fit
# is exceptional (a hard 0 would make Layer 4 behave like a second
# disqualifier gate, which is Layer 3/5's job, not this layer's).
MIN_MULTIPLIER = 0.25

# Relative weights of the four components inside the multiplier (must sum
# to 1.0). response_rate and recency are weighted highest since they most
# directly answer "would this person respond if contacted today."
COMPONENT_WEIGHTS = {
    "response": 0.35,
    "recency": 0.30,
    "interview": 0.20,
    "open_to_work": 0.15,
}


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _recency_score(last_active_date_str: str, reference_date: date) -> float:
    last_active = _parse_date(last_active_date_str)
    days_inactive = (reference_date - last_active).days
    if days_inactive <= RECENCY_FULL_CREDIT_DAYS:
        return 1.0
    if days_inactive >= RECENCY_ZERO_CREDIT_DAYS:
        return RECENCY_FLOOR
    # Linear decay between the two thresholds.
    span = RECENCY_ZERO_CREDIT_DAYS - RECENCY_FULL_CREDIT_DAYS
    progress = (days_inactive - RECENCY_FULL_CREDIT_DAYS) / span
    return 1.0 - progress * (1.0 - RECENCY_FLOOR)


def run_trust_multiplier(
    candidates: List[dict], reference_date: date = DEFAULT_REFERENCE_DATE
) -> pd.DataFrame:
    """
    Compute the behavioral trust multiplier for every candidate.

    Parameters
    ----------
    candidates : list of dict
        Candidate records matching candidate_schema.json.
    reference_date : date, optional
        "Today," for recency math. Defaults to DEFAULT_REFERENCE_DATE,
        kept consistent with Layer 5's own default.

    Returns
    -------
    pd.DataFrame
        Columns documented in the module docstring.
    """
    rows = []
    for c in candidates:
        signals = c["redrob_signals"]

        recency = _recency_score(signals["last_active_date"], reference_date)
        response = float(np.clip(signals.get("recruiter_response_rate", 0.0), 0.0, 1.0))
        interview = float(np.clip(signals.get("interview_completion_rate", 0.0), 0.0, 1.0))
        open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.4
        # NOTE: open_to_work uses 0.4, not 0.0, as its "false" value — a
        # candidate not currently flagged open-to-work is a weaker signal,
        # not a disqualifying one (people change jobs, update flags late,
        # or are open to the right opportunity despite the flag). This
        # mirrors Layer 3's philosophy of soft signals over hard cutoffs
        # wherever the underlying judgment is genuinely fuzzy.

        composite = (
            COMPONENT_WEIGHTS["response"] * response
            + COMPONENT_WEIGHTS["recency"] * recency
            + COMPONENT_WEIGHTS["interview"] * interview
            + COMPONENT_WEIGHTS["open_to_work"] * open_to_work
        )

        # Map the 0.0-1.0 composite into [MIN_MULTIPLIER, 1.0] rather than
        # [0.0, 1.0] directly — see MIN_MULTIPLIER's docstring above.
        multiplier = MIN_MULTIPLIER + composite * (1.0 - MIN_MULTIPLIER)

        rows.append({
            "candidate_id": c["candidate_id"],
            "trust_multiplier": round(multiplier, 4),
            "recency_score": round(recency, 4),
            "response_component": round(response, 4),
            "interview_component": round(interview, 4),
            "open_to_work_component": round(open_to_work, 4),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json
    import time
    import os

    sample_path = os.path.join(os.path.dirname(__file__), "sample_candidates.json")
    with open(sample_path) as f:
        sample = json.load(f)

    start = time.perf_counter()
    result = run_trust_multiplier(sample)
    elapsed = time.perf_counter() - start

    print(f"Ran Layer 4 on {len(sample)} sample candidates in {elapsed:.4f}s\n")
    print(result.sort_values("trust_multiplier", ascending=False).to_string(index=False))
