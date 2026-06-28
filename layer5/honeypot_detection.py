"""
Layer 5 — Honeypot & Anomaly Detection
Team Hyperion | India Runs Hackathon | Track 1: Data & AI Challenge
Owner: Swati Dubey

WHY THIS LAYER IS THE HIGHEST-STAKES PIECE IN THE WHOLE PIPELINE
------------------------------------------------------------------
Per submission_spec.docx, Section 7 ("Honeypot warning") and Section 5
(Stage 3): the dataset contains ~80 honeypot candidates with subtly
impossible profiles, forced to relevance tier 0 in the hidden ground
truth. If the team's submitted top-100 contains a honeypot rate above
10% (i.e. more than 10 honeypots in the top 100), THE ENTIRE TEAM IS
DISQUALIFIED AT STAGE 3 — regardless of how good the rest of the ranking
is. This is a hard pass/fail gate, not a scoring penalty.

This makes Layer 5 fundamentally different from Layer 3 (structural
disqualifiers): Layer 3's rules feed into a soft score that Nitanshu
blends with the semantic score. Layer 5's job is more binary and
defensive — it must reliably catch the deliberately-broken profiles
before they can rank highly, because a single missed cluster of
honeypots could end the team's participation entirely.

WHAT COUNTS AS A HONEYPOT — PER THE OFFICIAL SPEC, VERBATIM
------------------------------------------------------------------
From submission_spec.docx, Section 7:
    "...honeypot candidates with subtly impossible profiles (e.g., 8
    years of experience at a company founded 3 years ago; 'expert'
    proficiency in 10 skills with 0 years used)."

These are the ONLY two example mechanisms named explicitly in any
official document. Both were independently verified against the real
100,000-candidate dataset before writing any detector code (see
DATA-GROUNDED FINDINGS below) — this module does not guess at additional
honeypot types beyond what was actually confirmed present in the data.

IMPORTANT SCHEMA NOTE: there is no "company founding year" field anywhere
in candidate_schema.json. The "8 years at a company founded 3 years ago"
example cannot be checked via an external company-age lookup — no such
field exists for the candidate to reference, and no separate companies
table is provided in the hackathon bundle. Investigation of the real
dataset (see below) found the actual synthetic implementation of this
example: it manifests as an internal inconsistency between a current
role's start_date and its stated duration_months — fully checkable from
data already present in the candidate record, with no external lookup
required.

DATA-GROUNDED FINDINGS (verified against the full 100,000-row
candidates.jsonl before writing detector code; see honeypot_dev_notes.md
for the full derivation)
------------------------------------------------------------------
1. DURATION-DATE MISMATCH (honeypot_duration_mismatch)
   33 career_history entries, across 33 distinct candidates, where a
   CURRENT role (is_current=True) states a duration_months value that
   contradicts the actual elapsed time between start_date and today by
   more than 6 months. Example found: CAND_0007353's current "Frontend
   Engineer" role started 2023-09-10 (≈33 months before the dataset's
   reference date) but claims duration_months=166 (≈13.8 years) — exactly
   the shape of the JD's "8 years at a company founded 3 years ago"
   example, expressed as an internal date/duration contradiction instead
   of an external company-age lookup.

   Confirmed this pattern occurs EXCLUSIVELY on is_current=True entries
   (0 occurrences on closed-out past roles) — consistent with this being
   a deliberately injected anomaly rather than natural data noise.

   HONEST NOTE: the mismatch goes in both directions — 19 of the 33
   flagged entries OVERSTATE duration (claim more tenure than the dates
   allow, matching the JD's "experience exceeds plausible age" framing
   exactly), while 14 UNDERSTATE it (claim less tenure than the dates
   allow). Both directions are kept as flags here, because both are
   genuine internal data inconsistencies — a candidate whose own
   start_date and duration_months contradict each other is unreliable
   data regardless of which direction the error points, and the spec's
   framing ("subtly impossible profiles") is broader than just
   experience-inflation. This is reported honestly rather than narrowing
   the rule to only the overstatement direction to make it match the
   JD's specific example more neatly.

2. ZERO-DURATION EXPERT SKILL (honeypot_expert_zero_duration)
   21 distinct candidates each have 3-5 skills simultaneously marked
   proficiency="expert" with duration_months=0 (and typically very low
   or zero endorsements too) — this is the JD's named example almost
   verbatim ("'expert' proficiency in 10 skills with 0 years used").
   Example found: CAND_0016000 has 5 such skills (TypeScript, Go, Docker,
   Hadoop, Photoshop, all expert/0-months/near-zero-endorsements) mixed
   in among otherwise normal-looking skills with realistic durations
   (16, 8, 9, 12 months) — confirming this is a deliberate injection, not
   noise.

   These two patterns are confirmed NON-OVERLAPPING (0 candidates trip
   both), giving 54 distinct flagged candidates total from these two
   mechanisms — close to, though not exactly matching, the spec's
   approximate "~80" figure. Several OTHER candidate irregularities were
   investigated and explicitly REJECTED as honeypot signals because they
   were far too common to be a deliberate ~80-candidate injection (e.g.
   career-history-start preceding education-start affects 3,457
   candidates; skill duration exceeding total career length affects
   3,392 candidates) — these are normal dataset noise, not honeypots,
   and are NOT used as detection criteria here. See honeypot_dev_notes.md
   for the full list of patterns investigated and rejected, with reasoning.

OUTPUT CONTRACT
------------------------------------------------------------------
run_honeypot_detection(candidates) returns a pandas DataFrame, one row
per candidate_id, with columns:

    candidate_id                       str
    honeypot_duration_mismatch_flag    bool
    honeypot_expert_zero_duration_flag bool
    honeypot_flag                      bool   — True if ANY rule fired
    honeypot_evidence                  str    — human-readable reason(s), for the reasoning column / Stage 4 review

Unlike Layer 3's disqualifier_penalty (a soft 0.0-1.0 weight blended into
scoring), honeypot_flag is intentionally a hard boolean. Per the spec,
honeypots are forced to relevance TIER 0 in the ground truth — there is
no "partial honeypot." This module's recommendation to Nitanshu (Layer 4)
is: any candidate with honeypot_flag=True should be given a score of 0 or
excluded entirely from the top-100, full stop — not down-weighted like a
Layer 3 soft signal.

PERFORMANCE
------------------------------------------------------------------
Must run within the team's shared 5-minute/16GB/CPU-only budget. Like
Layer 3, this module loops in Python over candidates rather than using
true pandas vectorization, because the underlying checks depend on
nested career_history/skills arrays. See benchmark.py for measured
runtime/memory on the full 100,000-candidate file.
"""

from __future__ import annotations
from datetime import date, datetime
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Reference "today" for date-math against open-ended (is_current=True) roles.
# Set to the dataset's apparent reference point. This module exposes it as a
# parameter (see run_honeypot_detection) so it isn't silently hardcoded if
# the dataset is regenerated with a different reference date later.
# ---------------------------------------------------------------------------

DEFAULT_REFERENCE_DATE = date(2026, 6, 28)

# Tolerance, in months, before a duration/date mismatch is treated as a
# genuine anomaly rather than rounding noise. Chosen after inspecting the
# real mismatches found in the dataset, which were all much larger than
# this (smallest gap observed: actual=13mo vs stated=61mo — a 48-month
# gap) — 6 months gives generous slack for any legitimate day-of-month
# rounding while staying far below the smallest real anomaly found.
DURATION_MISMATCH_TOLERANCE_MONTHS = 6


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_duration_date_mismatch(
    candidate: dict, reference_date: date = DEFAULT_REFERENCE_DATE
) -> List[str]:
    """
    Detects the "experience exceeds plausible tenure" honeypot mechanism.

    For each is_current=True career_history entry, compares the stated
    duration_months against the actual elapsed time from start_date to
    reference_date. If they differ by more than
    DURATION_MISMATCH_TOLERANCE_MONTHS, this is flagged.

    Deliberately scoped to is_current=True entries only: verified against
    the full dataset that 100% of real mismatches occur on current roles
    (closed-out past roles never show this pattern) — checking past roles
    too would be checking start_date vs end_date math, which was already
    separately confirmed to be internally consistent everywhere in this
    dataset (i.e. that's not where the anomaly lives), so skipping it here
    is a deliberate scope decision, not an oversight.

    Returns
    -------
    list of str
        One human-readable evidence string per career_history entry that
        trips this check (usually 0 or 1 entries, since a candidate
        normally has only one is_current=True role; the function returns
        a list defensively in case the data ever contains more than one).
    """
    evidence = []
    for entry in candidate["career_history"]:
        if not entry.get("is_current"):
            continue
        start = _parse_date(entry["start_date"])
        actual_months = _months_between(start, reference_date)
        stated_months = entry["duration_months"]
        gap = abs(actual_months - stated_months)
        if gap > DURATION_MISMATCH_TOLERANCE_MONTHS:
            direction = "OVERSTATES" if stated_months > actual_months else "UNDERSTATES"
            evidence.append(
                f"Current role '{entry['title']}' at {entry['company']} started "
                f"{entry['start_date']} (~{actual_months} months ago), but "
                f"duration_months={stated_months} ({stated_months / 12:.1f} years) "
                f"{direction} this by {gap} months — an internal date/duration "
                f"inconsistency."
            )
    return evidence


def detect_expert_zero_duration_skills(candidate: dict, min_flagged_skills: int = 1) -> List[str]:
    """
    Detects the "'expert' proficiency in N skills with 0 years used"
    honeypot mechanism, per the JD's example almost verbatim.

    Fires per individual skill where proficiency == "expert" and
    duration_months == 0. A single such skill is already a clear logical
    impossibility (you cannot be an "expert" in something you've used for
    zero time) — min_flagged_skills defaults to 1 rather than requiring
    multiple, since even one occurrence is independently disqualifying as
    a data-integrity red flag, not something that needs corroboration.

    Note: in the real dataset, every genuine honeypot candidate found by
    this rule actually has 3-5 such skills simultaneously (never just 1),
    which is reported in the evidence string for context — but the
    detection threshold itself stays conservative at 1, since the logical
    impossibility doesn't become "more true" with more instances; it's
    already fully disqualifying on its own.

    Returns
    -------
    list of str
        One evidence string per flagged skill.
    """
    flagged = [
        s for s in candidate["skills"]
        if s.get("proficiency") == "expert" and s.get("duration_months", -1) == 0
    ]
    if len(flagged) < min_flagged_skills:
        return []
    return [
        f"Skill '{s['name']}' marked proficiency=expert with duration_months=0 "
        f"(endorsements={s.get('endorsements', 'unknown')})."
        for s in flagged
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_honeypot_detection(
    candidates: List[dict], reference_date: date = DEFAULT_REFERENCE_DATE
) -> pd.DataFrame:
    """
    Run all Layer 5 honeypot detectors across `candidates`.

    Parameters
    ----------
    candidates : list of dict
        Candidate records matching candidate_schema.json.
    reference_date : date, optional
        The "today" used for date-math against open-ended current roles.
        Defaults to DEFAULT_REFERENCE_DATE.

    Returns
    -------
    pd.DataFrame
        One row per candidate, per the module's documented output contract.
    """
    rows = []
    for c in candidates:
        duration_evidence = detect_duration_date_mismatch(c, reference_date=reference_date)
        expert_evidence = detect_expert_zero_duration_skills(c)

        all_evidence = duration_evidence + expert_evidence
        rows.append({
            "candidate_id": c["candidate_id"],
            "honeypot_duration_mismatch_flag": len(duration_evidence) > 0,
            "honeypot_expert_zero_duration_flag": len(expert_evidence) > 0,
            "honeypot_flag": len(all_evidence) > 0,
            "honeypot_evidence": " | ".join(all_evidence),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Submission-time safety check — meant to be called on the FINAL top-100
# ranking just before writing the submission CSV, as a last-mile guard
# against the Stage 3 disqualification threshold.
# ---------------------------------------------------------------------------

def check_honeypot_rate_in_top_n(
    ranked_candidate_ids: List[str],
    honeypot_results: pd.DataFrame,
    top_n: int = 100,
    disqualification_threshold: float = 0.10,
) -> dict:
    """
    Check the honeypot rate within the top N of a final ranking, against
    the Stage 3 disqualification threshold (>10% per submission_spec.docx
    Section 7).

    Parameters
    ----------
    ranked_candidate_ids : list of str
        The final ranked list of candidate_ids, rank 1 first. Only the
        first `top_n` are checked, matching how Stage 3 evaluates it.
    honeypot_results : pd.DataFrame
        Output of run_honeypot_detection(), used to look up each
        candidate's honeypot_flag.
    top_n : int
        How many top-ranked candidates to check (100, per the spec).
    disqualification_threshold : float
        The fraction above which the team is disqualified (0.10 = 10%).

    Returns
    -------
    dict with keys:
        honeypot_count          : int, how many honeypots are in the top N
        honeypot_rate           : float, honeypot_count / top_n
        disqualification_risk   : bool, True if honeypot_rate exceeds threshold
        flagged_candidate_ids   : list of str, which specific candidate_ids in
                                   the top N were flagged (for manual review
                                   before submitting — pull these and inspect
                                   them by hand before trusting the automated
                                   flag completely, since this is too high-
                                   stakes to skip a final human sanity check).
    """
    flag_lookup = honeypot_results.set_index("candidate_id")["honeypot_flag"].to_dict()

    top_slice = ranked_candidate_ids[:top_n]
    flagged = [cid for cid in top_slice if flag_lookup.get(cid, False)]

    rate = len(flagged) / top_n if top_n > 0 else 0.0

    return {
        "honeypot_count": len(flagged),
        "honeypot_rate": rate,
        "disqualification_risk": rate > disqualification_threshold,
        "flagged_candidate_ids": flagged,
    }


if __name__ == "__main__":
    import json as _json
    import time

    with open("sample_candidates.json") as f:
        sample = _json.load(f)

    start = time.perf_counter()
    result = run_honeypot_detection(sample)
    elapsed = time.perf_counter() - start

    print(f"Ran Layer 5 on {len(sample)} sample candidates in {elapsed:.4f}s\n")
    flagged = result[result["honeypot_flag"]]
    print(f"{len(flagged)} / {len(sample)} candidates flagged as honeypots in the 50-row sample.")
    if len(flagged):
        print(flagged[["candidate_id", "honeypot_evidence"]].to_string(index=False))
