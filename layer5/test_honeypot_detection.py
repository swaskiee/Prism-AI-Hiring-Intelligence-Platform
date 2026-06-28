"""
Tests for Layer 5 — Honeypot & Anomaly Detection
Run with: python3 test_honeypot_detection.py

Given this is the highest-stakes layer (>10% honeypot rate in the
submitted top-100 disqualifies the entire team at Stage 3), this test
suite is deliberately more thorough than Layer 3's, including an explicit
precision/recall-style cross-check against the full real dataset.
"""

import json
from datetime import date

from honeypot_detection import (
    detect_duration_date_mismatch,
    detect_expert_zero_duration_skills,
    run_honeypot_detection,
    check_honeypot_rate_in_top_n,
    DEFAULT_REFERENCE_DATE,
)


def make_candidate(
    candidate_id="CAND_0000000",
    career_history=None,
    skills=None,
):
    if career_history is None:
        # 2024-06-01 -> 2026-06-28 is exactly 25 months, matching
        # duration_months=25 below — internally consistent against the
        # module's DEFAULT_REFERENCE_DATE (2026-06-28), so this fixture is
        # genuinely "clean" by default rather than accidentally tripping
        # the duration-mismatch detector.
        career_history = [{
            "company": "Acme", "title": "Software Engineer", "start_date": "2024-06-01",
            "end_date": None, "duration_months": 25, "is_current": True,
            "industry": "Software", "company_size": "201-500", "description": "x",
        }]
    if skills is None:
        skills = []
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test", "headline": "x", "summary": "x",
            "location": "Bangalore", "country": "India", "years_of_experience": 6,
            "current_title": "Software Engineer", "current_company": "Acme",
            "current_company_size": "201-500", "current_industry": "Software",
        },
        "career_history": career_history,
        "education": [],
        "skills": skills,
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 80, "signup_date": "2025-01-01",
            "last_active_date": "2026-06-01", "open_to_work_flag": True,
            "profile_views_received_30d": 10, "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.5, "avg_response_time_hours": 24,
            "skill_assessment_scores": {}, "connection_count": 100,
            "endorsements_received": 10, "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 30},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 50, "search_appearance_30d": 10,
            "saved_by_recruiters_30d": 1, "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.5, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }


def skill(name, proficiency="advanced", endorsements=10, duration_months=24):
    return {"name": name, "proficiency": proficiency, "endorsements": endorsements,
            "duration_months": duration_months}


# ---------------------------------------------------------------------------
# detect_duration_date_mismatch tests
# ---------------------------------------------------------------------------

def test_duration_mismatch_fires_on_overstated_tenure():
    """The JD's named example: claims far more tenure than the start_date
    allows."""
    c = make_candidate(career_history=[{
        "company": "Wayne Enterprises", "title": "Frontend Engineer",
        "start_date": "2023-09-10", "end_date": None, "duration_months": 166,
        "is_current": True, "industry": "Software", "company_size": "201-500",
        "description": "x",
    }])
    evidence = detect_duration_date_mismatch(c, reference_date=date(2026, 6, 28))
    assert len(evidence) == 1
    assert "OVERSTATES" in evidence[0]
    print("[PASS] test_duration_mismatch_fires_on_overstated_tenure")


def test_duration_mismatch_fires_on_understated_tenure():
    c = make_candidate(career_history=[{
        "company": "Hooli", "title": "Business Analyst",
        "start_date": "2023-09-10", "end_date": None, "duration_months": 11,
        "is_current": True, "industry": "Software", "company_size": "201-500",
        "description": "x",
    }])
    evidence = detect_duration_date_mismatch(c, reference_date=date(2026, 6, 28))
    assert len(evidence) == 1
    assert "UNDERSTATES" in evidence[0]
    print("[PASS] test_duration_mismatch_fires_on_understated_tenure")


def test_duration_mismatch_does_not_fire_within_tolerance():
    """A small rounding gap (e.g. day-of-month differences) should not
    trigger a false positive."""
    c = make_candidate(career_history=[{
        "company": "Acme", "title": "Software Engineer",
        "start_date": "2024-01-15", "end_date": None, "duration_months": 29,
        "is_current": True, "industry": "Software", "company_size": "201-500",
        "description": "x",
    }])
    # ~29 months actual vs 29 stated -> no mismatch
    evidence = detect_duration_date_mismatch(c, reference_date=date(2026, 6, 28))
    assert evidence == []
    print("[PASS] test_duration_mismatch_does_not_fire_within_tolerance")


def test_duration_mismatch_ignores_past_roles():
    """Only is_current=True entries are checked — a past role with a
    deliberately mismatched duration (for test purposes) should NOT fire,
    since the dataset confirmed this pattern only ever appears on current
    roles, and past-role start/end date math was separately confirmed
    consistent."""
    c = make_candidate(career_history=[
        {"company": "Old Co", "title": "Engineer", "start_date": "2015-01-01",
         "end_date": "2016-01-01", "duration_months": 200, "is_current": False,
         "industry": "Software", "company_size": "201-500", "description": "x"},
        {"company": "Acme", "title": "Software Engineer", "start_date": "2024-06-01",
         "end_date": None, "duration_months": 25, "is_current": True,
         "industry": "Software", "company_size": "201-500", "description": "x"},
    ])
    evidence = detect_duration_date_mismatch(c, reference_date=date(2026, 6, 28))
    assert evidence == []  # the mismatched entry is_current=False, so ignored by design
    print("[PASS] test_duration_mismatch_ignores_past_roles")


def test_duration_mismatch_on_real_dataset_example():
    with open("sample_candidates.json") as f:
        sample = json.load(f)
    # Confirm a clean candidate in the sample shows no mismatch (regression
    # guard — sample has no honeypots, confirmed during development)
    for c in sample:
        evidence = detect_duration_date_mismatch(c)
        assert evidence == [], f"Unexpected mismatch flagged on {c['candidate_id']}: {evidence}"
    print("[PASS] test_duration_mismatch_on_real_dataset_example (0 false positives on 50-row sample)")


# ---------------------------------------------------------------------------
# detect_expert_zero_duration_skills tests
# ---------------------------------------------------------------------------

def test_expert_zero_duration_fires_on_single_occurrence():
    c = make_candidate(skills=[skill("Docker", proficiency="expert", duration_months=0, endorsements=1)])
    evidence = detect_expert_zero_duration_skills(c)
    assert len(evidence) == 1
    assert "Docker" in evidence[0]
    print("[PASS] test_expert_zero_duration_fires_on_single_occurrence")


def test_expert_zero_duration_fires_on_multiple_occurrences():
    c = make_candidate(skills=[
        skill("Docker", proficiency="expert", duration_months=0),
        skill("Go", proficiency="expert", duration_months=0),
        skill("Photoshop", proficiency="expert", duration_months=0),
    ])
    evidence = detect_expert_zero_duration_skills(c)
    assert len(evidence) == 3
    print("[PASS] test_expert_zero_duration_fires_on_multiple_occurrences")


def test_expert_zero_duration_does_not_fire_on_advanced_proficiency():
    """Only 'expert' proficiency counts — 'advanced' with 0 duration is a
    different (much weaker) signal and not part of the named honeypot
    pattern."""
    c = make_candidate(skills=[skill("Docker", proficiency="advanced", duration_months=0)])
    evidence = detect_expert_zero_duration_skills(c)
    assert evidence == []
    print("[PASS] test_expert_zero_duration_does_not_fire_on_advanced_proficiency")


def test_expert_zero_duration_does_not_fire_on_nonzero_duration():
    c = make_candidate(skills=[skill("Docker", proficiency="expert", duration_months=3)])
    evidence = detect_expert_zero_duration_skills(c)
    assert evidence == []
    print("[PASS] test_expert_zero_duration_does_not_fire_on_nonzero_duration")


def test_expert_zero_duration_on_real_dataset_sample():
    with open("sample_candidates.json") as f:
        sample = json.load(f)
    for c in sample:
        evidence = detect_expert_zero_duration_skills(c)
        assert evidence == [], f"Unexpected flag on {c['candidate_id']}: {evidence}"
    print("[PASS] test_expert_zero_duration_on_real_dataset_sample (0 false positives on 50-row sample)")


# ---------------------------------------------------------------------------
# run_honeypot_detection (orchestration) tests
# ---------------------------------------------------------------------------

def test_orchestration_returns_expected_columns():
    candidates = [make_candidate(candidate_id=f"CAND_000000{i}") for i in range(3)]
    result = run_honeypot_detection(candidates)
    expected_cols = {
        "candidate_id", "honeypot_duration_mismatch_flag",
        "honeypot_expert_zero_duration_flag", "honeypot_flag", "honeypot_evidence",
    }
    assert expected_cols.issubset(set(result.columns))
    assert len(result) == 3
    print("[PASS] test_orchestration_returns_expected_columns")


def test_orchestration_clean_candidate_not_flagged():
    c = make_candidate()
    result = run_honeypot_detection([c])
    assert result.iloc[0]["honeypot_flag"] == False
    assert result.iloc[0]["honeypot_evidence"] == ""
    print("[PASS] test_orchestration_clean_candidate_not_flagged")


def test_orchestration_combines_both_rules():
    """A candidate tripping BOTH rules simultaneously should have
    honeypot_flag=True with evidence from both, combined with ' | '."""
    c = make_candidate(
        career_history=[{
            "company": "Wayne Enterprises", "title": "Frontend Engineer",
            "start_date": "2023-09-10", "end_date": None, "duration_months": 166,
            "is_current": True, "industry": "Software", "company_size": "201-500",
            "description": "x",
        }],
        skills=[skill("Docker", proficiency="expert", duration_months=0)],
    )
    result = run_honeypot_detection([c])
    row = result.iloc[0]
    assert row["honeypot_flag"] == True
    assert row["honeypot_duration_mismatch_flag"] == True
    assert row["honeypot_expert_zero_duration_flag"] == True
    assert "OVERSTATES" in row["honeypot_evidence"]
    assert "Docker" in row["honeypot_evidence"]
    print("[PASS] test_orchestration_combines_both_rules")


def test_orchestration_exact_match_against_full_dataset(candidates_path="candidates.jsonl"):
    """CRITICAL regression test: re-run against the full 100K dataset and
    confirm exactly 54 candidates are flagged, matching the count manually
    verified during development. If this count changes unexpectedly after
    a future edit, it's a signal something broke.

    Looks for candidates.jsonl in the current directory by default (pass a
    different path as needed). The full 487MB file is NOT included in this
    repo — download it from the official hackathon dataset link and place
    it alongside this test file (or pass its path) to run this specific
    check. All other tests in this suite are self-contained and don't
    require it.
    """
    import os
    if not os.path.exists(candidates_path):
        print(f"[SKIP] test_orchestration_exact_match_against_full_dataset "
              f"(candidates.jsonl not found at '{candidates_path}' — download "
              f"the full dataset to run this specific regression check)")
        return

    candidates = []
    with open(candidates_path) as f:
        for line in f:
            candidates.append(json.loads(line))
    result = run_honeypot_detection(candidates)
    total_flagged = result["honeypot_flag"].sum()
    duration_flagged = result["honeypot_duration_mismatch_flag"].sum()
    expert_flagged = result["honeypot_expert_zero_duration_flag"].sum()
    assert total_flagged == 54, f"Expected 54 total honeypots, got {total_flagged}"
    assert duration_flagged == 33, f"Expected 33 duration-mismatch honeypots, got {duration_flagged}"
    assert expert_flagged == 21, f"Expected 21 expert-zero-duration honeypots, got {expert_flagged}"
    print(f"[PASS] test_orchestration_exact_match_against_full_dataset "
          f"(54 total: 33 duration + 21 expert-zero, 0 overlap)")


# ---------------------------------------------------------------------------
# check_honeypot_rate_in_top_n tests
# ---------------------------------------------------------------------------

def test_check_rate_under_threshold_is_safe():
    candidates = [make_candidate(candidate_id=f"CAND_000000{i}") for i in range(10)]
    honeypot_results = run_honeypot_detection(candidates)
    ranked_ids = [c["candidate_id"] for c in candidates]
    result = check_honeypot_rate_in_top_n(ranked_ids, honeypot_results, top_n=10)
    assert result["honeypot_count"] == 0
    assert result["disqualification_risk"] == False
    print("[PASS] test_check_rate_under_threshold_is_safe")


def test_check_rate_over_threshold_flags_risk():
    """11 honeypots in a top-100 (11%) should trip disqualification_risk
    since it's over the 10% threshold."""
    clean = [make_candidate(candidate_id=f"CAND_CLEAN_{i:03d}") for i in range(89)]
    honeypots = [
        make_candidate(
            candidate_id=f"CAND_HP_{i:03d}",
            skills=[skill("Docker", proficiency="expert", duration_months=0)],
        )
        for i in range(11)
    ]
    all_candidates = clean + honeypots
    honeypot_results = run_honeypot_detection(all_candidates)
    ranked_ids = [c["candidate_id"] for c in all_candidates]  # order doesn't matter for this check
    result = check_honeypot_rate_in_top_n(ranked_ids, honeypot_results, top_n=100)
    assert result["honeypot_count"] == 11
    assert abs(result["honeypot_rate"] - 0.11) < 1e-9
    assert result["disqualification_risk"] == True
    assert len(result["flagged_candidate_ids"]) == 11
    print("[PASS] test_check_rate_over_threshold_flags_risk")


def test_check_rate_only_considers_top_n():
    """A honeypot ranked at position 150 should NOT count against a
    top_n=100 check."""
    clean = [make_candidate(candidate_id=f"CAND_CLEAN_{i:03d}") for i in range(100)]
    honeypot = make_candidate(
        candidate_id="CAND_HP_999",
        skills=[skill("Docker", proficiency="expert", duration_months=0)],
    )
    all_candidates = clean + [honeypot]
    honeypot_results = run_honeypot_detection(all_candidates)
    ranked_ids = [c["candidate_id"] for c in clean] + [honeypot["candidate_id"]]  # honeypot at rank 101
    result = check_honeypot_rate_in_top_n(ranked_ids, honeypot_results, top_n=100)
    assert result["honeypot_count"] == 0
    assert result["disqualification_risk"] == False
    print("[PASS] test_check_rate_only_considers_top_n")


if __name__ == "__main__":
    tests = [
        test_duration_mismatch_fires_on_overstated_tenure,
        test_duration_mismatch_fires_on_understated_tenure,
        test_duration_mismatch_does_not_fire_within_tolerance,
        test_duration_mismatch_ignores_past_roles,
        test_duration_mismatch_on_real_dataset_example,
        test_expert_zero_duration_fires_on_single_occurrence,
        test_expert_zero_duration_fires_on_multiple_occurrences,
        test_expert_zero_duration_does_not_fire_on_advanced_proficiency,
        test_expert_zero_duration_does_not_fire_on_nonzero_duration,
        test_expert_zero_duration_on_real_dataset_sample,
        test_orchestration_returns_expected_columns,
        test_orchestration_clean_candidate_not_flagged,
        test_orchestration_combines_both_rules,
        test_orchestration_exact_match_against_full_dataset,
        test_check_rate_under_threshold_is_safe,
        test_check_rate_over_threshold_flags_risk,
        test_check_rate_only_considers_top_n,
    ]

    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    if failures:
        raise SystemExit(1)
