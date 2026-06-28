"""
Tests for Layer 3 — Structural Disqualifier Pass
Run with: python3 test_disqualifiers.py

Uses a mix of:
  (a) Hand-crafted minimal candidate fixtures, to test each rule's logic
      in isolation with full control over the inputs, and
  (b) Real candidate_ids pulled from the actual sample_candidates.json /
      candidates.jsonl, to confirm the rules behave correctly on genuine
      data, not just synthetic edge cases.

No network access, no GPU — matches the same compute constraints the
final ranking step must respect.
"""

import json
import copy

from disqualifiers import (
    detect_title_mismatch,
    detect_consulting_only,
    detect_title_chaser,
    detect_stale_architect,
    detect_cv_speech_robotics_without_nlp,
    detect_recent_only_llm_wrapper,
    run_structural_disqualifiers,
    _classify_title,
)
from jd_requirements import get_jd_requirements


# ---------------------------------------------------------------------------
# Minimal candidate fixture builder — only the fields each rule actually
# reads are required to be realistic; everything else uses safe defaults.
# ---------------------------------------------------------------------------

def make_candidate(
    candidate_id="CAND_0000000",
    current_title="Software Engineer",
    current_company="Acme Corp",
    current_industry="Software",
    skills=None,
    career_history=None,
):
    if skills is None:
        skills = []
    if career_history is None:
        career_history = [{
            "company": current_company,
            "title": current_title,
            "start_date": "2023-01-01",
            "end_date": None,
            "duration_months": 24,
            "is_current": True,
            "industry": current_industry,
            "company_size": "201-500",
            "description": "Worked on backend systems.",
        }]
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Test headline",
            "summary": "Test summary",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": 6,
            "current_title": current_title,
            "current_company": current_company,
            "current_company_size": "201-500",
            "current_industry": current_industry,
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
# detect_title_mismatch tests
# ---------------------------------------------------------------------------

def test_title_mismatch_fires_on_keyword_stuffed_non_engineering_title():
    c = make_candidate(
        current_title="Marketing Manager",
        skills=[skill("RAG"), skill("Pinecone"), skill("LangChain"), skill("FAISS")],
    )
    assert detect_title_mismatch(c) is True
    print("[PASS] test_title_mismatch_fires_on_keyword_stuffed_non_engineering_title")


def test_title_mismatch_does_not_fire_on_legit_engineering_title():
    """A Backend Engineer with some ML-adjacent skills is NOT the trap
    pattern — this title is a legitimate engineering title in the dataset's
    closed vocabulary."""
    c = make_candidate(
        current_title="Backend Engineer",
        skills=[skill("NLP"), skill("LoRA"), skill("Milvus"), skill("Fine-tuning LLMs")],
    )
    assert detect_title_mismatch(c) is False
    print("[PASS] test_title_mismatch_does_not_fire_on_legit_engineering_title")


def test_title_mismatch_does_not_fire_on_non_engineering_title_with_few_ai_skills():
    """Only 2 core AI keywords (below the min_count=3 threshold) should not
    fire — avoids over-flagging on light/coincidental overlap."""
    c = make_candidate(
        current_title="HR Manager",
        skills=[skill("RAG"), skill("Python")],
    )
    assert detect_title_mismatch(c) is False
    print("[PASS] test_title_mismatch_does_not_fire_on_non_engineering_title_with_few_ai_skills")


def test_title_mismatch_on_real_dataset_examples():
    """Cross-check against real candidate_ids identified during development
    as genuine trap examples."""
    with open("sample_candidates.json") as f:
        sample = {c["candidate_id"]: c for c in json.load(f)}
    # CAND_0000021 is a Project Manager with 7 core AI keyword skills —
    # confirmed during development to be a genuine trap example.
    assert detect_title_mismatch(sample["CAND_0000021"]) is True
    print("[PASS] test_title_mismatch_on_real_dataset_examples")


# ---------------------------------------------------------------------------
# detect_consulting_only tests
# ---------------------------------------------------------------------------

def test_consulting_only_fires_when_entire_career_is_consulting():
    c = make_candidate(
        current_title="Software Engineer",
        current_company="TCS",
        current_industry="IT Services",
        career_history=[
            {"company": "TCS", "title": "Software Engineer", "start_date": "2022-01-01",
             "end_date": None, "duration_months": 30, "is_current": True,
             "industry": "IT Services", "company_size": "10001+", "description": "x"},
            {"company": "Infosys", "title": "Software Engineer", "start_date": "2019-01-01",
             "end_date": "2021-12-01", "duration_months": 35, "is_current": False,
             "industry": "IT Services", "company_size": "10001+", "description": "x"},
        ],
    )
    assert detect_consulting_only(c) is True
    print("[PASS] test_consulting_only_fires_when_entire_career_is_consulting")


def test_consulting_only_respects_jd_exception_for_prior_product_experience():
    """JD explicit exception: currently at a consulting firm but with
    PRIOR product-company experience elsewhere -> should NOT fire."""
    c = make_candidate(
        current_title="Software Engineer",
        current_company="TCS",
        current_industry="IT Services",
        career_history=[
            {"company": "TCS", "title": "Software Engineer", "start_date": "2022-01-01",
             "end_date": None, "duration_months": 30, "is_current": True,
             "industry": "IT Services", "company_size": "10001+", "description": "x"},
            {"company": "Flipkart", "title": "Software Engineer", "start_date": "2019-01-01",
             "end_date": "2021-12-01", "duration_months": 35, "is_current": False,
             "industry": "E-commerce", "company_size": "10001+", "description": "x"},
        ],
    )
    assert detect_consulting_only(c) is False
    print("[PASS] test_consulting_only_respects_jd_exception_for_prior_product_experience")


def test_consulting_only_uses_industry_tag_for_unlisted_firms():
    """HCL is not in the JD's explicitly named list but consistently tags
    as IT Services in the real dataset -> should still fire via the
    industry-tag signal."""
    c = make_candidate(
        current_title="Software Engineer",
        current_company="HCL",
        current_industry="IT Services",
        career_history=[
            {"company": "HCL", "title": "Software Engineer", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 60, "is_current": True,
             "industry": "IT Services", "company_size": "10001+", "description": "x"},
        ],
    )
    assert detect_consulting_only(c) is True
    print("[PASS] test_consulting_only_uses_industry_tag_for_unlisted_firms")


def test_consulting_only_with_layer1_firm_list():
    """Confirm the Layer 1 integration path (passing jd_requirements'
    consulting_firms set) behaves identically to the module default."""
    jd_reqs = get_jd_requirements()
    firms = {f.strip().lower() for f in jd_reqs["consulting_firms"]}
    c = make_candidate(
        current_title="Software Engineer", current_company="TCS", current_industry="IT Services",
        career_history=[
            {"company": "TCS", "title": "Software Engineer", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 60, "is_current": True,
             "industry": "IT Services", "company_size": "10001+", "description": "x"},
        ],
    )
    assert detect_consulting_only(c, named_consulting_firms=firms) is True
    print("[PASS] test_consulting_only_with_layer1_firm_list")


# ---------------------------------------------------------------------------
# detect_title_chaser tests
# ---------------------------------------------------------------------------

def test_title_chaser_fires_on_genuine_short_stint_escalation():
    c = make_candidate(career_history=[
        {"company": "A", "title": "Software Engineer", "start_date": "2018-01-01",
         "end_date": "2019-06-01", "duration_months": 17, "is_current": False,
         "industry": "Software", "company_size": "201-500", "description": "x"},
        {"company": "B", "title": "Senior Software Engineer", "start_date": "2019-07-01",
         "end_date": "2020-12-01", "duration_months": 17, "is_current": False,
         "industry": "Software", "company_size": "201-500", "description": "x"},
        {"company": "C", "title": "Staff Machine Learning Engineer", "start_date": "2021-01-01",
         "end_date": None, "duration_months": 30, "is_current": True,
         "industry": "Software", "company_size": "201-500", "description": "x"},
    ])
    assert detect_title_chaser(c) is True
    print("[PASS] test_title_chaser_fires_on_genuine_short_stint_escalation")


def test_title_chaser_does_not_fire_on_single_promotion_then_stable_tenure():
    """One promotion followed by a long, stable stint is normal career
    growth, not title-chasing."""
    c = make_candidate(career_history=[
        {"company": "A", "title": "Software Engineer", "start_date": "2015-01-01",
         "end_date": "2017-01-01", "duration_months": 24, "is_current": False,
         "industry": "Software", "company_size": "201-500", "description": "x"},
        {"company": "A", "title": "Senior Software Engineer", "start_date": "2017-01-01",
         "end_date": None, "duration_months": 60, "is_current": True,
         "industry": "Software", "company_size": "201-500", "description": "x"},
    ])
    assert detect_title_chaser(c) is False
    print("[PASS] test_title_chaser_does_not_fire_on_single_promotion_then_stable_tenure")


def test_title_chaser_does_not_fire_on_fewer_than_3_entries():
    c = make_candidate(career_history=[
        {"company": "A", "title": "Software Engineer", "start_date": "2018-01-01",
         "end_date": "2019-01-01", "duration_months": 12, "is_current": False,
         "industry": "Software", "company_size": "201-500", "description": "x"},
        {"company": "B", "title": "Staff Machine Learning Engineer", "start_date": "2019-01-01",
         "end_date": None, "duration_months": 12, "is_current": True,
         "industry": "Software", "company_size": "201-500", "description": "x"},
    ])
    assert detect_title_chaser(c) is False
    print("[PASS] test_title_chaser_does_not_fire_on_fewer_than_3_entries")


def test_title_chaser_returns_false_on_full_real_dataset():
    """Documented honest finding: this rule fires 0 times across the real
    100K-candidate dataset because no candidate's career trajectory matches
    the strict escalating + short-stint pattern (verified during
    development — see disqualifiers.py module docstring for detail). This
    test locks in that finding for the 50-row sample as a regression check."""
    with open("sample_candidates.json") as f:
        sample = json.load(f)
    fires = sum(1 for c in sample if detect_title_chaser(c))
    assert fires == 0
    print("[PASS] test_title_chaser_returns_false_on_full_real_dataset (sample)")


# ---------------------------------------------------------------------------
# detect_stale_architect tests
# ---------------------------------------------------------------------------

def test_stale_architect_fires_on_long_tenure_lead_title_with_no_hands_on_skill():
    c = make_candidate(
        current_title="Lead AI Engineer",
        skills=[skill("PowerPoint"), skill("Agile")],  # no hands-on coding signal
        career_history=[{
            "company": "Acme", "title": "Lead AI Engineer", "start_date": "2022-01-01",
            "end_date": None, "duration_months": 30, "is_current": True,
            "industry": "Software", "company_size": "201-500", "description": "x",
        }],
    )
    assert detect_stale_architect(c) is True
    print("[PASS] test_stale_architect_fires_on_long_tenure_lead_title_with_no_hands_on_skill")


def test_stale_architect_does_not_fire_with_recent_hands_on_skill():
    """If a hands-on skill's duration_months covers the full current
    tenure, the candidate is still actively coding -> should not fire."""
    c = make_candidate(
        current_title="Staff Machine Learning Engineer",
        skills=[skill("Python", duration_months=40), skill("PyTorch", duration_months=36)],
        career_history=[{
            "company": "Acme", "title": "Staff Machine Learning Engineer", "start_date": "2022-01-01",
            "end_date": None, "duration_months": 30, "is_current": True,
            "industry": "Software", "company_size": "201-500", "description": "x",
        }],
    )
    assert detect_stale_architect(c) is False
    print("[PASS] test_stale_architect_does_not_fire_with_recent_hands_on_skill")


def test_stale_architect_does_not_fire_on_generic_manager_titles():
    """REGRESSION TEST for a real bug caught during development: an
    earlier version of this rule matched ANY title containing the
    substring 'manager', incorrectly firing on Operations/HR/Marketing/
    Project Managers who were never engineers at all. This test locks in
    the fix."""
    for title in ["Operations Manager", "HR Manager", "Marketing Manager", "Project Manager"]:
        c = make_candidate(
            current_title=title,
            skills=[],
            career_history=[{
                "company": "Acme", "title": title, "start_date": "2020-01-01",
                "end_date": None, "duration_months": 40, "is_current": True,
                "industry": "Software", "company_size": "201-500", "description": "x",
            }],
        )
        assert detect_stale_architect(c) is False, f"Should not fire on {title}"
    print("[PASS] test_stale_architect_does_not_fire_on_generic_manager_titles")


def test_stale_architect_does_not_fire_on_short_tenure():
    c = make_candidate(
        current_title="Lead AI Engineer",
        skills=[],
        career_history=[{
            "company": "Acme", "title": "Lead AI Engineer", "start_date": "2025-06-01",
            "end_date": None, "duration_months": 6, "is_current": True,
            "industry": "Software", "company_size": "201-500", "description": "x",
        }],
    )
    assert detect_stale_architect(c) is False
    print("[PASS] test_stale_architect_does_not_fire_on_short_tenure")


# ---------------------------------------------------------------------------
# detect_cv_speech_robotics_without_nlp tests
# ---------------------------------------------------------------------------

def test_cv_speech_robotics_fires_on_cv_title_with_zero_nlp_signal_anywhere():
    c = make_candidate(
        current_title="Computer Vision Engineer",
        skills=[skill("OpenCV"), skill("YOLO"), skill("CNN")],
        career_history=[{
            "company": "Acme", "title": "Computer Vision Engineer", "start_date": "2020-01-01",
            "end_date": None, "duration_months": 40, "is_current": True,
            "industry": "Software", "company_size": "201-500",
            "description": "Built object detection models for autonomous vehicle perception.",
        }],
    )
    assert detect_cv_speech_robotics_without_nlp(c) is True
    print("[PASS] test_cv_speech_robotics_fires_on_cv_title_with_zero_nlp_signal_anywhere")


def test_cv_speech_robotics_does_not_fire_if_description_mentions_nlp_adjacent_work():
    c = make_candidate(
        current_title="Computer Vision Engineer",
        skills=[skill("OpenCV"), skill("YOLO")],
        career_history=[{
            "company": "Acme", "title": "Computer Vision Engineer", "start_date": "2020-01-01",
            "end_date": None, "duration_months": 40, "is_current": True,
            "industry": "Software", "company_size": "201-500",
            "description": "Built object detection models and also worked on semantic search for product retrieval.",
        }],
    )
    assert detect_cv_speech_robotics_without_nlp(c) is False
    print("[PASS] test_cv_speech_robotics_does_not_fire_if_description_mentions_nlp_adjacent_work")


def test_cv_speech_robotics_does_not_fire_on_non_cv_title():
    """A Cloud Engineer with a couple of stray CV-flavored skills (dataset
    noise) should NOT fire — this was a real false-positive bug caught
    during development (see disqualifiers.py docstring)."""
    c = make_candidate(
        current_title="Cloud Engineer",
        skills=[skill("YOLO"), skill("Diffusion Models")],
    )
    assert detect_cv_speech_robotics_without_nlp(c) is False
    print("[PASS] test_cv_speech_robotics_does_not_fire_on_non_cv_title")


def test_cv_speech_robotics_returns_false_on_all_real_cv_titled_candidates():
    """Documented honest finding from full-dataset validation: every one
    of the 132 'Computer Vision Engineer'-titled candidates in the full
    100K dataset has at least some NLP/IR signal somewhere (skills or
    description), so this rule correctly returns False for all of them as
    currently implemented. Spot-checked here against the sample."""
    with open("sample_candidates.json") as f:
        sample = json.load(f)
    cv_titled = [c for c in sample if c["profile"]["current_title"] == "Computer Vision Engineer"]
    for c in cv_titled:
        assert detect_cv_speech_robotics_without_nlp(c) is False
    print(f"[PASS] test_cv_speech_robotics_returns_false_on_all_real_cv_titled_candidates "
          f"({len(cv_titled)} found in sample)")


# ---------------------------------------------------------------------------
# detect_recent_only_llm_wrapper tests
# ---------------------------------------------------------------------------

def test_recent_only_llm_wrapper_fires_on_genuine_pattern():
    c = make_candidate(
        skills=[skill("LangChain", duration_months=6), skill("Prompt Engineering", duration_months=5)],
        career_history=[{
            "company": "Acme", "title": "AI Engineer", "start_date": "2025-10-01",
            "end_date": None, "duration_months": 8, "is_current": True,
            "industry": "Software", "company_size": "201-500", "description": "x",
        }],
    )
    assert detect_recent_only_llm_wrapper(c) is True
    print("[PASS] test_recent_only_llm_wrapper_fires_on_genuine_pattern")


def test_recent_only_llm_wrapper_does_not_fire_with_prior_ml_experience():
    """Someone with 6 years of classical ML experience who only recently
    added LangChain should NOT be flagged — this is the JD's own example
    of someone who is NOT a wrapper-only candidate."""
    c = make_candidate(
        skills=[
            skill("LangChain", duration_months=6),
            skill("Machine Learning", duration_months=72),
            skill("Python", duration_months=80),
        ],
        career_history=[{
            "company": "Acme", "title": "AI Engineer", "start_date": "2025-10-01",
            "end_date": None, "duration_months": 8, "is_current": True,
            "industry": "Software", "company_size": "201-500", "description": "x",
        }],
    )
    assert detect_recent_only_llm_wrapper(c) is False
    print("[PASS] test_recent_only_llm_wrapper_does_not_fire_with_prior_ml_experience")


def test_recent_only_llm_wrapper_does_not_fire_on_long_tenure():
    c = make_candidate(
        skills=[skill("LangChain", duration_months=6)],
        career_history=[{
            "company": "Acme", "title": "AI Engineer", "start_date": "2020-01-01",
            "end_date": None, "duration_months": 36, "is_current": True,
            "industry": "Software", "company_size": "201-500", "description": "x",
        }],
    )
    assert detect_recent_only_llm_wrapper(c) is False
    print("[PASS] test_recent_only_llm_wrapper_does_not_fire_on_long_tenure")


# ---------------------------------------------------------------------------
# run_structural_disqualifiers (orchestration) tests
# ---------------------------------------------------------------------------

def test_orchestration_returns_expected_columns():
    candidates = [make_candidate(candidate_id=f"CAND_000000{i}") for i in range(3)]
    result = run_structural_disqualifiers(candidates)
    expected_cols = {
        "candidate_id", "title_is_engineering", "title_mismatch_flag",
        "consulting_only_flag", "title_chaser_flag", "stale_architect_flag",
        "cv_speech_robotics_flag", "recent_only_llm_wrapper_flag",
        "disqualifier_vector", "disqualifier_penalty",
    }
    assert expected_cols.issubset(set(result.columns)), f"Missing columns: {expected_cols - set(result.columns)}"
    assert len(result) == 3
    print("[PASS] test_orchestration_returns_expected_columns")


def test_orchestration_penalty_caps_at_one():
    """A candidate tripping every single rule should still cap at penalty
    1.0, not sum unboundedly past it."""
    c = make_candidate(
        current_title="Marketing Manager",
        current_company="TCS",
        current_industry="IT Services",
        skills=[skill("RAG"), skill("Pinecone"), skill("LangChain"), skill("FAISS")],
        career_history=[{
            "company": "TCS", "title": "Marketing Manager", "start_date": "2020-01-01",
            "end_date": None, "duration_months": 60, "is_current": True,
            "industry": "IT Services", "company_size": "10001+", "description": "x",
        }],
    )
    result = run_structural_disqualifiers([c])
    assert result.iloc[0]["disqualifier_penalty"] <= 1.0
    print("[PASS] test_orchestration_penalty_caps_at_one")


def test_orchestration_clean_candidate_has_zero_penalty():
    c = make_candidate()
    result = run_structural_disqualifiers([c])
    assert result.iloc[0]["disqualifier_penalty"] == 0.0
    assert result.iloc[0]["disqualifier_vector"] == ""
    print("[PASS] test_orchestration_clean_candidate_has_zero_penalty")


def test_orchestration_with_layer1_integration():
    jd_reqs = get_jd_requirements()
    candidates = [make_candidate(candidate_id=f"CAND_000000{i}") for i in range(3)]
    result = run_structural_disqualifiers(candidates, jd_requirements=jd_reqs)
    assert len(result) == 3
    print("[PASS] test_orchestration_with_layer1_integration")


def test_orchestration_runs_on_real_sample_without_error():
    with open("sample_candidates.json") as f:
        sample = json.load(f)
    result = run_structural_disqualifiers(sample)
    assert len(result) == len(sample)
    assert result["candidate_id"].is_unique
    print(f"[PASS] test_orchestration_runs_on_real_sample_without_error ({len(sample)} candidates)")


# ---------------------------------------------------------------------------
# Title classifier tests
# ---------------------------------------------------------------------------

def test_classify_title_known_engineering_title():
    assert _classify_title("Senior Machine Learning Engineer") is True
    print("[PASS] test_classify_title_known_engineering_title")


def test_classify_title_known_non_engineering_title():
    assert _classify_title("Marketing Manager") is False
    print("[PASS] test_classify_title_known_non_engineering_title")


def test_classify_title_fuzzy_fallback_for_unseen_variant():
    """A title not in either closed vocabulary should fall back to fuzzy
    matching / keyword heuristics rather than crashing."""
    result = _classify_title("Sr. SDE - Machine Learning")
    assert isinstance(result, bool)
    print(f"[PASS] test_classify_title_fuzzy_fallback_for_unseen_variant (returned {result})")


if __name__ == "__main__":
    tests = [
        test_title_mismatch_fires_on_keyword_stuffed_non_engineering_title,
        test_title_mismatch_does_not_fire_on_legit_engineering_title,
        test_title_mismatch_does_not_fire_on_non_engineering_title_with_few_ai_skills,
        test_title_mismatch_on_real_dataset_examples,
        test_consulting_only_fires_when_entire_career_is_consulting,
        test_consulting_only_respects_jd_exception_for_prior_product_experience,
        test_consulting_only_uses_industry_tag_for_unlisted_firms,
        test_consulting_only_with_layer1_firm_list,
        test_title_chaser_fires_on_genuine_short_stint_escalation,
        test_title_chaser_does_not_fire_on_single_promotion_then_stable_tenure,
        test_title_chaser_does_not_fire_on_fewer_than_3_entries,
        test_title_chaser_returns_false_on_full_real_dataset,
        test_stale_architect_fires_on_long_tenure_lead_title_with_no_hands_on_skill,
        test_stale_architect_does_not_fire_with_recent_hands_on_skill,
        test_stale_architect_does_not_fire_on_generic_manager_titles,
        test_stale_architect_does_not_fire_on_short_tenure,
        test_cv_speech_robotics_fires_on_cv_title_with_zero_nlp_signal_anywhere,
        test_cv_speech_robotics_does_not_fire_if_description_mentions_nlp_adjacent_work,
        test_cv_speech_robotics_does_not_fire_on_non_cv_title,
        test_cv_speech_robotics_returns_false_on_all_real_cv_titled_candidates,
        test_recent_only_llm_wrapper_fires_on_genuine_pattern,
        test_recent_only_llm_wrapper_does_not_fire_with_prior_ml_experience,
        test_recent_only_llm_wrapper_does_not_fire_on_long_tenure,
        test_orchestration_returns_expected_columns,
        test_orchestration_penalty_caps_at_one,
        test_orchestration_clean_candidate_has_zero_penalty,
        test_orchestration_with_layer1_integration,
        test_orchestration_runs_on_real_sample_without_error,
        test_classify_title_known_engineering_title,
        test_classify_title_known_non_engineering_title,
        test_classify_title_fuzzy_fallback_for_unseen_variant,
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
