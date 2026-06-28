"""
Layer 1 — JD Requirement Extraction
Team Hyperion | India Runs Hackathon | Track 1: Data & AI Challenge
Owner: Swati Dubey

PURPOSE
-------
Convert the unstructured Senior AI Engineer (Founding Team) job description
from Redrob AI into a single structured, machine-usable requirement object.

This module is intentionally NOT a prompt and NOT a call to any LLM. The JD
text is static for the duration of the hackathon, so this is a deterministic
parsing/config module: the same input always produces the same output, with
no network calls and near-zero runtime cost (microseconds, not relevant to
the 5-minute compute budget shared by the other layers).

CONTRACT
--------
get_jd_requirements() returns a dict with this exact top-level shape, agreed
with Nitanshu (team lead, owner of Layers 2/4) as the stable interface that
Layer 2 (semantic scorer) and Layer 3 (structural disqualifier pass, also
owned by Swati) both consume:

{
    "must_have": [str, ...],
    "nice_to_have": [str, ...],
    "hard_disqualifiers": [str, ...],          # short machine-readable codes
    "hard_disqualifiers_detail": {code: str},  # human-readable description per code
    "experience_band_years": [int, int],
    "location_preference": [str, ...],
    "ideal_profile_notes": str,
    "consulting_firms": [str, ...],            # used by Layer 3's consulting-only detector
}

Do not rename top-level keys without flagging Nitanshu first — Layer 2 and
Layer 3 both key off these names directly.

VALIDATION
----------
Run this file directly to execute self-checks:
    python3 jd_requirements.py
This confirms the contract shape is intact (correct keys, correct types,
non-empty lists) before it's handed off or imported elsewhere. It does not
require any candidate dataset — Layer 1 has no dependency on candidate data.
"""

from __future__ import annotations
from typing import Dict, List, Union
import json


# ---------------------------------------------------------------------------
# 1. Source JD text (verbatim, for traceability back to Redrob's own wording)
# ---------------------------------------------------------------------------
# Kept here as a single source of truth so that if the JD is ever updated,
# only this string needs to change before re-running extraction below.

JD_ROLE_TITLE = "Senior AI Engineer — Founding Team"
JD_COMPANY = "Redrob AI"
JD_LOCATION = "Pune/Noida (hybrid, flexible)"
JD_EXPERIENCE_BAND_RAW = "5-9 years (soft band, not a hard cutoff)"

JD_REDROB_TRAP_QUOTE = (
    "The 'right answer' to this JD is not 'find candidates whose skills "
    "section contains the most AI keywords.' That's a trap we've explicitly "
    "built into the dataset. The right answer involves reasoning about the "
    "gap between what the JD says and what the JD means. A Tier 5 candidate "
    "may not use the words 'RAG' or 'Pinecone' in their profile, but if "
    "their career history shows they built a recommendation system at a "
    "product company, they're a fit. A candidate who has all the AI "
    "keywords listed as skills but whose title is 'Marketing Manager' is "
    "not a fit, no matter how perfect their skill list looks."
)


# ---------------------------------------------------------------------------
# 2. Structured requirement extraction
# ---------------------------------------------------------------------------

def get_jd_requirements() -> Dict[str, Union[List[str], Dict[str, str], List[int], str]]:
    """
    Return the structured requirement object for the Senior AI Engineer JD.

    Returns
    -------
    dict
        Stable-shaped contract described in the module docstring above.
    """

    must_have: List[str] = [
        "production_embeddings_retrieval",
        "production_vector_db_or_hybrid_search",
        "strong_python_demonstrated_in_systems",
        "ranking_evaluation_framework_experience",
    ]

    must_have_detail: Dict[str, str] = {
        "production_embeddings_retrieval": (
            "Production experience with embeddings-based retrieval systems "
            "(sentence-transformers, OpenAI embeddings, BGE, E5, or similar) "
            "deployed to real users — specifically experience handling "
            "embedding drift, index refresh, and retrieval-quality "
            "regression in production."
        ),
        "production_vector_db_or_hybrid_search": (
            "Production experience with vector databases or hybrid search "
            "infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, "
            "Elasticsearch, FAISS, or similar)."
        ),
        "strong_python_demonstrated_in_systems": (
            "Strong Python, with real code-quality signal demonstrated "
            "through actual systems built — not just 'knows Python' listed "
            "as a skill."
        ),
        "ranking_evaluation_framework_experience": (
            "Hands-on experience designing evaluation frameworks for "
            "ranking systems — NDCG, MRR, MAP, offline-to-online "
            "correlation, A/B test interpretation."
        ),
    }

    nice_to_have: List[str] = [
        "llm_fine_tuning_lora_qlora_peft",
        "learning_to_rank_xgboost_or_neural",
        "hr_tech_recruiting_marketplace_background",
        "distributed_systems_or_inference_optimization",
        "open_source_ai_ml_contributions",
    ]

    nice_to_have_detail: Dict[str, str] = {
        "llm_fine_tuning_lora_qlora_peft": "LLM fine-tuning experience (LoRA, QLoRA, PEFT).",
        "learning_to_rank_xgboost_or_neural": "Learning-to-rank models (XGBoost-based or neural).",
        "hr_tech_recruiting_marketplace_background": "Prior HR-tech / recruiting tech / marketplace product experience.",
        "distributed_systems_or_inference_optimization": "Distributed systems or large-scale inference optimization background.",
        "open_source_ai_ml_contributions": "Open-source contributions in AI/ML.",
    }

    hard_disqualifiers: List[str] = [
        "pure_research_no_production",
        "recent_only_llm_wrapper_experience",
        "stale_architect_18mo_no_code",
        "title_chaser_pattern",
        "framework_tourist_pattern",
        "pure_consulting_career_no_product_co",
        "cv_speech_robotics_without_nlp",
        "closed_source_only_no_external_validation",
    ]

    hard_disqualifiers_detail: Dict[str, str] = {
        "pure_research_no_production": (
            "Career spent entirely in academic labs / research-only roles "
            "with zero production deployment."
        ),
        "recent_only_llm_wrapper_experience": (
            "'AI experience' is primarily under 12 months of "
            "LangChain-calls-OpenAI style work, with no pre-LLM-era "
            "production ML experience."
        ),
        "stale_architect_18mo_no_code": (
            "Senior engineer who has not written production code in 18+ "
            "months due to moving into a pure architecture/tech-lead role. "
            "This role writes code."
        ),
        "title_chaser_pattern": (
            "Career trajectory shows switching companies roughly every 1.5 "
            "years purely to climb Senior to Staff to Principal-equivalent "
            "titles. Redrob wants 3+ year commitment signal."
        ),
        "framework_tourist_pattern": (
            "GitHub/portfolio consists of 'how I built X with hot "
            "framework Y' tutorial-style projects with no systems thinking "
            "evident."
        ),
        "pure_consulting_career_no_product_co": (
            "Entire career at a consulting/services firm (e.g. TCS, "
            "Infosys, Wipro, Accenture, Cognizant, Capgemini) with no "
            "product-company experience anywhere in history. EXCEPTION: "
            "currently at one of these firms but with prior product-company "
            "experience elsewhere in their history is NOT disqualified."
        ),
        "cv_speech_robotics_without_nlp": (
            "Specialist in computer vision, speech, or robotics with no "
            "significant NLP/IR exposure anywhere in their career — would "
            "be re-learning fundamentals for this role."
        ),
        "closed_source_only_no_external_validation": (
            "5+ years entirely on closed-source proprietary systems with "
            "zero papers, talks, or open-source contributions. Redrob wants "
            "to see evidence of how the person thinks, not just trust that "
            "they can."
        ),
    }

    experience_band_years: List[int] = [5, 9]

    location_preference: List[str] = ["Pune", "Noida", "Hyderabad", "Mumbai", "Delhi NCR"]

    ideal_profile_notes: str = (
        "6-8 years total experience, 4-5 of which are in applied ML/AI at "
        "product companies (not pure services). Has shipped at least one "
        "end-to-end ranking, search, or recommendation system to real users "
        "at meaningful scale. Has strong, defensible opinions on retrieval "
        "(hybrid vs dense), evaluation (offline vs online), and LLM "
        "integration (fine-tune vs prompt) backed by systems actually "
        "built. Located in or willing to relocate to Noida or Pune. Active "
        "on the Redrob platform or showing clear job-market signal (i.e. "
        "good redrob_signals)."
    )

    # Used directly by Layer 3's consulting-only-career detector.
    consulting_firms: List[str] = [
        "TCS",
        "Tata Consultancy Services",
        "Infosys",
        "Wipro",
        "Accenture",
        "Cognizant",
        "Capgemini",
        # Additional consulting/IT-services firms confirmed present in the
        # actual candidates.jsonl dataset (verified via industry tag
        # "IT Services" during Layer 3 development) but not explicitly
        # named in the JD text itself. Included so Layer 3's
        # consulting-only detector catches the full real population, not
        # just the JD's named examples.
        "HCL",
        "HCLTech",
        "Mphasis",
        "Tech Mahindra",
        "Genpact",
    ]

    return {
        "must_have": must_have,
        "must_have_detail": must_have_detail,
        "nice_to_have": nice_to_have,
        "nice_to_have_detail": nice_to_have_detail,
        "hard_disqualifiers": hard_disqualifiers,
        "hard_disqualifiers_detail": hard_disqualifiers_detail,
        "experience_band_years": experience_band_years,
        "location_preference": location_preference,
        "ideal_profile_notes": ideal_profile_notes,
        "consulting_firms": consulting_firms,
        "role_title": JD_ROLE_TITLE,
        "company": JD_COMPANY,
        "location_text_raw": JD_LOCATION,
        "experience_band_raw": JD_EXPERIENCE_BAND_RAW,
        "redrob_trap_quote": JD_REDROB_TRAP_QUOTE,
    }


# ---------------------------------------------------------------------------
# 3. Self-checks — run on import-time call via __main__, no dataset needed
# ---------------------------------------------------------------------------

def _self_check(reqs: dict) -> None:
    """Raise AssertionError with a clear message if the contract is violated."""

    required_keys = {
        "must_have", "must_have_detail",
        "nice_to_have", "nice_to_have_detail",
        "hard_disqualifiers", "hard_disqualifiers_detail",
        "experience_band_years", "location_preference",
        "ideal_profile_notes", "consulting_firms",
    }
    missing = required_keys - reqs.keys()
    assert not missing, f"Missing required contract keys: {missing}"

    # Type checks
    assert isinstance(reqs["must_have"], list) and len(reqs["must_have"]) == 4, \
        "must_have should be a list of exactly 4 hard requirements per the JD."
    assert isinstance(reqs["nice_to_have"], list) and len(reqs["nice_to_have"]) == 5, \
        "nice_to_have should be a list of exactly 5 items per the JD."
    assert isinstance(reqs["hard_disqualifiers"], list) and len(reqs["hard_disqualifiers"]) == 8, \
        "hard_disqualifiers should be a list of exactly 8 disqualifier codes per the JD."
    assert isinstance(reqs["experience_band_years"], list) and len(reqs["experience_band_years"]) == 2, \
        "experience_band_years must be a [min, max] pair."
    assert reqs["experience_band_years"][0] < reqs["experience_band_years"][1], \
        "experience_band_years min must be less than max."
    assert isinstance(reqs["location_preference"], list) and len(reqs["location_preference"]) > 0, \
        "location_preference must be a non-empty list."
    assert isinstance(reqs["ideal_profile_notes"], str) and len(reqs["ideal_profile_notes"]) > 0, \
        "ideal_profile_notes must be a non-empty string."
    assert isinstance(reqs["consulting_firms"], list) and len(reqs["consulting_firms"]) > 0, \
        "consulting_firms must be a non-empty list."

    # Cross-consistency: every code in hard_disqualifiers must have a detail entry, and vice versa
    codes = set(reqs["hard_disqualifiers"])
    detail_keys = set(reqs["hard_disqualifiers_detail"].keys())
    assert codes == detail_keys, (
        f"Mismatch between hard_disqualifiers codes and hard_disqualifiers_detail keys.\n"
        f"In codes but not detail: {codes - detail_keys}\n"
        f"In detail but not codes: {detail_keys - codes}"
    )

    must_codes = set(reqs["must_have"])
    must_detail_keys = set(reqs["must_have_detail"].keys())
    assert must_codes == must_detail_keys, (
        f"Mismatch between must_have codes and must_have_detail keys.\n"
        f"In codes but not detail: {must_codes - must_detail_keys}\n"
        f"In detail but not codes: {must_detail_keys - must_codes}"
    )

    nice_codes = set(reqs["nice_to_have"])
    nice_detail_keys = set(reqs["nice_to_have_detail"].keys())
    assert nice_codes == nice_detail_keys, (
        f"Mismatch between nice_to_have codes and nice_to_have_detail keys.\n"
        f"In codes but not detail: {nice_codes - nice_detail_keys}\n"
        f"In detail but not codes: {nice_detail_keys - nice_codes}"
    )

    # JSON-serializability check — Layer 2/3 will likely load this from a
    # written file or pass it directly in-memory; either way it must be
    # clean JSON with no exotic types.
    try:
        json.dumps(reqs)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"Requirement object is not JSON-serializable: {exc}")


if __name__ == "__main__":
    requirements = get_jd_requirements()
    _self_check(requirements)

    print("Layer 1 self-check passed. Contract shape is valid.\n")
    print(json.dumps(requirements, indent=2))
