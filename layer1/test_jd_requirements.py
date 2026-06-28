"""
Tests for Layer 1 — JD Requirement Extraction
Run with: python3 test_jd_requirements.py
No dataset required.
"""

import json
from jd_requirements import get_jd_requirements, _self_check


def test_contract_shape():
    reqs = get_jd_requirements()
    _self_check(reqs)
    print("[PASS] test_contract_shape")


def test_deterministic_output():
    """Calling get_jd_requirements() twice must produce identical output —
    this is a static parsing module, not something with hidden randomness
    or external state."""
    r1 = get_jd_requirements()
    r2 = get_jd_requirements()
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True), \
        "get_jd_requirements() is not deterministic across calls."
    print("[PASS] test_deterministic_output")


def test_consulting_exception_documented():
    """The consulting-only-career exception (currently at a consulting firm
    but with prior product-company experience = NOT disqualified) must be
    explicitly present in the detail text, since Layer 3 depends on this
    exact nuance."""
    reqs = get_jd_requirements()
    detail = reqs["hard_disqualifiers_detail"]["pure_consulting_career_no_product_co"]
    assert "EXCEPTION" in detail, "Consulting exception clause missing from detail text."
    print("[PASS] test_consulting_exception_documented")


def test_no_duplicate_codes_across_categories():
    """A code should not accidentally appear in both must_have and
    hard_disqualifiers, etc. — would indicate a copy-paste error."""
    reqs = get_jd_requirements()
    all_codes = reqs["must_have"] + reqs["nice_to_have"] + reqs["hard_disqualifiers"]
    assert len(all_codes) == len(set(all_codes)), "Duplicate codes found across categories."
    print("[PASS] test_no_duplicate_codes_across_categories")


def test_json_roundtrip():
    """Confirms the object can be written to disk and read back identically
    — this is how it will likely be handed to Layer 2/3 in practice."""
    reqs = get_jd_requirements()
    serialized = json.dumps(reqs)
    deserialized = json.loads(serialized)
    assert deserialized == reqs, "JSON round-trip produced a different object."
    print("[PASS] test_json_roundtrip")


if __name__ == "__main__":
    test_contract_shape()
    test_deterministic_output()
    test_consulting_exception_documented()
    test_no_duplicate_codes_across_categories()
    test_json_roundtrip()
    print("\nAll Layer 1 tests passed.")
