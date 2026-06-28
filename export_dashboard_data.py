#!/usr/bin/env python3
"""
export_dashboard_data.py — Team Hyperion / Prism
Produces a rich JSON file for the recruiter dashboard, containing the
full per-candidate breakdown (all layer sub-scores + evidence + profile
fields) for the final top-100 ranking. This is presentation-layer data
ONLY — the official submission.csv (candidate_id, rank, score, reasoning)
remains the single source of truth for grading; this file exists purely
to power the interactive dashboard described in DASHBOARD.md.
"""

import json
import os
import sys
import time

_BASE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("layer1", "layer2", "layer3", "layer4", "layer5", "fusion"):
    sys.path.insert(0, os.path.join(_BASE, _sub))

from jd_requirements import get_jd_requirements
from semantic_scorer import run_semantic_scoring
from disqualifiers import run_structural_disqualifiers
from trust_multiplier import run_trust_multiplier
from honeypot_detection import run_honeypot_detection, check_honeypot_rate_in_top_n
from score_fusion import fuse_and_rank
from reasoning_generator import generate_reasoning_for_ranking

import pandas as pd


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    stripped = content.lstrip()
    if stripped.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def main(candidates_path: str, out_path: str, top_n: int = 100):
    t0 = time.perf_counter()
    print(f"Loading {candidates_path} ...")
    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates):,} candidates")

    jd_requirements = get_jd_requirements()

    print("Running all five layers ...")
    layer2_df = run_semantic_scoring(candidates, jd_requirements=jd_requirements)
    layer3_df = run_structural_disqualifiers(candidates, jd_requirements=jd_requirements)
    layer4_df = run_trust_multiplier(candidates)
    layer5_df = run_honeypot_detection(candidates)

    ranked = fuse_and_rank(layer2_df, layer3_df, layer4_df, layer5_df, top_n=top_n)
    final = generate_reasoning_for_ranking(ranked, candidates, jd_requirements)

    rate_check = check_honeypot_rate_in_top_n(
        ranked_candidate_ids=final["candidate_id"].tolist(),
        honeypot_results=layer5_df,
        top_n=top_n,
    )

    candidate_lookup = {c["candidate_id"]: c for c in candidates}
    # Merge in full Layer 3 + Layer 5 detail (individual flags + evidence)
    # for the dashboard's per-candidate drill-down view.
    layer3_lookup = layer3_df.set_index("candidate_id").to_dict("index")
    layer5_lookup = layer5_df.set_index("candidate_id").to_dict("index")

    records = []
    for _, row in final.iterrows():
        cid = row["candidate_id"]
        c = candidate_lookup[cid]
        l3 = layer3_lookup[cid]
        l5 = layer5_lookup[cid]

        records.append({
            "candidate_id": cid,
            "rank": int(row["rank"]),
            "score": round(float(row["score"]), 4),
            "reasoning": row["reasoning"],
            "profile": {
                "current_title": c["profile"]["current_title"],
                "current_company": c["profile"]["current_company"],
                "years_of_experience": c["profile"]["years_of_experience"],
                "location": c["profile"]["location"],
                "headline": c["profile"]["headline"],
            },
            "scores": {
                "semantic_fit_score": round(float(row["semantic_fit_score"]), 4),
                "trust_multiplier": round(float(row["trust_multiplier"]), 4),
                "disqualifier_penalty": round(float(row["disqualifier_penalty"]), 4),
                "final_score": round(float(row["score"]), 4),
            },
            "matched_requirements": (row["matched_requirements"] or "").split(",") if row["matched_requirements"] else [],
            "disqualifier_flags": {
                k: bool(v) for k, v in l3.items()
                if k.endswith("_flag")
            },
            "honeypot": {
                "flagged": bool(l5["honeypot_flag"]),
                "evidence": l5["honeypot_evidence"],
            },
        })

    summary = {
        "generated_for_jd": jd_requirements["role_title"],
        "total_candidates_scanned": len(candidates),
        "layer3_total_flagged": int((layer3_df["disqualifier_penalty"] > 0).sum()),
        "layer5_total_honeypots": int(layer5_df["honeypot_flag"].sum()),
        "honeypot_rate_in_top_n": rate_check["honeypot_rate"],
        "disqualification_risk": rate_check["disqualification_risk"],
        "top_n": top_n,
        "candidates": records,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote dashboard data ({len(records)} candidates) to {out_path}")
    print(f"Total time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="dashboard/dashboard_data.json")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    main(args.candidates, args.out, args.top_n)
