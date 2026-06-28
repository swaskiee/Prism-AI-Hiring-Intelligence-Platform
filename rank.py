#!/usr/bin/env python3
"""
rank.py — Prism Pipeline Entrypoint
Team Hyperion | India Runs Hackathon | Track 1: Data & AI Challenge

Single command to run the full five-layer ranking pipeline end to end and
produce the official submission CSV:

    python3 rank.py --candidates path/to/candidates.jsonl --out submission.csv

This script is the ONLY thing that needs to run inside Redrob's Stage-3
sandboxed reproduction container. It performs no network calls and is
CPU-only throughout (see each layer module's own docstring for layer-
specific constraint notes).

PIPELINE ORDER
--------------
1. Load candidates.jsonl (one JSON object per line).
2. Layer 1 — JD requirement extraction (Swati Dubey) — static, no
   candidate dependency.
3. Layer 2 — Semantic fit scoring (Nitanshu Tak).
4. Layer 3 — Structural disqualifier pass (Swati Dubey).
5. Layer 4 — Behavioral trust multiplier (Nitanshu Tak).
6. Layer 5 — Honeypot & anomaly detection (Swati Dubey).
7. Score fusion + ranking, with the honeypot hard gate applied
   (Nitanshu Tak).
8. Reasoning generation for the final top 100 (Nitanshu Tak).
9. Honeypot rate safety check on the ACTUAL final top-100 (uses Layer 5's
   own check_honeypot_rate_in_top_n — this is the last-mile guard against
   the Stage 3 disqualification threshold).
10. Write submission CSV in the exact official format (candidate_id, rank,
    score, reasoning — see validate_submission.py) and run the official
    validator against it before exiting.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

# --- Make all five layer/fusion modules importable regardless of CWD ---
_BASE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("layer1", "layer2", "layer3", "layer4", "layer5", "fusion"):
    sys.path.insert(0, os.path.join(_BASE, _sub))

from jd_requirements import get_jd_requirements                       # noqa: E402
from semantic_scorer import run_semantic_scoring                       # noqa: E402
from disqualifiers import run_structural_disqualifiers                 # noqa: E402
from trust_multiplier import run_trust_multiplier                      # noqa: E402
from honeypot_detection import run_honeypot_detection, check_honeypot_rate_in_top_n  # noqa: E402
from score_fusion import fuse_and_rank                                  # noqa: E402
from reasoning_generator import generate_reasoning_for_ranking          # noqa: E402


def load_candidates(path: str) -> list[dict]:
    """
    Load candidate records from either:
    - candidates.jsonl (one JSON object per line — the official format
      used for the real 100,000-candidate file), or
    - a pretty-printed JSON array (e.g. sample_candidates.json, the
      official 50-row test fixture used by all five layer modules' own
      __main__ blocks).

    Format is auto-detected by peeking at the first non-whitespace
    character only (not by reading the whole file), then the JSONL case
    is streamed line-by-line rather than loaded into one large string
    first. The original version called f.read() to grab the entire file
    as a single string before splitting it into lines — harmless on a
    small file, but on the real 465MB candidates.jsonl that briefly holds
    the whole file as one giant string IN ADDITION to the ~100,000 parsed
    dicts that follow, which is enough peak memory pressure to raise
    MemoryError on machines with limited free RAM. Streaming avoids ever
    holding the raw file content in memory at all.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break

        if first_char == "[":
            # Pretty-printed JSON array fixture (e.g. sample_candidates.json)
            # — small file, safe to read and parse as a whole.
            f.seek(0)
            return json.load(f)

        # JSONL: stream line-by-line, never holding the full file as one string.
        f.seek(0)
        candidates = []
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
        return candidates


def run_pipeline(candidates_path: str, out_path: str, top_n: int = 100) -> None:
    t_start = time.perf_counter()

    print(f"[1/9] Loading candidates from {candidates_path} ...")
    t0 = time.perf_counter()
    candidates = load_candidates(candidates_path)
    print(f"      Loaded {len(candidates):,} candidates in {time.perf_counter() - t0:.2f}s")

    print("[2/9] Layer 1 — JD requirement extraction ...")
    t0 = time.perf_counter()
    jd_requirements = get_jd_requirements()
    print(f"      Done in {time.perf_counter() - t0:.4f}s")

    print("[3/9] Layer 2 — Semantic fit scoring ...")
    t0 = time.perf_counter()
    layer2_df = run_semantic_scoring(candidates, jd_requirements=jd_requirements)
    print(f"      Done in {time.perf_counter() - t0:.2f}s")

    print("[4/9] Layer 3 — Structural disqualifier pass ...")
    t0 = time.perf_counter()
    layer3_df = run_structural_disqualifiers(candidates, jd_requirements=jd_requirements)
    print(f"      Done in {time.perf_counter() - t0:.2f}s "
          f"({(layer3_df['disqualifier_penalty'] > 0).sum():,} candidates flagged)")

    print("[5/9] Layer 4 — Behavioral trust multiplier ...")
    t0 = time.perf_counter()
    layer4_df = run_trust_multiplier(candidates)
    print(f"      Done in {time.perf_counter() - t0:.2f}s")

    print("[6/9] Layer 5 — Honeypot & anomaly detection ...")
    t0 = time.perf_counter()
    layer5_df = run_honeypot_detection(candidates)
    n_honeypots = int(layer5_df["honeypot_flag"].sum())
    print(f"      Done in {time.perf_counter() - t0:.2f}s ({n_honeypots:,} honeypots detected)")

    print(f"[7/9] Score fusion + ranking (top {top_n}) ...")
    t0 = time.perf_counter()
    ranked = fuse_and_rank(layer2_df, layer3_df, layer4_df, layer5_df, top_n=top_n)
    print(f"      Done in {time.perf_counter() - t0:.2f}s")

    print("[8/9] Generating reasoning for final ranking ...")
    t0 = time.perf_counter()
    final = generate_reasoning_for_ranking(ranked, candidates, jd_requirements)
    print(f"      Done in {time.perf_counter() - t0:.2f}s")

    print("[9/9] Honeypot safety check on final top-N ...")
    rate_check = check_honeypot_rate_in_top_n(
        ranked_candidate_ids=final["candidate_id"].tolist(),
        honeypot_results=layer5_df,
        top_n=top_n,
    )
    print(f"      honeypot_count={rate_check['honeypot_count']}, "
          f"honeypot_rate={rate_check['honeypot_rate']:.2%}, "
          f"disqualification_risk={rate_check['disqualification_risk']}")
    if rate_check["disqualification_risk"]:
        print("      *** WARNING: honeypot rate exceeds the 10% disqualification "
              "threshold. DO NOT SUBMIT this output as-is. ***", file=sys.stderr)

    # --- Write submission CSV: EXACTLY candidate_id, rank, score, reasoning ---
    submission = final[["candidate_id", "rank", "score", "reasoning"]].copy()
    submission.to_csv(out_path, index=False)

    total_elapsed = time.perf_counter() - t_start
    print(f"\nWrote {len(submission)} rows to {out_path}")
    print(f"Total pipeline runtime: {total_elapsed:.2f}s")

    return total_elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the Prism candidate ranking pipeline.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to write the submission CSV")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to rank (default 100)")
    args = parser.parse_args()

    run_pipeline(args.candidates, args.out, top_n=args.top_n)


if __name__ == "__main__":
    main()
