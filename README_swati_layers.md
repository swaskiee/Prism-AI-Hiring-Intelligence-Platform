# Swati's Layers — Layer 1, Layer 3, Layer 5

Team Hyperion | India Runs Hackathon | Track 1: Data & AI Challenge
Owner: Swati Dubey

This folder contains the three pipeline layers owned by Swati. Nitanshu
owns Layers 2 (semantic embedding scorer) and 4 (signal-weighted final
score combination + submission assembly).

## What's in each folder

### `layer1/` — JD Requirement Extraction
- `jd_requirements.py` — call `get_jd_requirements()` to get the
  structured JD object (must-haves, nice-to-haves, hard disqualifiers,
  experience band, location preferences, consulting-firm list).
  No candidate data needed — pure JD parsing, runs in microseconds.
- `jd_requirements.json` — the same object pre-exported to disk.
- `test_jd_requirements.py` — 5 tests, all passing.

### `layer3/` — Structural Disqualifier Pass
- `disqualifiers.py` — call `run_structural_disqualifiers(candidates, jd_requirements=...)`
  to get a per-candidate DataFrame with 6 rule flags + a soft
  `disqualifier_penalty` (0.0–1.0) to blend into Layer 4's final score.
- `jd_requirements.py` — a copy of Layer 1's module (it's a real
  dependency — `disqualifiers.py` imports `get_jd_requirements()` from it
  to source the consulting-firms list).
- `test_disqualifiers.py` — 31 tests, all passing.
- `benchmark.py` — confirms runtime/memory on the full 100K candidates.
  Run as `python3 benchmark.py path/to/candidates.jsonl` (defaults to
  `candidates.jsonl` in the current folder if no path given).
- `disqualifiers.md` — honest dev notes: 3 real bugs found and fixed
  during development, and why 3 of the 6 rules legitimately fire 0 times
  on this dataset (verified, not a bug). **Worth reading before Stage 5.**
- `sample_candidates.json` — the official 50-row sample, used as a test
  fixture.
- `layer3_output_full.csv` — this layer's output already run against the
  full 100,000 candidates (candidate_id + all 6 flags + penalty + vector).

### `layer5/` — Honeypot & Anomaly Detection
- `honeypot_detection.py` — call `run_honeypot_detection(candidates)` for
  a per-candidate DataFrame with a hard `honeypot_flag` (bool) +
  human-readable evidence string. Also exposes
  `check_honeypot_rate_in_top_n(ranked_ids, honeypot_results)` —
  **run this on the actual final top-100 before submitting** to confirm
  you're under the 10% disqualification threshold (see "Critical" below).
- `test_honeypot_detection.py` — 17 tests, all passing (one regression
  test needs the full `candidates.jsonl` present locally to run; it skips
  gracefully otherwise).
- `benchmark.py` — same usage pattern as Layer 3's.
- `honeypot_dev_notes.md` — the full investigation: what patterns were
  checked, which were confirmed (33 + 21 = 54 honeypots found), and which
  were investigated and explicitly rejected as too common to be
  deliberate traps. **Also worth reading before Stage 5.**
- `sample_candidates.json` — same 50-row sample.
- `layer5_output_full.csv` — this layer's output run against the full
  100,000 candidates.

## How Layer 4 should consume these three layers

```python
from layer1.jd_requirements import get_jd_requirements
from layer3.disqualifiers import run_structural_disqualifiers
from layer5.honeypot_detection import run_honeypot_detection

jd_reqs = get_jd_requirements()
layer3_df = run_structural_disqualifiers(candidates, jd_requirements=jd_reqs)
layer5_df = run_honeypot_detection(candidates)

# Merge on candidate_id with your Layer 2 semantic scores, then:
#   - any candidate with layer5_df.honeypot_flag == True -> exclude or
#     score 0 (hard gate, not a soft penalty)
#   - blend layer3_df.disqualifier_penalty into your final score as a
#     down-weighting multiplier/subtraction (soft signal)
```

**Important — column name note:** both `layer3_df` and `layer5_df` are
keyed by `candidate_id` as a plain column (not the index). Use
`.merge(..., on="candidate_id")` or `.set_index("candidate_id")` as
needed on your side.

## Critical: before final submission

Run `check_honeypot_rate_in_top_n()` (in `layer5/honeypot_detection.py`)
against your actual final top-100 ranking. If `disqualification_risk` is
`True`, the whole team is disqualified at Stage 3 regardless of score —
this is a hard gate, not something to discover after submitting. Pull and
manually eyeball any `flagged_candidate_ids` it returns before trusting
it completely.

## Dataset note

`sample_candidates.json` (50 rows) is included for tests. The full
`candidates.jsonl` (487MB) is NOT included in this repo — it's the
official hackathon dataset, available from the Drive link in the
challenge bundle. Place it alongside the layer folders (or pass its path
to `benchmark.py`) to re-run the full-scale checks.

## Compute budget used by these three layers (measured, not estimated)

| Layer | Runtime (full 100K) | Peak memory |
|---|---|---|
| Layer 1 | ~0.004 ms | negligible |
| Layer 3 | ~4.4s | ~0.04 GB |
| Layer 5 | ~6.2s | ~0.03 GB |
| **Total** | **~11s** | **~0.07 GB** |

Out of the shared budget of 300s / 16GB — leaves the overwhelming
majority of both for Layer 2's embedding model.
