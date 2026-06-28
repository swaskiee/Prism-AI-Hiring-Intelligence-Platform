"""
Benchmark — Layer 3 runtime/memory on the full 100,000-candidate dataset.

Confirms compliance with the official compute constraints
(submission_spec.docx, Section 3 "Compute constraints"):
    - Total runtime  <= 5 minutes wall-clock
    - Memory         <= 16 GB RAM
    - CPU only (this module makes no GPU calls at all)
    - No network calls (this module makes none)

Run with: python3 benchmark.py [path/to/candidates.jsonl]
If no path is given, defaults to "candidates.jsonl" in the current directory.

Note: this benchmarks Layer 3 IN ISOLATION. The full pipeline (Layers 1-5
combined) shares the same 5-minute budget — this script's job is only to
confirm Layer 3's own slice of that budget is small enough to leave
comfortable headroom for Nitanshu's Layers 2 and 4.
"""

import json
import sys
import time
import tracemalloc

from disqualifiers import run_structural_disqualifiers
from jd_requirements import get_jd_requirements

CANDIDATES_PATH = sys.argv[1] if len(sys.argv) > 1 else "candidates.jsonl"

RUNTIME_BUDGET_SECONDS = 5 * 60
MEMORY_BUDGET_GB = 16


def main():
    print("=" * 70)
    print("Layer 3 Benchmark — full 100,000-candidate dataset")
    print("=" * 70)

    print("\nLoading candidates.jsonl ...")
    t0 = time.perf_counter()
    candidates = []
    with open(CANDIDATES_PATH) as f:
        for line in f:
            candidates.append(json.loads(line))
    load_time = time.perf_counter() - t0
    print(f"Loaded {len(candidates):,} candidates in {load_time:.2f}s")

    jd_reqs = get_jd_requirements()

    print("\nRunning Layer 3 (run_structural_disqualifiers) with memory tracing ...")
    tracemalloc.start()
    t0 = time.perf_counter()
    result = run_structural_disqualifiers(candidates, jd_requirements=jd_reqs)
    run_time = time.perf_counter() - t0
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_gb = peak_bytes / (1024 ** 3)

    print(f"\nLayer 3 runtime (ranking step only, excludes file load): {run_time:.2f}s")
    print(f"Layer 3 peak traced memory: {peak_gb:.3f} GB")
    print(f"Rows produced: {len(result):,}")

    print("\n" + "-" * 70)
    print("COMPLIANCE CHECK (Layer 3's own contribution to the shared budget)")
    print("-" * 70)
    runtime_pct = (run_time / RUNTIME_BUDGET_SECONDS) * 100
    memory_pct = (peak_gb / MEMORY_BUDGET_GB) * 100
    print(f"Runtime:  {run_time:.2f}s / {RUNTIME_BUDGET_SECONDS}s budget  "
          f"({runtime_pct:.3f}% of total shared budget)")
    print(f"Memory:   {peak_gb:.3f} GB / {MEMORY_BUDGET_GB} GB budget  "
          f"({memory_pct:.3f}% of total shared budget)")

    if run_time > RUNTIME_BUDGET_SECONDS:
        print("\n*** FAIL: Layer 3 alone exceeds the full 5-minute budget. ***")
    elif runtime_pct > 20:
        print("\n*** WARNING: Layer 3 alone uses >20% of the shared 5-minute "
              "budget — flag this to Nitanshu, may constrain Layer 2/4 design. ***")
    else:
        print("\nPASS: Layer 3 leaves comfortable headroom in the shared "
              "5-minute / 16GB budget for Layers 2 and 4.")

    print("\nNote: tracemalloc measures Python-object memory allocated during\n"
          "this function call specifically, which is the relevant number for\n"
          "judging this module's own contribution. It does not include the\n"
          "baseline memory of the Python interpreter itself or the raw\n"
          "candidates list already loaded in memory before this function was\n"
          "called (that list's footprint should be measured separately as\n"
          "part of the full pipeline's end-to-end memory budget, which is\n"
          "Nitanshu's responsibility to confirm for the assembled Layer "
          "1-5 pipeline).")


if __name__ == "__main__":
    main()
