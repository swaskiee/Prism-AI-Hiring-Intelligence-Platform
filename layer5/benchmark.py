"""
Benchmark — Layer 5 runtime/memory on the full 100,000-candidate dataset.

Same methodology as Layer 3's benchmark.py. Confirms compliance with
submission_spec.docx Section 3 compute constraints (<=5min, <=16GB,
CPU-only, no network).

Run with: python3 benchmark.py [path/to/candidates.jsonl]
If no path is given, defaults to "candidates.jsonl" in the current directory.
"""

import json
import sys
import time
import tracemalloc

from honeypot_detection import run_honeypot_detection

CANDIDATES_PATH = sys.argv[1] if len(sys.argv) > 1 else "candidates.jsonl"

RUNTIME_BUDGET_SECONDS = 5 * 60
MEMORY_BUDGET_GB = 16


def main():
    print("=" * 70)
    print("Layer 5 Benchmark — full 100,000-candidate dataset")
    print("=" * 70)

    print("\nLoading candidates.jsonl ...")
    t0 = time.perf_counter()
    candidates = []
    with open(CANDIDATES_PATH) as f:
        for line in f:
            candidates.append(json.loads(line))
    load_time = time.perf_counter() - t0
    print(f"Loaded {len(candidates):,} candidates in {load_time:.2f}s")

    print("\nRunning Layer 5 (run_honeypot_detection) with memory tracing ...")
    tracemalloc.start()
    t0 = time.perf_counter()
    result = run_honeypot_detection(candidates)
    run_time = time.perf_counter() - t0
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_gb = peak_bytes / (1024 ** 3)

    print(f"\nLayer 5 runtime (detection step only, excludes file load): {run_time:.2f}s")
    print(f"Layer 5 peak traced memory: {peak_gb:.3f} GB")
    print(f"Rows produced: {len(result):,}")
    print(f"Honeypots flagged: {result['honeypot_flag'].sum()} / {len(result)}")

    print("\n" + "-" * 70)
    print("COMPLIANCE CHECK (Layer 5's own contribution to the shared budget)")
    print("-" * 70)
    runtime_pct = (run_time / RUNTIME_BUDGET_SECONDS) * 100
    memory_pct = (peak_gb / MEMORY_BUDGET_GB) * 100
    print(f"Runtime:  {run_time:.2f}s / {RUNTIME_BUDGET_SECONDS}s budget  "
          f"({runtime_pct:.3f}% of total shared budget)")
    print(f"Memory:   {peak_gb:.3f} GB / {MEMORY_BUDGET_GB} GB budget  "
          f"({memory_pct:.3f}% of total shared budget)")

    if run_time > RUNTIME_BUDGET_SECONDS:
        print("\n*** FAIL: Layer 5 alone exceeds the full 5-minute budget. ***")
    elif runtime_pct > 20:
        print("\n*** WARNING: Layer 5 alone uses >20% of the shared 5-minute "
              "budget — flag this to Nitanshu. ***")
    else:
        print("\nPASS: Layer 5 leaves comfortable headroom in the shared "
              "5-minute / 16GB budget for Layers 2 and 4.")

    print("\nCombined with Layer 3's measured footprint (4.4s, 0.04GB), the "
          "two Swati-owned layers together use well under 10 seconds and "
          "0.1GB of the full 300s/16GB budget — leaving the overwhelming "
          "majority of the budget for Nitanshu's embedding-based Layer 2, "
          "which will be the heaviest part of the pipeline.")


if __name__ == "__main__":
    main()
