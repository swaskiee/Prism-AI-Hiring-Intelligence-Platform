# Prism Dashboard

`dashboard/index.html` is a single-file, no-backend recruiter dashboard built on top of the real, validated output of `rank.py`. It is a presentation layer, not a scoring component — it does not affect grading, the official `hyperion_submission.csv`, or anything checked by `validate_submission.py`. It exists to make the same pipeline that script-and-CSV represent inspectable and explorable by a human, interactively.

**Why this exists alongside the CLI pipeline:** `rank.py` is what runs inside Redrob's Stage-3 sandboxed reproduction container, and the CSV it produces is the official grading artifact. Neither of those is something a recruiter would actually open and use. The dashboard is the same data, the same scores, the same reasoning — made interactive, searchable, and visually explorable. It is also Track 1's required sandbox/demo asset.

## What it shows

- **Live stats** pulled from the real pipeline run (candidates scanned, Layer 3 flag count, Layer 5 honeypot count, honeypot rate in the final top-100, total runtime).
- **The full top-100 ranking**, searchable by title/company/candidate ID, filterable by rank tier or structural-flag status.
- **Per-candidate drill-down** — click any row to expand the full score breakdown (semantic fit, trust multiplier, disqualifier penalty), every structural/honeypot check with its result, the specific JD requirements matched, and the generated reasoning string — i.e., the same explainability data Stage 4 manual review checks, but browsable live instead of read from a flat CSV.

## How it's built

A static HTML/CSS/vanilla-JS file with the candidate data embedded directly as a JSON blob in a `<script>` tag — no server, no build step, no framework, no external runtime dependency beyond two Google Fonts loaded over HTTPS. This was a deliberate choice for the same reason Layer 2 avoided a heavy ML dependency: zero install/hosting friction. Opening the file in any browser is the entire deployment story; it can also be dropped onto GitHub Pages, Vercel, or Netlify with no configuration.

## Regenerating it after a pipeline change

If you change any layer's logic (e.g. retune Layer 4's weights, add a Layer 3 rule), the dashboard's embedded data goes stale and needs regenerating. Two steps:

```bash
# 1. Re-run the full pipeline and export the rich per-candidate breakdown
#    (this is NOT the official submission CSV — it's a superset, for the
#    dashboard only — see export_dashboard_data.py's own docstring)
python3 export_dashboard_data.py --candidates path/to/candidates.jsonl --out dashboard/dashboard_data.json --top-n 100

# 2. Minify and re-embed it into dashboard/index.html
python3 -c "
import json
with open('dashboard/dashboard_data.json') as f:
    data = json.load(f)
minified = json.dumps(data)
with open('dashboard/index.html') as f:
    html = f.read()
import re
html = re.sub(r'const DASHBOARD_DATA = .*?;\n', f'const DASHBOARD_DATA = {minified};\n', html, count=1, flags=re.S)
with open('dashboard/index.html', 'w') as f:
    f.write(html)
print('Dashboard data refreshed.')
"
```

Then delete the intermediate `dashboard/dashboard_data.json` before committing — only `dashboard/index.html` (with the data already embedded) needs to be tracked.

## Honest scope notes

- This is a **read-only viewer** over a single completed pipeline run — it does not call `rank.py` live, re-rank on the fly, or talk to any backend. Re-running the pipeline requires the regeneration steps above.
- The 100 candidates shown are exactly the 100 rows in `hyperion_submission.csv` — same scores, same ranks, same reasoning text. Nothing is recalculated independently in the browser; it is the same numbers, visualized.
- Best opened directly via the GitHub repo's raw file link, a static host (GitHub Pages / Vercel / Netlify, zero-config), or just locally via `open dashboard/index.html`.
