<div align="center">

<a href="https://github.com/swaskiee/Prism-AI-Hiring-Intelligence-Platform" target="_blank">
  <img src="./prism-logo.png" alt="Prism Logo" width="320" height="320"/>
</a>

<p>
  <img src="https://img.shields.io/badge/Hackathon-India%20Runs-5B2E91?style=for-the-badge">
  <img src="https://img.shields.io/badge/Challenge-Redrob%20AI-111111?style=for-the-badge">
  <img src="https://img.shields.io/badge/Team-Hyperion-orange?style=for-the-badge">
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/TF--IDF%20%2B%20SVD-LSA-FF6F00?style=for-the-badge">
</p>

<p>
  <img src="https://img.shields.io/badge/Runtime-100s%20%2F%20100K%20candidates-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Peak%20Memory-~3GB-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Compute-CPU%20Only-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Network-Disabled%20at%20Inference-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Cost-₹0-brightgreen?style=for-the-badge">
</p>

### 🧠 Hybrid Ranking &nbsp;•&nbsp; 🛡️ Honeypot Defense &nbsp;•&nbsp; 🔍 Explainable Scoring &nbsp;•&nbsp; ⚡ Zero-LLM Inference

</br>
</div>

**An Explainable, Trap-Resistant Candidate Ranking Engine**

*Built for Redrob AI's Intelligent Candidate Discovery & Ranking Challenge*

[![Python](https://img.shields.io/badge/Python-3.12%2F3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.x%2F3.x-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-97BC62?style=flat-square)](LICENSE)

---

*Built for the **India Runs Hackathon** by Hack2skill × Redrob AI* *Track 1 — The Data & AI Challenge: Intelligent Candidate Discovery*

---

### **Authors & Contributors**
* **Nitanshu Tak** — Semantic Scoring, Behavioral Signal Weighting, Score Fusion, Explainability Engine, Pipeline Architecture & Integration, Recruiter Dashboard
* **Swati Dubey** — JD Requirement Extraction, Structural Disqualifier Engine, Honeypot & Anomaly Detection

</div>

---

## The Problem

Redrob AI's recruiters search a pool of hundreds of thousands of candidate profiles for every open role. Keyword-based filtering looks for surface matches between a skills list and a job description, and as a result fails in two opposite directions at once.

**It misses good candidates.** Someone who built a production recommendation system at a real product company, but whose profile never happens to contain the word "RAG" or "Pinecone," gets filtered out by a keyword scan even though their actual engineering substance is a strong match.

**It promotes bad candidates.** Someone whose skills section lists every AI buzzword in existence, but whose actual job title and career history have nothing to do with engineering, scores artificially high on keyword density alone.

Redrob's own job description for this exact role states the trap directly: *the right answer is not "find candidates whose skills section contains the most AI keywords."* Prism is built specifically to close that gap — to rank the way an experienced technical recruiter actually would.

---

## What Prism Does

Prism is a five-layer hybrid ranking engine that takes Redrob's 100,000-candidate pool and the Senior AI Engineer job description, and produces a ranked top-100 shortlist — each entry with a specific, evidence-grounded explanation for its rank.

**It reads the JD, not just its keywords.** The job description is decomposed into structured must-haves, nice-to-haves, hard disqualifiers (with their exceptions), and an explicit "ideal candidate" profile.

**It scores meaning, not vocabulary.** A locally-trained latent semantic model compares the substance of a candidate's career history against the JD's actual requirements — a candidate who clearly did the work without using trendy terminology still scores correctly.

**It applies the structural judgment a senior recruiter would.** An independent rule-based layer checks title sanity, consulting-only career patterns (with Redrob's own explicit exception correctly applied), job-hopping patterns, and domain mismatches — catching exactly what semantic similarity alone would miss.

**It treats "available" as part of "qualified."** Behavioral signals — recruiter response rate, login recency, interview completion rate, open-to-work status — multiply the skill-fit score, so a perfect-on-paper candidate who has gone quiet for months is ranked accordingly.

**It refuses to be fooled by impossible data.** An anomaly layer scans every profile for internal contradictions and hard-excludes confirmed honeypots before they can ever reach the top 100.

**Every rank comes with a reason, generated from real numbers.** No templates, no LLM call, no boilerplate — each top-100 entry's justification is built directly from that candidate's own computed sub-scores and profile fields.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│   candidates.jsonl (100,000 profiles)      job_description.md        │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│             LAYER 1 — JD REQUIREMENT EXTRACTION  ·  Swati Dubey       │
│   Unstructured JD text → structured requirement object                │
│   must_have[4] · nice_to_have[5] · hard_disqualifiers[8]              │
│   experience_band · location_preference · consulting_firms[]          │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
┌─────────────────────────────┐ ┌─────────────────────────────────────┐
│ LAYER 2 — SEMANTIC FIT      │ │ LAYER 3 — STRUCTURAL DISQUALIFIER     │
│        Nitanshu Tak           │ │            Swati Dubey                │
│ TF-IDF + Truncated SVD (LSA) │ │ Title sanity · consulting-only       │
│ JD requirements ↔ career text │ │ job-hop · stale-architect · domain    │
│ Cosine similarity, weighted   │ │ 14,883 / 100,000 candidates flagged   │
└──────────────┬────────────────┘ └───────────────────┬───────────────────┘
               │                                       │
               ▼                                       ▼
┌─────────────────────────────┐ ┌─────────────────────────────────────┐
│ LAYER 4 — BEHAVIORAL TRUST  │ │ LAYER 5 — HONEYPOT & ANOMALY         │
│       Nitanshu Tak            │ │           Swati Dubey                 │
│ response rate · recency      │ │ Impossible tenure · zero-duration    │
│ interview completion          │ │ "expert" skills                       │
│ Multiplicative, bounded       │ │ HARD exclusion gate, not a penalty    │
│ [0.25, 1.0]                   │ │ 54 / 100,000 honeypots detected       │
└──────────────┬────────────────┘ └───────────────────┬───────────────────┘
               │                                       │
               └───────────────────┬───────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│            SCORE FUSION & RANKING  ·  Nitanshu Tak                    │
│   final_score = semantic_fit_score × trust_multiplier                │
│                  × (1 − disqualifier_penalty)                        │
│   Honeypots hard-excluded → sorted → top 100, official tie-break     │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│           EXPLAINABILITY ENGINE  ·  Nitanshu Tak                      │
│   Per-candidate reasoning generated from real computed sub-scores    │
│   No templates · no LLM call · no hallucination risk by construction │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
              hyperion_submission.csv (top 100, ranked)
        Validated against the official validate_submission.py ✓
```

---

## The Dashboard — Making the Ranking Explorable, Not Just Gradable

`rank.py` and `hyperion_submission.csv` are what get graded — a CLI script and a CSV is the correct, sandbox-compatible artifact for Stage 3. But it isn't something a recruiter at Redrob would ever actually open and use, and a flat CSV makes the explainability this system is built around harder to *see* than it should be.

**`dashboard/index.html`** is a single-file, no-backend interactive viewer over the exact same validated output — the real top-100, the real scores, the real reasoning — built so a non-technical reviewer can search, filter, and drill into *why* each candidate ranked where they did, live, instead of reading 100 rows of a CSV.

Open it directly in any browser (zero install, zero server) or host it for free on GitHub Pages / Vercel. See `dashboard/DASHBOARD.md` for details.

---

## Why This Architecture, Specifically

Every design choice below maps to a specific constraint or trap in Redrob's own challenge spec — none of it is incidental.

**No hosted LLM calls during ranking.** The compute budget is 5 minutes wall-clock, 16 GB RAM, CPU-only, no network, for the full 100,000-candidate pool. Every layer in Prism runs locally; the actual measured total runtime on a real run is **~100–170 seconds** depending on hardware — comfortably under budget with real margin.

**TF-IDF + Truncated SVD instead of a sentence-transformer model — a deliberate tradeoff, not a downgrade.** A transformer model requires PyTorch and either bundled weights or a one-time download, both of which add real failure surface inside an unfamiliar sandboxed reproduction container with no network access. TF-IDF + Truncated SVD is classical Latent Semantic Analysis, with decades of information-retrieval literature behind it. Fit on this specific 100,000-candidate corpus, it captures the JD-specific semantic structure transparently — every dimension traces back to real vocabulary in this dataset, which is a more concrete defense than "the transformer's attention weights decided this."

**Rules alongside embeddings, not instead of them.** Semantic similarity alone would still rank a buzzword-heavy non-engineering profile highly, because the words are semantically close to the JD. Layer 3 catches what embedding similarity cannot — measurably: **14,883** candidates flagged across the full dataset.

**Behavioral signals as a multiplier, not an afterthought.** A perfect-on-paper candidate with a low recruiter response rate and months of inactivity is, for hiring purposes, not actually available. Layer 4 encodes this directly as a multiplicative factor bounded to [0.25, 1.0].

**A hard honeypot gate, not a soft penalty.** Submissions with a honeypot rate above 10% in the top 100 are disqualified outright. Layer 5 is a hard pre-filter for exactly this reason. **54 honeypots detected** across the dataset, **0 honeypots in the final top-100** — a 0.00% rate.

**Reasoning generated from real sub-scores, not a second LLM pass.** Building the explanation directly from the same numbers that produced the rank makes hallucination structurally impossible, not just unlikely.

**Tie-breaking matches the official validator exactly.** `validate_submission.py` requires that equal scores break by ascending `candidate_id`. `fusion/score_fusion.py`'s `assign_ranks()` sorts on `(-final_score, candidate_id)` for exactly this reason.

---

## Results — Measured on the Real Dataset

| Metric | Value |
|---|---|
| Total pipeline runtime (full 100,000 candidates) | **~100–170 seconds**, hardware-dependent (of a 300-second budget) |
| Peak memory usage | **~3 GB** (of a 16 GB budget) |
| Candidates loaded | 100,000 |
| Layer 3 — candidates flagged by at least one disqualifier rule | 14,883 (14.9%) |
| Layer 5 — honeypots detected across full dataset | 54 |
| Honeypot rate in final top-100 | **0.00%** (0 / 100) |
| Official `validate_submission.py` result | **"Submission is valid."** |
| Final top-100 score range | 0.3800 (rank 100) → 0.4796 (rank 1) |

**Real reasoning output, top of the actual run:**

```
CAND_0077337, rank 1, score 0.4796:
"7.0 years of experience, currently Staff Machine Learning Engineer at
Paytm. Career history shows direct evidence of production embeddings-
based retrieval, production vector DB / hybrid search experience,
demonstrated Python systems experience, ranking evaluation framework
experience, learning-to-rank modeling, HR-tech/marketplace background,
distributed systems / inference optimization, and open-source AI/ML
contributions. Behavioral signals (response rate, recent activity)
indicate a highly engaged, available candidate."
```

---

## Repository Structure

```
Prism-AI-Hiring-Intelligence-Platform/
│
├── layer1/                          JD Requirement Extraction — Swati Dubey
│   ├── jd_requirements.py           get_jd_requirements() — structured JD object
│   ├── jd_requirements.json         Same object, pre-exported to disk
│   └── test_jd_requirements.py      5 tests, all passing
│
├── layer2/                          Semantic Fit Scorer — Nitanshu Tak
│   ├── semantic_scorer.py           run_semantic_scoring() — TF-IDF + SVD cosine similarity
│   └── sample_candidates.json       Official 50-row test fixture
│
├── layer3/                          Structural Disqualifier Pass — Swati Dubey
│   ├── disqualifiers.py             run_structural_disqualifiers() — 6 rule detectors
│   ├── jd_requirements.py           Shared dependency (consulting_firms list)
│   ├── disqualifiers.md             Dev notes: bugs found/fixed, honest 0-fire reporting
│   ├── test_disqualifiers.py        31 tests, all passing
│   ├── benchmark.py                 Runtime/memory confirmation on full 100K
│   ├── sample_candidates.json       Official 50-row test fixture
│   └── layer3_output_full.csv       This layer's output, run on the full 100,000 candidates
│
├── layer4/                          Behavioral Trust Multiplier — Nitanshu Tak
│   ├── trust_multiplier.py          run_trust_multiplier() — 4-signal multiplicative weighting
│   └── sample_candidates.json       Official 50-row test fixture
│
├── layer5/                          Honeypot & Anomaly Detection — Swati Dubey
│   ├── honeypot_detection.py        run_honeypot_detection() + check_honeypot_rate_in_top_n()
│   ├── honeypot_dev_notes.md        Full investigation: confirmed + rejected patterns
│   ├── test_honeypot_detection.py   17 tests, all passing
│   ├── benchmark.py                 Runtime/memory confirmation on full 100K
│   ├── sample_candidates.json       Official 50-row test fixture
│   └── layer5_output_full.csv       This layer's output, run on the full 100,000 candidates
│
├── fusion/                          Score Fusion & Explainability — Nitanshu Tak
│   ├── score_fusion.py              fuse_and_rank() + assign_ranks() — official tie-break logic
│   └── reasoning_generator.py       generate_reasoning_for_ranking() — grounded explanations
│
├── dashboard/                       Recruiter-facing dashboard — Nitanshu Tak
│   ├── index.html                   Single-file interactive viewer over the real top-100 output
│   └── DASHBOARD.md                 What it is, how to regenerate it, honest scope notes
│
├── export_dashboard_data.py         Generates the dashboard's per-candidate breakdown JSON
├── rank.py                          Single pipeline entrypoint (see Local Setup below)
├── validate_submission.py           Official format validator (shipped with this repo)
├── hyperion_submission.csv          Final top-100 ranked output — validated, ready to submit
├── WINDOWS_SETUP.md                 Windows-specific setup notes (multi-Python-install fix)
├── prism-logo.png
└── README.md                        This file
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.12 / 3.13 | Required by spec; mature ML/data ecosystem |
| Semantic Embeddings | scikit-learn `TfidfVectorizer` + `TruncatedSVD` (LSA) | CPU-only, zero GPU/network dependency, fully local, classical and defensible |
| Data Processing | Pandas / NumPy | Vectorized operations across 100,000 records, streamed I/O for low memory footprint |
| Rule Engine | Pure Python (`disqualifiers.py`, `honeypot_detection.py`) | Deterministic, explainable, auditable line-by-line |
| Score Fusion | Weighted multiplicative combination, fully documented formula | Multiplicative by design — see Score Fusion docstring for full rationale |
| Output Format | CSV (UTF-8) | Per official submission spec; validated against the real `validate_submission.py` |
| **Total Cost** | **₹0** | Entirely local, open-source tooling — no paid API usage anywhere in the pipeline |

---

## Local Setup

```bash
git clone https://github.com/swaskiee/Prism-AI-Hiring-Intelligence-Platform.git
cd Prism-AI-Hiring-Intelligence-Platform
pip install pandas numpy scikit-learn scipy
```

**Windows users:** if you have multiple Python versions installed, see `WINDOWS_SETUP.md` — using `py -0` to check installed versions and `py -3.13` (or your version) consistently for every command avoids the most common setup issue.

### Running the Full Ranking Pipeline

```bash
python rank.py --candidates path/to/candidates.jsonl --out hyperion_submission.csv
```

### Validating the Submission

```bash
python validate_submission.py hyperion_submission.csv
# → Submission is valid.
```

### Running Each Layer Standalone

```bash
python layer1/jd_requirements.py
python layer2/semantic_scorer.py
python layer3/disqualifiers.py
python layer4/trust_multiplier.py
python layer5/honeypot_detection.py
python fusion/score_fusion.py
python fusion/reasoning_generator.py
```

### Running Tests

```bash
cd layer1 && python -m pytest test_jd_requirements.py -v
cd ../layer3 && python -m pytest test_disqualifiers.py -v
cd ../layer5 && python -m pytest test_honeypot_detection.py -v
```

---

## Evaluation Awareness

| Stage | What It Checks | How Prism Addresses It |
|---|---|---|
| 1. Format validation | CSV structure, exactly 100 rows, valid IDs | Confirmed locally with the official `validate_submission.py` — passes |
| 2. Scoring | NDCG/MAP/P@10 against hidden ground truth | Architecture optimized for precision at the top via multiplicative fusion |
| 3. Code reproduction + honeypot check | Sandboxed 5-min/16GB/CPU/no-network run; >10% honeypot rate disqualifies | Measured well under both limits; honeypot rate confirmed at 0.00% |
| 4. Manual review | Reasoning quality, hallucination, rank-consistency, git history authenticity | Reasoning is field-grounded by construction; real incremental git history with documented bug-fix commits |
| 5. Defend-your-work interview | Live defense of architecture | Every design choice is documented here and in per-layer dev-notes files specifically for this conversation |

---

## Beyond This Challenge — Why This Generalizes

- **Layer 1 is the only JD-specific code.** Swap in a different job description, and Layers 2–5 run unchanged.
- **The structural disqualifier and honeypot logic are role-agnostic by construction** — title-mismatch detection, career-pattern sanity checks, and the two confirmed honeypot mechanisms apply to any technical role.
- **Zero-cost, zero-dependency-risk by design.** No paid API, no GPU, no hosted model — realistically deployable by any team with a laptop CPU.
- **The dashboard is a template, not a one-off.** Point `export_dashboard_data.py` at a different ranking run and the same interactive viewer works for a different JD, a different candidate pool, a different team.

---

## Team

### Nitanshu Tak
**B.Tech CSE (Cloud Computing & Virtualization Technology) · UPES Dehradun**
SDE @ SapMen C.

[![GitHub](https://img.shields.io/badge/GitHub-Nitanshu715-181717?style=flat-square&logo=github)](https://github.com/Nitanshu715)

### Swati Dubey
**BCA Computer Science and Management · Dr. Bhim Rao Ambedkar University, Agra**


[![GitHub](https://img.shields.io/badge/GitHub-swaskiee-181717?style=flat-square&logo=github)](https://github.com/swaskiee)

---

## Hackathon Context

**India Runs · Hack2skill × Redrob AI**
Track 1 — The Data & AI Challenge: Intelligent Candidate Discovery

Built specifically against Redrob's own stated goal for this role: *"we'd rather see 10 great matches than 1000 maybes."*

---

## License

MIT License — built for the India Runs hackathon; open for review, reproduction, and evaluation by Redrob AI and Hack2skill judges.

---

<div align="center">

**PRISM** · Built by Team Hyperion · 2026
*Ranking that reads the resume, not just the keywords.*

`~100-170s / 100K candidates` · `~3GB peak` · `0% honeypot rate` · `₹0 inference cost`

</div>
