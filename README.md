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
  <img src="https://img.shields.io/badge/Sentence--Transformers-FF6F00?style=for-the-badge">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
</p>

<p>
  <img src="https://img.shields.io/badge/Compute-CPU%20Only-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Network-Disabled%20at%20Inference-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Runtime-%3C5min%2F100K%20candidates-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Cost-₹0-brightgreen?style=for-the-badge">
</p>

### 🧠 Hybrid Ranking &nbsp;•&nbsp; 🛡️ Honeypot Defense &nbsp;•&nbsp; 🔍 Explainable Scoring &nbsp;•&nbsp; ⚡ Zero-LLM Inference

</br>
</div>

**An Explainable, Trap-Resistant Candidate Ranking Engine**

*Built for Redrob AI's Intelligent Candidate Discovery & Ranking Challenge*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-MiniLM--L6-FF6F00?style=flat-square)](https://www.sbert.net/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-97BC62?style=flat-square)](LICENSE)

---

*Built for the **India Runs Hackathon** by Hack2skill × Redrob AI* *Track 1 — The Data & AI Challenge: Intelligent Candidate Discovery*
*Submission window closes **2 July 2026***

---

### **Team Hyperion**
* **Nitanshu Tak** — Semantic Scoring, Behavioral Signal Weighting, Score Fusion, Explainability Engine, System Architecture
* **Swati Dubey** — JD Requirement Extraction, Structural Disqualifier Engine, Honeypot & Anomaly Detection

</div>

---

## The Problem

Redrob AI's recruiters search a pool of hundreds of thousands of candidate profiles for every open role. Traditional keyword-based filtering looks for surface matches — does the skills list contain the right buzzwords — and as a result it systematically fails in two opposite directions at once:

**It misses good candidates.** A candidate who built a production recommendation system at a product company, but whose profile never happens to use the word "RAG" or "Pinecone," gets filtered out by a keyword scan even though their actual engineering experience is a strong match.

**It promotes bad candidates.** A candidate whose skills section lists every AI buzzword in existence, but whose actual job title is "Marketing Manager" and who has never shipped a production system, scores artificially high on keyword density while being a fundamentally wrong fit.

Redrob's own job description for this exact role states it directly: *the right answer is not "find candidates whose skills section contains the most AI keywords."* That sentence is the design brief for this entire project.

**Prism is built to close that gap** — to rank candidates the way an experienced technical recruiter actually would: by reading career history for substance, checking behavioral signals for real availability, and refusing to be fooled by keyword density in either direction.

---

## What Prism Does

Prism is a five-layer hybrid ranking engine that takes Redrob's 100,000-candidate pool and a single job description, and produces a ranked shortlist of the top 100 best-fit candidates — each with a specific, evidence-based explanation for its rank.

**It reads the JD, not just the keywords in it.** The job description is decomposed into structured must-haves, nice-to-haves, hard disqualifiers, and an explicit "ideal candidate" profile — capturing not just what skills are named, but what Redrob said they explicitly do *not* want (title-chasers, framework tourists, pure-research backgrounds, consulting-only careers with no product experience).

**It scores meaning, not vocabulary.** A local sentence-transformer embedding model compares the *semantics* of a candidate's career history against the JD's actual requirements — so a candidate who clearly did the work without using the exact trendy terminology still scores correctly.

**It applies the same structural judgment a senior recruiter would.** A rule-based disqualifier layer independently checks title sanity, consulting-only career patterns (with the product-company exception Redrob explicitly carved out), job-hopping/title-chasing patterns, and industry mismatches — catching exactly the keyword-stuffer trap that pure semantic similarity cannot.

**It treats "available" as part of "qualified."** Behavioral signals — recruiter response rate, login recency, interview completion rate, open-to-work status — are applied as a multiplier on top of skill-fit, because a perfect-on-paper candidate who has gone silent for six months is not, for hiring purposes, actually a top candidate.

**It refuses to be fooled by impossible data.** An anomaly-detection layer scans every profile for internal contradictions — "expert" proficiency claimed with near-zero time-in-skill, experience duration exceeding company age — and hard-excludes these honeypot profiles before they can ever reach the top 100, regardless of how well they'd otherwise score.

**Every rank comes with a reason, generated from real numbers.** No templated language, no boilerplate — each of the top 100 entries gets a 1-2 sentence justification built directly from that candidate's actual computed sub-scores, named skills, and signal values.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│                                                                        │
│   candidates.jsonl (100,000 profiles)      job_description.md        │
│   23 redrob_signals per candidate          Senior AI Engineer role   │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  LAYER 1 — JD REQUIREMENT EXTRACTION                  │
│                          Swati Dubey                                   │
│                                                                        │
│   Unstructured JD text  →  structured requirement object              │
│   must_have[] · nice_to_have[] · hard_disqualifiers[]                 │
│   experience_band · location_preference · ideal_profile_notes         │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
┌─────────────────────────────┐ ┌─────────────────────────────────────┐
│   LAYER 2 — SEMANTIC FIT     │ │  LAYER 3 — STRUCTURAL DISQUALIFIER   │
│        Nitanshu Tak           │ │            Swati Dubey                │
│                               │ │                                       │
│  sentence-transformers        │ │  Title sanity check                  │
│  (all-MiniLM-L6-v2, local)   │ │  Consulting-only career detector     │
│  JD requirements ↔ candidate │ │  Title-chaser / job-hop detector     │
│  career_history embeddings    │ │  Stale-architect detector            │
│  Cosine similarity score      │ │  CV/speech/robotics-without-NLP flag │
└──────────────┬────────────────┘ └───────────────────┬───────────────────┘
               │                                       │
               ▼                                       ▼
┌─────────────────────────────┐ ┌─────────────────────────────────────┐
│  LAYER 4 — BEHAVIORAL TRUST  │ │   LAYER 5 — HONEYPOT & ANOMALY       │
│       MULTIPLIER              │ │            DETECTION                  │
│        Nitanshu Tak           │ │            Swati Dubey                │
│                               │ │                                       │
│  recruiter_response_rate      │ │  Impossible-tenure detection         │
│  last_active_date recency     │ │  Impossible-proficiency detection    │
│  interview_completion_rate    │ │  Internal date/duration contradiction │
│  open_to_work_flag            │ │  Hard exclusion gate (not a penalty) │
└──────────────┬────────────────┘ └───────────────────┬───────────────────┘
               │                                       │
               └───────────────────┬───────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    SCORE FUSION & RANKING                            │
│                          Nitanshu Tak                                   │
│                                                                        │
│   final_score = f(semantic_fit, disqualifier_penalty,                │
│                    trust_multiplier, honeypot_gate)                  │
│   Honeypots hard-excluded → remaining candidates sorted → top 100    │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    EXPLAINABILITY ENGINE                              │
│                          Nitanshu Tak                                   │
│                                                                        │
│   Per-candidate reasoning generated from real computed sub-scores    │
│   No templates · no hallucinated claims · grounded in actual profile │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
                  submission.csv (top 100, ranked)
```

> **Note on diagram accuracy:** this reflects the finalized design. As implementation progresses, this diagram will be updated if any layer's internals change — see [Project Status](#project-status) below for what's built vs. designed.

---

## Why This Architecture, Specifically

This is not the "biggest" system we could have built — it is deliberately shaped by Redrob's own constraints, and every design choice below maps to a specific line in their challenge spec.

**No hosted LLM calls during ranking.** The compute budget is 5 minutes wall-clock, 16 GB RAM, CPU-only, no network, for the full 100,000-candidate pool. An LLM-per-candidate pipeline cannot fit this budget — Redrob says so explicitly, because a system that can't run at this scale in production isn't a real solution to their actual problem. Every layer in Prism runs locally on CPU.

**Rules alongside embeddings, not instead of them.** Semantic similarity alone is exactly the failure mode Redrob warns about — it would still rank a "Marketing Manager" with a buzzword-heavy skills list highly, because the words *are* semantically close to the JD. The structural disqualifier layer (Layer 3) exists specifically to catch what embedding similarity cannot.

**Behavioral signals as a multiplier, not an afterthought.** Redrob's own documentation states that a perfect-on-paper candidate with a 5% recruiter response rate and six months of inactivity is, for hiring purposes, not actually available. Layer 4 encodes this directly rather than treating it as a minor tiebreaker.

**A hard honeypot gate, not a soft penalty.** Submissions with a honeypot rate above 10% in the top 100 are disqualified outright, regardless of ranking quality elsewhere. Layer 5 is built as a hard exclusion step for exactly this reason — there is no acceptable tradeoff here.

**Reasoning generated from real sub-scores, not from a second LLM pass.** Stage 4 of the evaluation manually samples reasoning text and checks it against the candidate's actual profile for hallucination and rank-consistency. Generating reasoning straight from the same numbers that produced the rank is the only way to guarantee the explanation can't drift from the decision.

---

## Project Status

> **Honesty note:** This README documents the full system design Team Hyperion has committed to. Sections below are marked according to actual build status as of each update — this project is being built in public, layer by layer, with real iteration history in the commit log.

| Layer | Status | Owner |
|---|---|---|
| Layer 1 — JD Requirement Extraction | `[ FILL IN: Not Started / In Progress / Complete ]` | Swati Dubey |
| Layer 2 — Semantic Fit Scorer | `[ FILL IN: Not Started / In Progress / Complete ]` | Nitanshu Tak |
| Layer 3 — Structural Disqualifier Engine | `[ FILL IN: Not Started / In Progress / Complete ]` | Swati Dubey |
| Layer 4 — Behavioral Trust Multiplier | `[ FILL IN: Not Started / In Progress / Complete ]` | Nitanshu Tak |
| Layer 5 — Honeypot & Anomaly Detection | `[ FILL IN: Not Started / In Progress / Complete ]` | Swati Dubey |
| Score Fusion & Ranking | `[ FILL IN: Not Started / In Progress / Complete ]` | Nitanshu Tak |
| Explainability Engine | `[ FILL IN: Not Started / In Progress / Complete ]` | Nitanshu Tak |
| Sandbox Deployment | `[ FILL IN: Not Started / In Progress / Complete ]` | Nitanshu Tak |

---

## Results

> **`[ FILL IN — do not publish this section with placeholder numbers ]`**
> This section must only contain numbers actually produced by running the pipeline end-to-end on the real `candidates.jsonl`. Suggested structure once real runs exist:

| Metric | Value | How Measured |
|---|---|---|
| `[ FILL IN: total runtime on full 100K pool ]` | `[ FILL IN ]` | `time python rank.py --candidates ./candidates.jsonl --out ./submission.csv` |
| `[ FILL IN: peak memory usage ]` | `[ FILL IN ]` | `[ FILL IN: profiling tool used, e.g. memory_profiler ]` |
| `[ FILL IN: honeypot rate in top 100 ]` | `[ FILL IN ]` | Manual cross-check against suspected honeypot patterns |
| `[ FILL IN: disqualifier hits in candidate pool ]` | `[ FILL IN ]` | Count of candidates flagged by Layer 3 across full pool |

**Sample reasoning output** — `[ FILL IN with 2–3 real generated reasoning strings from an actual top-100 run once available, e.g.: ]`

```
CAND_00XXXXX, rank 1: "[ Real generated reasoning string here ]"
CAND_00XXXXX, rank 47: "[ Real generated reasoning string here ]"
```

---

## Repository Structure

> **`[ FILL IN once the repo layout is finalized — update this tree to match the actual repo exactly before submission, since the Stage 3 reviewers will compare this against the real structure ]`**

```
Prism-AI-Hiring-Intelligence-Platform/
│
├── data/
│   ├── candidates.jsonl              [ FILL IN: gitignored or sample-only? ]
│   ├── candidate_schema.json
│   └── job_description.md
│
├── src/
│   ├── jd_extraction.py              Layer 1 — Swati
│   ├── semantic_scorer.py            Layer 2 — Nitanshu
│   ├── disqualifier_engine.py        Layer 3 — Swati
│   ├── trust_multiplier.py           Layer 4 — Nitanshu
│   ├── honeypot_detector.py          Layer 5 — Swati
│   ├── score_fusion.py               Score combination — Nitanshu
│   ├── reasoning_generator.py        Explainability — Nitanshu
│   └── rank.py                       Single entrypoint script
│
├── sandbox/
│   └── [ FILL IN: HuggingFace Space / Streamlit app / notebook, per chosen platform ]
│
├── tests/
│   └── [ FILL IN once test suite exists ]
│
├── submission.csv                    Final top-100 ranked output
├── submission_metadata.yaml          Per official template
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.11 | Required by spec; mature ML/NLP ecosystem |
| Semantic Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Small, fast, fully local — no GPU, no network call, no API key |
| Data Processing | Pandas / NumPy | Vectorized operations across 100,000 records within the 5-minute budget |
| Rule Engine | Python (`src/disqualifier_engine.py`, `src/honeypot_detector.py`) | Deterministic, explainable, auditable in the Stage 5 interview |
| Score Fusion | `[ FILL IN: e.g. weighted linear combination, documented formula ]` | `[ FILL IN: rationale once finalized ]` |
| Output Format | CSV (UTF-8) | Per official submission spec |
| Sandbox Host | `[ FILL IN: HuggingFace Spaces / Streamlit Cloud / Replit / Colab / Docker ]` | `[ FILL IN once chosen ]` |
| **Total Cost** | **₹0** | Entirely local/open-source models, no paid API usage during ranking |

---

## Compute Constraints (Hard Requirements)

Every constraint below comes directly from Redrob's official `submission_spec.md` and is treated as non-negotiable, not aspirational:

| Constraint | Limit | Status |
|---|---|---|
| Total runtime | ≤ 5 minutes wall-clock | `[ FILL IN: measured value ]` |
| Memory | ≤ 16 GB RAM | `[ FILL IN: measured value ]` |
| Compute | CPU only — no GPU | ✅ By design (no GPU dependency anywhere in the pipeline) |
| Network | Disabled during ranking — no OpenAI/Anthropic/Cohere/Gemini/hosted LLM calls | ✅ By design (all models run locally) |
| Disk | ≤ 5 GB intermediate state | `[ FILL IN: measured value ]` |

---

## Local Setup

> **`[ FILL IN once Layer 1 and the entrypoint script exist — placeholder structure below should be updated to match real commands exactly ]`**

```bash
git clone https://github.com/swaskiee/Prism-AI-Hiring-Intelligence-Platform.git
cd Prism-AI-Hiring-Intelligence-Platform

python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Running the Full Ranking Pipeline

```bash
python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

`[ FILL IN: expected console output once the pipeline runs end-to-end, e.g. progress indicators per layer, final runtime printout ]`

### Validating the Submission

```bash
python validate_submission.py submission.csv
```

---

## Methodology Deep Dive

### Layer 1 — JD Requirement Extraction

The job description for Redrob's Senior AI Engineer role is decomposed into a structured object rather than used as free text. This captures:

- **Must-have requirements**: production embeddings-based retrieval experience, production vector DB/hybrid search experience, strong Python, ranking-evaluation framework experience.
- **Nice-to-have signals**: LLM fine-tuning, learning-to-rank models, HR-tech background, distributed systems experience, open-source contributions.
- **Hard disqualifiers**: pure-research-only background, recent-only LLM-wrapper experience with no pre-LLM production history, stale architects with 18+ months away from hands-on code, title-chasing career patterns, framework-tutorial-only GitHub profiles, consulting-only careers with no product-company experience (with Redrob's own explicit exception for current consulting employees with prior product experience), computer-vision/speech/robotics specialists without NLP exposure, and closed-source-only careers with no external validation.
- **The "ideal candidate" rubric**: 6–8 years total experience with 4–5 years in applied ML/AI at product companies, has shipped an end-to-end ranking/search/recommendation system at meaningful scale, holds defensible technical opinions on retrieval and evaluation strategy, located in or willing to relocate to Pune/Noida, and shows active engagement signal on the Redrob platform.

`[ FILL IN: link to the actual extraction script output / JSON artifact once built ]`

### Layer 2 — Semantic Fit Scoring

`[ FILL IN: final embedding strategy once implemented — e.g. which candidate fields are concatenated before embedding, how JD requirement embeddings are aggregated, exact similarity formula used ]`

### Layer 3 — Structural Disqualifier Engine

Five independent rule-based checks run across the full candidate pool, each producing a flag and a documented rationale:

1. **Title sanity check** — fuzzy-matches `current_title` and historical titles against engineering/ML role patterns, catching candidates whose actual function is unrelated to the role despite a buzzword-heavy skill list.
2. **Consulting-only career detector** — flags candidates whose entire career history sits within named consulting firms, with the explicit carve-out for current consulting employees who have prior product-company experience elsewhere in their history.
3. **Title-chaser / job-hop detector** — looks for a sustained pattern of ~18-month tenures paired with consistently escalating seniority titles across most of a candidate's career history, not a single short stint.
4. **Stale-architect detector** — flags long stretches in pure architecture/leadership titles with no recent hands-on technical signal.
5. **CV/speech/robotics-without-NLP detector** — flags candidates whose domain focus is computer vision, speech, or robotics with no NLP/IR/retrieval exposure anywhere in their history.

`[ FILL IN: actual flag rates across the 100K pool once run, e.g. "X% of candidates flagged by at least one rule" ]`

### Layer 4 — Behavioral Trust Multiplier

`[ FILL IN: exact multiplier formula once finalized — which of the 23 redrob_signals fields are used, how they're weighted and combined, and the reasoning for that weighting ]`

### Layer 5 — Honeypot & Anomaly Detection

This layer treats internal data contradictions as detectable, not just "suspicious-feeling." Specifically checked:

- **Impossible tenure** — experience or company tenure that is mathematically inconsistent with the candidate's own stated dates/durations elsewhere in their record.
- **Impossible proficiency** — skills listed as "expert" with near-zero `duration_months`, or an implausible number of "expert" skills relative to total years of experience.

Detected honeypots are **hard-excluded** before the top-100 cut is made — not down-weighted — per Redrob's explicit disqualification rule for honeypot rates above 10%.

`[ FILL IN: actual detection count and any manually-verified false-positive checking once run on the real dataset ]`

### Score Fusion

`[ FILL IN: exact final formula combining semantic_fit_score, disqualifier_penalty, trust_multiplier, with honeypot_flag as a hard pre-filter — document the formula here once locked, since this is the first thing the Stage 5 interview will ask about ]`

### Explainability Engine

Each of the top 100 candidates receives a reasoning string built directly from their own computed values — years of experience, the specific JD requirement that matched, named skills, and relevant signal values such as notice period or recruiter response rate. No reasoning is templated, and no claim is generated that doesn't trace back to an actual field in that candidate's record.

---

## Evaluation Awareness

This project is built with Redrob's full evaluation pipeline in mind, not just the final score:

| Stage | What It Checks | How Prism Addresses It |
|---|---|---|
| 1. Format validation | CSV structure, exactly 100 rows, valid candidate IDs | Validated locally with `validate_submission.py` before every submission |
| 2. Scoring | NDCG@10 / NDCG@50 / MAP / P@10 against hidden ground truth | Architecture optimized for precision at the top of the ranking, where weight is highest |
| 3. Code reproduction + honeypot check | Runs ranking step in a sandboxed 5-min/16GB/CPU/no-network container; honeypot rate >10% disqualifies | Entire pipeline designed within these limits from day one; Layer 5 is a hard gate, not a soft penalty |
| 4. Manual review | Reasoning quality, hallucination checks, rank-consistency, git history authenticity | Explainability Engine grounds every reasoning string in real data; commits made incrementally through real development, not as a single dump |
| 5. Defend-your-work interview | Live defense of architecture and design choices | Every layer's design choice is documented here with its rationale, for exactly this conversation |

---

## Dataset

Candidate pool and job description provided directly by Redrob AI / Hack2skill as part of the official India Runs hackathon bundle (100,000 synthetic-but-realistic candidate profiles, 23 behavioral signal fields per candidate, ~80 deliberately embedded honeypot profiles). Not redistributed in this repository in full — see `data/` for schema and sample references only.

`[ FILL IN: confirm final decision on whether the full candidates.jsonl is committed, gitignored, or fetched via script, and document that decision here ]`

---

## Team

### Nitanshu Tak
**B.Tech CSE (Cloud Computing & Virtualization Technology) · UPES Dehradun**
SDE @ SapMen C. · Founder, MediFlow AI

*Contributions: Semantic scoring engine · behavioral signal weighting · score fusion logic · explainability engine · system architecture · sandbox deployment*

[![GitHub](https://img.shields.io/badge/GitHub-Nitanshu715-181717?style=flat-square&logo=github)](https://github.com/Nitanshu715)

---

### Swati Dubey

*Contributions: JD requirement extraction · structural disqualifier engine · honeypot and anomaly detection · feature engineering across the full candidate pool*

---

## Hackathon Context

**India Runs · Hack2skill × Redrob AI**
Track 1 — The Data & AI Challenge: Intelligent Candidate Discovery
Submission deadline: **2 July 2026**

Built specifically against Redrob's own stated goal for this role: *"we'd rather see 10 great matches than 1000 maybes."*

---

## License

MIT License — built for the India Runs hackathon; open for review, reproduction, and evaluation by Redrob AI and Hack2skill judges.

---

<div align="center">

**PRISM** · Built by Team Hyperion · 2026
*Ranking that reads the resume, not just the keywords.*

`Layer 5 Architecture` · `CPU-Only` · `₹0 Inference Cost`

</div>
