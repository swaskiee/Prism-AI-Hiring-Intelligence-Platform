"""
Layer 2 — Semantic Fit Scorer
Team Hyperion | India Runs Hackathon | Track 1: Data & AI Challenge
Owner: Nitanshu Tak

PURPOSE
-------
Score how well each candidate's actual career history MEANS what the JD
needs, independent of whether they used the exact same vocabulary. This is
the layer that catches the JD's own named "Tier 5" case:

    "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their
    profile, but if their career history shows they built a recommendation
    system at a product company, they're a fit."

Layer 3 (Swati's structural disqualifier pass) catches the OPPOSITE trap
(buzzword-heavy skills, irrelevant title). This layer and Layer 3 are
deliberately complementary, not redundant — see disqualifiers.py's own
docstring: "Semantic similarity alone ranks keyword-stuffed profiles
highly. This layer is what catches them" (Layer 3 talking about itself,
relative to this layer).

WHY TF-IDF + TRUNCATED SVD (LSA), NOT A SENTENCE-TRANSFORMER MODEL
--------------------------------------------------------------------
The original architecture plan called for a local sentence-transformer
(all-MiniLM-L6-v2). That was changed deliberately, for two concrete,
defensible reasons — documented here because Stage 5 will ask "why this
embedding approach":

1. DEPENDENCY RISK INSIDE REDROB'S SANDBOX. Our actual ranking script
   has to run inside Redrob's Stage-3 reproduction container — CPU-only,
   16GB RAM, NO network access. A sentence-transformer model requires
   PyTorch (typically 500MB-1GB+ as a dependency) and, on a clean machine,
   either bundles its weights or needs a one-time download — which is a
   network call, exactly what's banned at ranking time. Pre-bundling model
   weights into the repo avoids the network call but adds real size and a
   real new failure surface (torch import failures, version mismatches)
   we don't control inside an unfamiliar sandboxed environment. TF-IDF +
   SVD has zero such risk: it's pure scikit-learn (already a near-certain
   dependency for any ML hackathon submission), trains in seconds directly
   on this candidate pool, and has no binary/GPU-library surface to break.

2. THIS IS A CLASSICAL, WELL-UNDERSTOOD, FULLY DEFENSIBLE TECHNIQUE.
   TF-IDF + Truncated SVD is exactly Latent Semantic Analysis (LSA) — a
   technique with decades of IR literature behind it, directly relevant to
   the JD's own "ranking evaluation framework experience" and "hybrid
   search" must-haves. It is not a downgrade dressed up as a tradeoff: for
   a single, fixed, known-vocabulary JD compared against ~100K candidate
   documents that share a lot of overlapping technical/domain language
   (this dataset's candidate pool, not open-domain web text), a corpus-fit
   LSA space captures the JD-specific semantic structure (which skills
   co-occur with which career-history language) at least as well as a
   general-purpose pretrained sentence embedding would, while being fully
   transparent: every dimension is traceable back to actual vocabulary in
   THIS dataset, which makes Stage 5 defense far more concrete than "the
   transformer's attention weights decided this."

This tradeoff is explicit, not hidden — if Redrob's sandbox is later
confirmed to have a working, GPU-free, pre-cached sentence-transformers
environment, swapping the encode_documents()/encode_query() functions below
for a sentence-transformer call is a contained change (see "SWAP-IN POINT"
comment below) that does not require touching Layers 3, 4, 5, or the
fusion/ranking logic, since the output contract (a single float
similarity score per candidate) stays identical either way.

INPUT
-----
- candidates: list of candidate dicts (candidate_schema.json shape).
- jd_requirements: the structured object from Layer 1
  (jd_requirements.get_jd_requirements()).

OUTPUT
------
A pandas DataFrame, one row per candidate_id, with columns:

    candidate_id           str
    semantic_fit_score     float, 0.0-1.0 (cosine similarity in LSA space,
                            clipped to [0, 1] — raw cosine on SVD components
                            can occasionally be slightly negative for very
                            poor matches; clipping keeps the score a clean
                            "fit" scale for fusion in Layer 4)
    matched_requirements   str, comma-joined must_have/nice_to_have codes
                            whose individual requirement-text similarity to
                            this candidate exceeded MATCH_THRESHOLD — used
                            directly by the Explainability Engine so
                            reasoning text can name SPECIFIC matched
                            requirements instead of just citing one overall
                            number.

PERFORMANCE
-----------
Fit once (TF-IDF vectorizer + SVD) on the full candidate corpus + JD
requirement texts combined, then transform everything in one vectorized
batch. See benchmark.py for measured runtime/memory on the full 100,000
candidates — this is the most compute-heavy of the five layers, so it gets
the lion's share of the shared 300s/16GB budget (Swati's three layers
together measured at ~11s / ~0.07GB; see README_swati_layers.md).
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Number of latent semantic dimensions for the SVD step. Chosen as a
# moderate value appropriate for a vocabulary built from ~100K short
# technical documents (career-history descriptions + summaries) — large
# enough to separate distinct technical domains (NLP/retrieval vs CV vs
# data engineering vs frontend, etc.), small enough to stay fast and avoid
# overfitting to noise in any single candidate's phrasing.
N_SVD_COMPONENTS = 120

# Per-requirement similarity threshold above which a requirement is
# considered "matched" for explainability purposes. Tuned conservatively
# (see benchmark/dev notes) so matched_requirements lists stay precise
# rather than padding every candidate with marginal matches.
MATCH_THRESHOLD = 0.18


def _candidate_document(candidate: dict) -> str:
    """
    Build the single text document representing a candidate's experience,
    for embedding. Concatenates the profile summary/headline with every
    career_history entry's title + description — this is deliberately the
    SAME set of fields Layer 3's rule detectors read for their own
    description-text checks (see disqualifiers.py's
    detect_cv_speech_robotics_without_nlp), so Layers 2 and 3 are looking
    at consistent source text, not two different views of the candidate.
    """
    profile = candidate["profile"]
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for entry in candidate.get("career_history", []):
        parts.append(entry.get("title", ""))
        parts.append(entry.get("description", ""))
    # Skills are deliberately EXCLUDED from this document. Including raw
    # skill names here would let a keyword-stuffed skills list (the exact
    # trap this dataset plants — see Layer 3's title_mismatch_flag dev
    # notes) directly inflate semantic similarity even when career history
    # contradicts it. Semantic fit is scored on narrative substance
    # (what the candidate actually DID), not on what they listed.
    return " ".join(p for p in parts if p)


def _requirement_documents(jd_requirements: dict) -> Dict[str, str]:
    """
    Build one text document per must_have/nice_to_have requirement code,
    using Layer 1's *_detail dicts (the full descriptive sentence per
    requirement) rather than the short code alone — richer text gives the
    TF-IDF/SVD space more signal to work with than a 3-word code would.
    """
    docs: Dict[str, str] = {}
    docs.update(jd_requirements.get("must_have_detail", {}))
    docs.update(jd_requirements.get("nice_to_have_detail", {}))
    return docs


def _fit_lsa_space(documents: List[str]) -> tuple[TfidfVectorizer, TruncatedSVD]:
    """
    Fit the TF-IDF vectorizer and TruncatedSVD jointly on the full set of
    documents (all candidate documents + all requirement documents
    together), so candidate and requirement vectors land in the SAME
    latent space and are directly comparable via cosine similarity.

    SWAP-IN POINT: if a sentence-transformer model becomes the preferred
    approach later (see module docstring), this function and
    `_transform_lsa` below are the only two functions that need replacing
    — everything downstream (matching, scoring, output contract) stays
    identical, since both approaches reduce to "given a document, return a
    fixed-length vector usable with cosine similarity."
    """
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    n_components = min(N_SVD_COMPONENTS, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(tfidf_matrix)

    return vectorizer, svd


def _transform_lsa(texts: List[str], vectorizer: TfidfVectorizer, svd: TruncatedSVD) -> np.ndarray:
    """Transform raw text into L2-normalized LSA vectors for cosine similarity via dot product."""
    tfidf = vectorizer.transform(texts)
    lsa = svd.transform(tfidf)
    return normalize(lsa, norm="l2", axis=1)


def run_semantic_scoring(candidates: List[dict], jd_requirements: dict) -> pd.DataFrame:
    """
    Run Layer 2 across `candidates` against the JD requirements from
    Layer 1, returning the per-candidate semantic fit table described in
    the module docstring.

    Parameters
    ----------
    candidates : list of dict
        Candidate records matching candidate_schema.json.
    jd_requirements : dict
        Output of layer1.jd_requirements.get_jd_requirements().

    Returns
    -------
    pd.DataFrame
        Columns: candidate_id, semantic_fit_score, matched_requirements.
    """
    candidate_docs = [_candidate_document(c) for c in candidates]
    requirement_docs_map = _requirement_documents(jd_requirements)
    requirement_codes = list(requirement_docs_map.keys())
    requirement_docs = [requirement_docs_map[code] for code in requirement_codes]

    # Fit jointly so candidate and requirement vectors share one space.
    vectorizer, svd = _fit_lsa_space(candidate_docs + requirement_docs)

    candidate_vecs = _transform_lsa(candidate_docs, vectorizer, svd)
    requirement_vecs = _transform_lsa(requirement_docs, vectorizer, svd)

    # Cosine similarity of every candidate against every requirement, in
    # one matrix multiply (vectors are already L2-normalized, so dot
    # product IS cosine similarity) — this is the vectorized batch
    # operation the module docstring's PERFORMANCE section refers to.
    sim_matrix = candidate_vecs @ requirement_vecs.T  # shape: (n_candidates, n_requirements)

    # must_have requirements weighted higher than nice_to_have, reflecting
    # the JD's own framing — a candidate's overall semantic fit should be
    # driven primarily by the must-haves, with nice-to-haves as a smaller
    # boost, not weighted equally.
    must_have_codes = set(jd_requirements.get("must_have", []))
    weights = np.array([
        1.0 if code in must_have_codes else 0.4
        for code in requirement_codes
    ])
    weights = weights / weights.sum()

    raw_scores = sim_matrix @ weights
    semantic_fit_score = np.clip(raw_scores, 0.0, 1.0)

    matched_requirements = []
    for row in sim_matrix:
        matched = [
            requirement_codes[i] for i, score in enumerate(row)
            if score >= MATCH_THRESHOLD
        ]
        matched_requirements.append(",".join(matched))

    return pd.DataFrame({
        "candidate_id": [c["candidate_id"] for c in candidates],
        "semantic_fit_score": np.round(semantic_fit_score, 4),
        "matched_requirements": matched_requirements,
    })


if __name__ == "__main__":
    import json
    import time
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "layer1"))
    from jd_requirements import get_jd_requirements  # noqa: E402

    sample_path = os.path.join(os.path.dirname(__file__), "sample_candidates.json")
    with open(sample_path) as f:
        sample = json.load(f)

    jd_reqs = get_jd_requirements()

    start = time.perf_counter()
    result = run_semantic_scoring(sample, jd_requirements=jd_reqs)
    elapsed = time.perf_counter() - start

    print(f"Ran Layer 2 on {len(sample)} sample candidates in {elapsed:.4f}s\n")
    print(result.sort_values("semantic_fit_score", ascending=False).to_string(index=False))
