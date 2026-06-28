# Layer 5 — Development Notes (honeypot_dev_notes.md)

This documents the actual investigation process for finding honeypot
patterns in the real dataset, including patterns that were checked and
explicitly REJECTED as honeypot signals. Kept for Stage 4/5 defense, same
spirit as Layer 3's disqualifiers.md.

## What the spec actually says (verbatim, submission_spec.docx Section 7)

> "The dataset contains a small number (~80) of honeypot candidates with
> subtly impossible profiles (e.g., 8 years of experience at a company
> founded 3 years ago; 'expert' proficiency in 10 skills with 0 years
> used)."

These are the ONLY two example mechanisms named in any official document.
No other document (README, job_description, redrob_signals_doc) adds
detail — the README's claim that redrob_signals_doc.md covers "trap
candidates and signal envelopes" turned out NOT to be accurate; that
document is purely about the 23 behavioral signals, with no honeypot
content.

## A real constraint: there is no company-founding-year field

`candidate_schema.json` has no field anywhere for when a company was
founded, and no separate companies-reference table is provided in the
bundle. This means the "8 years at a company founded 3 years ago" example
cannot be checked literally via an external lookup. Investigation of the
real data found how the synthetic generator actually implemented this
concept instead — see Pattern 1 below.

## Pattern 1 — Duration/date mismatch on current roles (CONFIRMED, used)

**Check:** for each `career_history` entry with `is_current: true`,
compute the actual elapsed months between `start_date` and today, and
compare against the stated `duration_months`. Flag if they differ by more
than 6 months.

**Found:** 33 entries across 33 distinct candidates. Example:
`CAND_0007353`'s current "Frontend Engineer" role started 2023-09-10
(~33 months before the dataset's reference point) but states
`duration_months=166` (~13.8 years) — directly analogous to the JD's
"8 years at a 3-year-old company" example, expressed as an internal
date/duration contradiction rather than an external company-age check.

**Confirmed exclusively on current roles:** checked whether the same
mismatch pattern ever appears on closed-out (`is_current: false`) entries
— zero occurrences. This supports the conclusion that this is a
deliberately injected anomaly on a specific field, not generic noise.

**Honest finding — bidirectional:** 19 of the 33 entries overstate
duration (the JD's framing), 14 understate it. Both are kept as flags,
since both represent the same underlying data-integrity problem
(start_date and duration_months contradicting each other), and the
spec's framing ("subtly impossible profiles") is broader than just
experience-inflation specifically.

## Pattern 2 — Expert proficiency with zero duration (CONFIRMED, used)

**Check:** any skill with `proficiency == "expert"` and
`duration_months == 0`.

**Found:** 21 distinct candidates, each with 3-5 such skills
simultaneously (never just one) — e.g. `CAND_0016000` has 5 skills
(TypeScript, Go, Docker, Hadoop, Photoshop) all marked expert/0-months,
mixed in among other skills with realistic durations (16, 8, 9, 12
months) and typically near-zero endorsements too. This is the JD's
example almost verbatim.

**No overlap with Pattern 1:** confirmed 0 candidates trip both rules —
54 distinct candidates total from the two independent mechanisms.

## Patterns investigated and explicitly REJECTED

These were checked against the full 100K dataset and found to be far too
common to represent a deliberate ~80-candidate honeypot injection — they
are normal dataset noise, and using them as honeypot signals would cause
massive false-positive rates that could themselves push a clean
submission's "honeypot rate" over the disqualification threshold by
miscounting ordinary candidates as honeypots.

| Pattern checked | Candidates affected | Verdict |
|---|---|---|
| Career history start year precedes earliest education start year | 3,457 | Too common — normal (people work during/before formal schooling in this dataset's generation) |
| A skill's `duration_months` exceeds total `career_history` time | 3,392 | Too common — skill duration appears generated independently of career length |
| `years_of_experience` vs total `career_history` months, >18mo slack | 24 (subset of Pattern 1, not independent) | Folded into Pattern 1's date-based check instead, which is more precise |
| Duplicate skill names on one profile | 0 | Doesn't occur in this dataset at all |
| `company_size` tagged as tiny (1-10) with very long claimed tenure (8+ yrs) | 0 | Doesn't occur — company_size appears independently randomized per entry, not tied to a fixed company age |
| Education `end_year < start_year` | 0 | Doesn't occur in this dataset at all |

## Honest summary

| Rule | Fires on full 100K | Verdict |
|---|---|---|
| `honeypot_duration_mismatch_flag` | 33 | Confirmed real, matches JD's named mechanism |
| `honeypot_expert_zero_duration_flag` | 21 | Confirmed real, matches JD's named mechanism almost verbatim |
| **Combined (`honeypot_flag`)** | **54** | No overlap between the two rules |

54 is close to, but not exactly, the spec's approximate "~80" figure.
This gap is reported honestly rather than papered over by loosening
thresholds to manufacture more matches — doing so risks the opposite
failure mode (flagging real, legitimate candidates as honeypots, which
would itself hurt ranking quality without any compensating benefit,
since the >10% threshold is about not ranking ACTUAL honeypots highly,
not about hitting an exact count). If a Stage 5 interviewer points out
the gap to ~80, the honest answer is: two precise, narrow, well-evidenced
mechanisms were found and confirmed against the real data; rather than
loosen thresholds speculatively to chase a round number, the team chose
to ship verified detectors and rely on Layer 2's semantic scoring plus
human review of the final top-100 (via `check_honeypot_rate_in_top_n`'s
returned `flagged_candidate_ids`) as a second line of defense against
any honeypot type not caught by these two rules.

## Recommendation to Nitanshu (Layer 4 / final assembly)

`honeypot_flag == True` should result in the candidate being excluded
from the top-100 entirely, or scored at 0 — not soft-penalized like a
Layer 3 disqualifier. Before final submission, run
`check_honeypot_rate_in_top_n()` on the actual top-100 ranking and:
1. Confirm `disqualification_risk` is `False`.
2. Manually pull and eyeball every `candidate_id` in
   `flagged_candidate_ids`, if any — don't trust the automated flag alone
   for a check this high-stakes.
