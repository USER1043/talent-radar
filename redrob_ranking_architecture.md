# Redrob Hackathon — Ranking System Architecture (v1.0, Final)

This document is the implementation spec. Hand it to a coding agent as-is; it
contains exact file layout, exact rules, exact schema field names, and a
build order. Anything marked `AGENT TODO` is a decision left open for
implementation (model checkpoint names, exact hyperparameters) because it
depends on what's actually installable/runs fast enough in the sandbox.

---

## 0. Grounded in

- `job_description.docx` — full JD text, explicit disqualifiers, "ideal
  candidate" narrative, explicit anti-keyword-stuffing warning.
- `submission_spec.docx` — CSV schema, compute limits, scoring formula,
  Stage 1–5 pipeline, reasoning-audit checks.
- `candidate_schema.json` — candidate record structure.
- `redrob_signals_doc.docx` — the 23 behavioral signal fields.
- `sample_submission.csv` — worked example of the keyword-stuffing trap
  (HR Manager / Content Writer ranked top with "AI core skills").
- `README.docx` — bundle contents, trap categories (keyword stuffers,
  plain-language Tier 5s, behavioral twins, ~80 honeypots).

**Not yet supplied** — needed before this can actually run:
`candidates.jsonl.gz` (100K pool), `validate_submission.py`,
`submission_metadata_template.yaml`. Get these before implementation starts.

---

## 1. Mission & what the scoring formula tells us to prioritize

```
composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
```

80% of the score is about the **top 10 and top 50 being in the right
order**, not just "relevant." Optimize precision-at-the-top over recall
across the full 100. A system that's excellent at rows 50–100 and mediocre
at rows 1–10 scores badly; the reverse scores well. This should bias every
tradeoff below (e.g. spend more compute budget on careful scoring of a
smaller shortlist rather than coarse scoring of a larger one).

---

## 2. Hard constraints (apply to the *ranking step* only)

| Constraint | Limit |
|---|---|
| Runtime | ≤ 5 min wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only, no GPU |
| Network | Off — no hosted LLM API calls |
| Disk (intermediate) | ≤ 5 GB |
| Honeypot rate in top 100 | ≤ 10% (else Stage 3 disqualification) |
| Submissions allowed | 3 max, last valid one counts |

Precompute (embeddings, indexes, trained ranker, eval labels) has **no**
time limit but must be scripted and reproducible, per spec §10.3.

---

## 3. Pipeline overview

```
OFFLINE (unlimited time, scripted)          ONLINE — rank.py (≤5 min, the graded run)
────────────────────────────────────        ──────────────────────────────────────────
1. parse_candidates.py                       1. Load cached artifacts (one-time)
   100K JSONL.gz → features.parquet          2. Hybrid retrieve shortlist (~4–5K)
2. company_stats.py                          3. Join precomputed features
   median first-seen date per company        4. Score: rules + behavioral + learned
3. honeypot_flags.py                         5. Hard-filter honeypots
   implausibility flags per candidate         6. Sort, take top 100, assign rank
4. build_index.py                            7. Generate reasoning (LLM + fact-check)
   sentence embeddings + FAISS + BM25         8. Write + self-validate CSV
5. eval_set.csv (hand-labeled, ~120 pairs)
6. train_ranker.py → ranker_model.pkl
   (LightGBM LambdaMART on the eval set)
```

---

## 4. Repository layout

```
repo/
├── README.md                      # single command to reproduce submission.csv
├── requirements.txt
├── submission_metadata.yaml
├── data/
│   └── job_description.json       # JD text + structured rule constants (§7)
├── precompute/
│   ├── parse_candidates.py
│   ├── company_stats.py
│   ├── honeypot_flags.py
│   ├── build_index.py
│   ├── label_eval_set.py          # helper to produce eval_set.csv
│   ├── train_ranker.py
│   └── artifacts/                 # output of the above — gitignored bulk,
│       ├── features.parquet            but checked-in small ones (model, jd embeds)
│       ├── company_stats.parquet
│       ├── honeypot_flags.parquet
│       ├── faiss.index
│       ├── bm25.pkl
│       ├── jd_embeddings.npy
│       ├── eval_set.csv
│       └── ranker_model.pkl
├── rank.py                         # THE graded entry point
├── reasoning.py                    # LLM generation + fact-check (imported by rank.py)
├── validate_local.py               # mirrors the official validator, run before submit
├── app/
│   └── streamlit_app.py            # sandbox demo, small-sample only
└── tests/
    └── test_no_stuffing_regression.py   # asserts output ≠ sample_submission.csv pattern
```

---

## 5. Candidate feature extraction (from `candidate_schema.json`)

One flattened row per candidate, computed once offline:

| Feature | Source | Notes |
|---|---|---|
| `years_of_experience` | `profile.years_of_experience` | |
| `current_title`, `current_company`, `current_industry` | `profile.*` | |
| `n_career_entries` | `len(career_history)` | |
| `total_tenure_months` | `sum(duration_months)` | |
| `n_employers` | `len(set(company))` | |
| `avg_tenure_months` | `total_tenure_months / n_employers` | job-hop signal |
| `tenure_consistency_flag` | per entry: `duration_months` vs `(end_date - start_date)` in months | mismatch = data-integrity red flag |
| `is_consulting_only` | every `career_history[].company` ∈ `{TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, ...}` | exact JD disqualifier |
| `has_product_company_experience` | any employer not in consulting list and not a pure research org | |
| `n_skills`, `skill_names` | `skills[]` | |
| `n_expert_or_advanced_skills` | `proficiency in {expert, advanced}` | |
| `n_zero_duration_high_proficiency_skills` | `proficiency in {expert, advanced}` AND `duration_months == 0` | **honeypot signal** |
| `skill_to_experience_ratio` | `n_skills / max(years_of_experience, 0.5)` | **honeypot signal** |
| `education_tier` | `education[].tier` | institution prestige — **not** the same as the hidden relevance tier; don't conflate |
| `redrob_*` | all 23 fields verbatim, see §8 | |
| `narrative_text` | concatenation of `headline + summary + all career_history[].description + skill names` | input to embeddings + BM25 |

---

## 6. Honeypot / trap detection

Three trap families exist per the README: **keyword stuffers**,
**plain-language Tier 5s** (legit, just no jargon), **behavioral twins**
(two similar profiles, one real one not), and **~80 honeypots** (the
disqualifying ones, "impossible profiles"). Only the last group counts
against the 10% threshold — but all four matter for ranking quality.

### 6.1 Honeypot rules (OR-logic, not averaging)

Any single rule firing is enough to flag — averaging dilutes a single
strong signal, which is the exact bypass to avoid:

```python
def is_honeypot(candidate, company_stats) -> tuple[bool, list[str]]:
    reasons = []

    # Rule 1 — skill claims unsupported by time spent
    if candidate.n_zero_duration_high_proficiency_skills >= 1:
        reasons.append("expert/advanced skill with 0 months duration")

    # Rule 2 — skill count wildly outpaces experience
    if candidate.skill_to_experience_ratio > THRESHOLD_A:   # AGENT TODO: tune from data distribution
        reasons.append("skill count implausible vs years of experience")

    # Rule 3 — tenure predates the company's existence
    for entry in candidate.career_history:
        stat = company_stats.get(entry.company)
        if stat is None:
            continue  # single-occurrence company — see 6.2, handled separately
        if stat.is_corroborated and entry.start_date < stat.median_first_seen - BUFFER_MONTHS:
            reasons.append(f"tenure at {entry.company} predates its inferred founding")

    # Rule 4 — overlapping full-time roles
    if has_overlapping_career_entries(candidate.career_history):
        reasons.append("overlapping employment dates")

    # Rule 5 — duration_months doesn't match declared date range
    if candidate.tenure_consistency_flag:
        reasons.append("duration_months inconsistent with start/end dates")

    return (len(reasons) > 0), reasons
```

### 6.2 The single-occurrence-company gap, fixed properly

A fabricated company that appears in **only one** profile has no
independent data to check against — its own lie becomes the only "evidence."
Two-tier handling:

- **Tier A — corroborated companies** (appear in ≥2 candidates across the
  100K pool): use the **median** `start_date` seen across all mentions as
  the inferred "earliest plausible" date. Median, not minimum — one bad
  data point can't drag it.
- **Tier B — single-occurrence companies**: company-timeline checking is
  *not possible*. Fall back to **standalone** plausibility rules that don't
  need cross-referencing: does claimed tenure exceed total plausible career
  length from education end_year? Does the role's seniority/title jump
  implausibly relative to total years of experience? These rules apply
  regardless of how many other candidates mention the company.

### 6.3 Action on flagged candidates

Honeypot-flagged candidates are **removed from the candidate pool entirely**
before top-100 selection (not just penalized) — this is the one place a
hard filter is justified, because the ground truth forces these to tier 0
and the submission-level penalty for missing this is a full disqualification,
not just a lower score.

---

## 7. JD rule layer — the explicit disqualifiers as features

The JD is structured as an explicit allow/deny list, not just descriptive
text. Encode each clause as a feature with a confidence-scaled penalty
(not a hard zero, except where the JD itself says "we will not move
forward" — those get the heavier penalty):

| JD clause | Strength | Feature | Penalty |
|---|---|---|---|
| Pure research/academic, no production deployment | **Hard** ("we will not move forward") | `is_pure_research` | severe (e.g. ×0.05) |
| <12mo "AI experience" = LangChain+OpenAI calls only, no pre-LLM ML history | Soft ("probably not") | `is_recent_wrapper_only` | strong (e.g. ×0.3) |
| Senior, but no production code in 18+ months (architecture/tech-lead only) | Soft | `no_recent_coding` | strong |
| Career entirely at named consulting firms | Soft, with carve-out if prior product-co experience exists | `is_consulting_only` | strong, unless `has_product_company_experience` |
| Computer vision/speech/robotics with no NLP/IR exposure | Soft (respectful framing, still a no) | `cv_speech_robotics_no_nlp` | strong |
| 5+ years entirely closed-source, no external validation (papers/talks/OSS) | Soft | `no_external_validation` | moderate |
| Title-chasing (Senior→Staff→Principal via company-hopping every ~1.5yr) | Soft, "culture fit" | `title_chasing_pattern` | mild — `avg_tenure_months < 18` across 2+ employers |
| "Framework enthusiast" (tutorial-style GitHub/blog, no systems thinking) | Soft, hard to detect from this schema | `framework_enthusiast_signal` | mild — **low confidence, see note below** |

**Implementation note:** most of these require lightweight NLP over free
text (`career_history[].description`, `profile.summary`, `profile.headline`)
— keyword/phrase cues (e.g. "research lab," "PhD thesis," "tutorial,"
"deployed to production," "fine-tuned," "served X requests/day") rather than
exact string matches. These will be noisy. Keep them as **soft, confidence-
weighted penalties** feeding into the learned ranker (§9) rather than hard
filters — a wrong hard filter here silently removes a genuinely strong
candidate, which is worse for NDCG@10 than a slightly noisy soft penalty.

### 7.1 Two JD-side representations for retrieval/scoring

Embed **both**, and let the learned ranker decide how to weight them:

1. The literal JD text.
2. The JD's own "ideal candidate" paragraph (6–8 yrs, 4–5 in applied
   ML/AI at product companies, shipped an end-to-end ranking/search/rec
   system, Noida/Pune-based or willing to relocate). This narrative form
   often matches *how a real candidate's profile is written* better than
   the bullet-point JD does — directly addresses the JD's own framing of
   "the gap between what the JD says and what it means."

---

## 8. Behavioral signal scoring (the 23 `redrob_signals`)

Per the signals doc, these matter because a perfect-on-paper but unreachable
candidate isn't actually hireable. Combine into one `behavioral_score`
(0–1), with sensible handling of the documented sentinel values:

```python
def behavioral_score(s):
    recency = recency_decay(s.last_active_date)             # exponential decay, ~90-day half-life
    responsiveness = s.recruiter_response_rate               # already 0-1
    speed = 1 / (1 + s.avg_response_time_hours / 24)          # faster = closer to 1
    reliability = s.interview_completion_rate                 # 0-1
    offer_signal = s.offer_acceptance_rate if s.offer_acceptance_rate >= 0 else NEUTRAL  # -1 sentinel = no history
    availability = 1.0 if s.open_to_work_flag else 0.6
    trust = 0.5*s.verified_email + 0.3*s.verified_phone + 0.2*s.linkedin_connected
    github = (s.github_activity_score / 100) if s.github_activity_score >= 0 else NEUTRAL  # -1 = no GitHub linked

    return weighted_sum([recency, responsiveness, speed, reliability,
                          offer_signal, availability, trust, github])
                          # AGENT TODO: exact weights come from LightGBM training (§9), this is the cold-start fallback
```

`notice_period_days` is JD-relevant directly (JD explicitly prefers
sub-30-day, will buy out up to 30, still considers 30+ but with a higher
bar) — encode as a smooth penalty curve, not a cliff.

---

## 9. The validation-signal fix (the architecture's actual fatal flaw, addressed)

Without labels, every weight above is a guess. Fix:

1. **Hand-label ~100–150 candidate–JD pairs** on a 0–5 relevance scale
   (`precompute/label_eval_set.py` just samples diverse candidates —
   including a few you've manually checked are honeypots, a few obvious
   keyword-stuffers, a few plain-language strong fits — and writes a CSV
   for a human to fill in `relevance` by hand). 30–60 min of human time.
2. **Train a LightGBM LambdaMART ranker** (`train_ranker.py`) on:
   `[semantic_sim_to_jd, semantic_sim_to_ideal_candidate, bm25_score,
   years_of_experience, structured JD-rule features from §7,
   behavioral_score and its components, honeypot-adjacent ratios]`
   → learns real relative weights instead of hand-picked ones.
3. Compute your own NDCG@10/NDCG@50/MAP/P@10 against this eval set before
   every submission — won't match the hidden grader exactly, but catches
   obviously broken ordering.
4. **Sanity-check assertions**: a small fixed set of "obviously good" vs
   "obviously bad" candidate pairs that must always order correctly —
   cheap regression tests for the ranker.
5. With only ~150 labels, LightGBM may overfit — fall back to a simpler
   logistic regression or even hand-tuned-but-eval-validated linear
   weights if LightGBM's cross-validated NDCG isn't clearly better.
   `AGENT TODO`: try both, keep whichever wins on held-out folds of the
   eval set.

This model is trained offline once; `rank.py` just loads `ranker_model.pkl`.

---

## 10. Online ranking step — `rank.py` (the graded ≤5 min run)

```
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```//(must also accept .jsonl.gz transparently — detect via magic bytes/extension)

Stages and rough time budget (tune once profiled — these are starting points):

| Stage | What | Budget |
|---|---|---|
| 1 | Load cached artifacts (parquet, FAISS, BM25, model, JD embeddings) | ~15s |
| 2 | Embed JD (or load cached, since JD is fixed) | <1s |
| 3 | Hybrid retrieve: FAISS top-K ∪ BM25 top-K → shortlist (~4–5K, not 1–2K — widened per the retrieval-ceiling fix) | ~5–10s |
| 4 | Join precomputed features for shortlist | ~5s |
| 5 | Compute JD-rule features (§7) + behavioral_score (§8) for shortlist, vectorized in pandas | ~10–20s |
| 6 | Score with `ranker_model.pkl` | ~5s |
| 7 | Drop honeypot-flagged candidates (§6) from the pool | ~1s |
| 8 | Sort, take top 100, assign rank 1–100, tie-break by `candidate_id` ascending on exact score ties | ~1s |
| 9 | Generate reasoning for the 100 finalists (§11) | **the long pole — budget 2–3 min** |
| 10 | Write CSV, run `validate_local.py` checks inline, exit | ~5s |

If stage 9 is running long, **degrade gracefully**: cap per-candidate
generation attempts, and for any candidate still ungenerated when the time
budget is nearly exhausted, fall back to a *shorter LLM call* with a
stripped-down prompt (still LLM-generated, never a hardcoded template —
the explicit hackathon rule is no templates, not "no LLM").

---

## 11. Reasoning generation (final 100 only — never the full 100K)

**Constraint reminder:** no hardcoded templates, no external API calls.
A small local quantized LLM, loaded once and reused for all 100 calls.

`AGENT TODO` — pick a concrete checkpoint that's actually small/fast enough
on CPU within budget: a 1–3B instruct model in GGUF format via
`llama-cpp-python`, Q4_K_M quantization. Verify wall-clock for 100 short
generations (~40 tokens each) fits comfortably inside the stage-9 budget
on the target machine before locking this in.

### 11.1 Prompt design (must satisfy the Stage 4 audit's 6 checks)

Feed the model **only verified facts** for that candidate, plus its rank
and its computed weak point, so the output is naturally calibrated:

```
Facts you may use (do not invent anything else):
- years_of_experience: {yoe}
- current_title: {title}, current_company: {company}
- matched_skills: {skills_that_overlap_with_jd}
- relevant_career_history: {1-2 most relevant past roles, verbatim company+title+short paraphrase}
- behavioral: last_active {recency}, response_rate {rate}, notice_period {days}d
- this candidate is ranked #{rank} of 100 for this role
- primary concern/gap relative to the JD: {weakest_matching_dimension}

Write 1-2 sentences a recruiter could read in 5 seconds. Reference at least
one specific fact above. If rank is below ~20, the tone should include the
concern, not just praise. Never mention a skill, company, or fact not listed
above.
```

The "primary concern" slot is what makes rank-tone consistency work
automatically — a rank-90 candidate's prompt always contains a real gap,
so the model has something honest to write about rather than improvising
generic praise.

### 11.2 Grounding / fact-check pass

After generation, verify the output doesn't hallucinate:

- Extract candidate noun phrases from the generated sentence.
- Check each against the candidate's **allowed facts** (the exact list fed
  into the prompt) using a small local **NLI model** (entailment check:
  does "candidate has X years at Y" follow from the allowed facts?) —
  catches paraphrases correctly (so "ML" matching "Machine Learning"
  doesn't false-positive) while still catching real fabrications.
- If a generation fails the check, regenerate once with a stricter prompt
  ("you mentioned X which is not in the facts, try again using only the
  facts given"). If it fails twice, fall back to the stage-9 degraded path
  in §10 — still LLM output, just shorter and more constrained, not a
  canned string.

### 11.3 Self-check before writing the CSV

Run a cheap automated version of the Stage 4 audit on your own output:
sample 10 rows, check non-empty, check no two are identical, check tone
roughly tracks rank (e.g. flag if a rank >70 row has zero hedging language).
This won't catch everything a human would, but it catches the cheap,
obvious failures before they reach a real audit.

---

## 12. Honeypot/trap defense — summary by trap type

| Trap (per README) | Defense |
|---|---|
| Keyword stuffers (e.g. HR Manager with 9 "AI skills," per `sample_submission.csv`) | JD-rule layer (§7) checks *title/role coherence*, not just skill-list overlap; learned ranker is trained on labels that explicitly include this exact failure mode |
| Plain-language Tier 5s (real fit, no jargon) | Dual JD-side embeddings (§7.1) — the "ideal candidate" narrative form matches plain-language profiles better than literal JD bullets; BM25 alone would miss these entirely, which is why hybrid retrieval matters here specifically |
| Behavioral twins (two similar profiles, one real) | `behavioral_score` (§8) differentiates on engagement signals even when skill/experience profiles look identical |
| The ~80 honeypots | §6 — OR-logic flag rules, hard-removed from the pool before top-100 selection |

---

## 13. Local validation before every submission (3-attempt cap — make these count)

- Run the **actual provided `validate_submission.py`** directly — no need
  to write a mirrored version, just call it as the last step of `rank.py`
  or in CI. It checks: exact header match, exactly 100 data rows, valid
  `CAND_XXXXXXX` IDs, each rank 1–100 exactly once, no duplicate IDs, score
  non-increasing by rank, and on score ties — candidate_id ascending.
- **Gaps it does *not* check** (your own pipeline must self-enforce these,
  since the script takes only the CSV, no candidates file, as input):
  - it never verifies a `candidate_id` actually exists in
    `candidates.jsonl` — a typo'd or wrong ID will pass validation and
    only fail later;
  - it doesn't check whether `reasoning` is non-empty or non-identical
    across rows — that's a Stage 4 manual-review concern, not a Stage 1
    format one, but worth a local check anyway since it's free to add.
- Honeypot self-check: run your own `honeypot_flags.py` against your own
  top 100, confirm ≤10% (target near 0%, not just under the line).
- `tests/test_no_stuffing_regression.py`: assert the top 10 doesn't
  structurally resemble `sample_submission.csv` (e.g. title-role mismatch
  count in top 10 should be ~0).
- Time + memory profile the full `rank.py` run on the actual 100K file on
  a CPU-only, 16GB-capped machine (e.g. `docker run --memory=16g --cpus=N`)
  — Stage 3 reproduces this exactly; if it doesn't pass locally under the
  same constraints, it won't pass there either.

---

## 14. Deliverables checklist (spec §10)

| Item | Where it lives here |
|---|---|
| Submission CSV | output of `rank.py` |
| README with single reproduce command | repo root, the exact command in §10 |
| Full source, no hidden/manual steps | `precompute/` + `rank.py` + `reasoning.py` |
| Pre-computed artifacts or script to produce them | `precompute/artifacts/` + the scripts that built them |
| `requirements.txt` | pinned versions |
| `submission_metadata.yaml` | mirrors portal metadata |
| Sandbox (small-sample, ≤100 candidates, ≤5 min CPU) | `app/streamlit_app.py` |
| Git history showing real iteration | commit per stage as you build (parser → honeypot rules → index → ranker → reasoning → validation) — **not** one final dump; this is an explicit Stage 4 check |
| AI tools declaration | honest, separate from this doc |

---

## 15. Known risks carried into this version (don't re-litigate, just track)

| Risk | Current mitigation | Residual risk |
|---|---|---|
| No ground truth | Hand-labeled eval set + LTR | Eval set is small and self-authored — may not generalize to hidden tiers |
| Single-occurrence honeypot companies | Tier A/B split (§6.2) | Tier B rules are heuristic, not airtight |
| Retrieval ceiling cuts off good candidates | Hybrid search, widened shortlist (~4-5K) | Still possible to lose someone who's both BM25- and embedding-distant from the JD text |
| LLM reasoning hallucination | Fact-restricted prompt + NLI grounding check | NLI models aren't perfect; some false negatives/positives expected |
| JD-rule features are heuristic NLP on free text | Soft, confidence-weighted penalties, not hard filters | Noisy signal could mis-penalize some good candidates — accept as a known tradeoff, don't over-fit single-keyword rules |
| Time budget for 100 LLM calls | Graceful degradation path (§10, §11.2) | Needs real profiling on target hardware before locking in model size |

---

## 16. Build order for the implementation agent

1. `parse_candidates.py` — stream-parse, write `features.parquet`.
2. `company_stats.py` — median first-seen dates, Tier A/B split.
3. `honeypot_flags.py` — implement all 5 OR-logic rules, write `honeypot_flags.parquet`.
4. `build_index.py` — embeddings, FAISS, BM25, JD + ideal-candidate embeddings.
5. `label_eval_set.py` + manually fill in `eval_set.csv`.
6. `train_ranker.py` — LightGBM vs. logistic-regression bake-off on the eval set, save winner.
7. `rank.py` stages 1–8 (retrieval + scoring + honeypot filter + assembly) — get this fully correct and fast first.
8. `reasoning.py` — LLM + NLI fact-check integration, wire into `rank.py` stage 9.
9. `validate_local.py` + `tests/test_no_stuffing_regression.py`.
10. Profile full run under Docker with the exact constraints; tune shortlist size / LLM model size until comfortably inside 5 min / 16GB.
11. `app/streamlit_app.py` sandbox demo on a ≤100-candidate sample.
12. README, requirements.txt, submission_metadata.yaml.
13. Commit history sanity check — confirm it reads as real iteration, not a dump.
