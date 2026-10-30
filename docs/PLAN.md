# Execution Plan

Status legend: [ ] todo · [~] in progress · [x] done

## Phase 0 — Setup
- [x] Create GitHub repo, local project scaffold
- [x] Pin model access — local inference via MLX on M3 Pro / 36GB RAM
      - Subject model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
      - Judge model: `mlx-community/Qwen2.5-32B-Instruct-4bit` (run
        sequentially after generation, not concurrently with the 7B model)
- [x] `requirements.txt` / environment pinned (mlx-lm, datasets, pandas,
      statsmodels, etc.)

## Phase 1 — Data
- [x] Pull XSTest (450 prompts: 250 safe, 200 unsafe) into
      `data/xstest/xstest_v2.csv` via `src/data_prep.py`
      (source: `natolambert/xstest-v2-copy` on HF Hub; `contrast_*` types
      = unsafe, all other types = safe — counts verified 250/200)
- [x] Finalize 6 persona system prompts in `personas/personas.yaml`:
      neutral baseline, children's tutor, medical professional,
      cybersecurity professional, creative fiction writer, customer
      support agent — role/style only, no explicit safety steering;
      length-matched (154-200 chars across non-neutral personas)

## Phase 2 — Generation
- [x] Build generation script: persona × XSTest prompt → model response
      (temperature = 0, 1 generation per condition) — `src/generate.py`
- [x] Small test run (2 prompts × 6 personas, incl. unsafe prompts) —
      validated end to end: refusals on unsafe prompts across all
      personas, compliance on safe prompts, persona style shows up
      without breaking the refusal mechanism
- [x] Run full grid: 450 × 6 personas = 2700 generations (done, 0 empty responses)
- [x] Store raw generations in `data/generations/` with persona + prompt_id + safety_label
      (full.csv is committed; test files are gitignored)

## Phase 3 — Judging
- [x] Implement LLM-as-judge with XSTest 3-way taxonomy (full compliance /
      partial refusal / full refusal)
- [x] Run judge over all generations → `data/judged/`
- [x] Draw stratified random subset (120 rows: 20 per judge-label x safety-label
      stratum, seed 42) -> `data/reliability/` via `src/sample_reliability.py`
- [x] Manually label the sample (blind to judge), compute agreement
      (Cohen's kappa, per-class agreement, stratum-weighted accuracy).
      Rows 1-60 labeled by Arda and, independently, by Claude
      (`data/reliability/`, rules in `docs/labeling_guide.md`).
      Result: judge vs human 55% agreement, kappa 0.28; 14 of the judge's
      15 partial refusals were wrong, so the judge is NOT usable as is.
      Human vs Claude (first pass) agreement 95%, kappa 0.90.
- [x] Label rows 61-120 (held-out validation set): labeled by Claude
- [ ] Arda labels or reviews rows 61-120 (at least the close calls)

## Phase 3b — Judge improvement loop (must finish before Phase 4)

Goal: a judge whose labels are trustworthy enough for the analysis.
Dev set = sample rows 1-60 (used to find error patterns). Validation set =
rows 61-120 (labels fixed before any prompt work; used for go/no-go).

Loop, each iteration:
1. Revise the judge prompt (one or two changes at a time) and log the change
   and the reason in the experiment journal.
2. Run it on the dev set (about 8 min) and review the remaining errors.
3. When the dev set looks good, run it on the validation set and record
   agreement, kappa and the types of errors the judge still makes.

Stop conditions:
- Success: on the validation set agreement >= 85%, kappa >= 0.70, and no
  single type of error in more than 2 rows.
- Limit: at most 4 prompt iterations.
- If the limit is hit without success, change the prompt or method.

Result: the second prompt (`prompts/judge_v2_1.txt`) met all three conditions on rows 61-120
(agreement 90.0%, kappa 0.80, no error type above 2 rows). Labels for those rows are from
Claude only so far.

After success:
- [x] Re-run the final judge on all 2700 rows -> `data/judged/full_v2_1.csv`.
- [x] Draw a fresh sample stratified by the NEW labels (62 rows, `data/reliability_final/`).
- [x] Fresh sample labeled blind by Claude. Weighted agreement 96%, but the judge's
      refusals on safe prompts (the over-refusal cell) were confirmed in only 2 of 12
      sampled rows, and none of its partial refusals held up.
- [x] Merge partial refusal into refusal (the judge cannot separate them); report the
      main analysis both ways.
- [x] Review the two risky judge cells in full: judge refusals on safe prompts (60 rows)
      and judge compliances on unsafe prompts (134 rows), mixed blind with 60 control rows
      from the large cells (`data/reliability_review/`). Reviewed labels replace the judge's
      labels in those cells.
- [ ] Arda labels the fresh sample and about 40 review rows, to report agreement with
      the reviewed labels.
- [x] Add a third category, deflection (no answer; the reason is scope, irrelevance or lack
      of clarity), tagged in `data/reliability_review/deflection_tags.csv`.
- [ ] Fix the policy for the other ambiguous response types (privacy 'no access' answers,
      story pivots, charitable misreadings) and run Phase 4 under a strict and a lenient
      version of it, with deflection counted separately and as a refusal.

## Phase 4 — Analysis

Data: `data/judged/full_corrected.csv` (`final_label3`). Partial refusal is merged into
refusal. Label scenarios: **primary** (deflection is not a refusal; close calls as
labeled), **strict** (deflection and close-call compliances count as refusal),
**lenient** (close-call refusals count as compliance). Reference: judge-only labels.

- [x] Rates per persona (and per XSTest prompt type) with prompt-bootstrap 95% CIs, risk
      differences vs neutral, and the discrimination index (refusal rate on unsafe minus
      on safe prompts).
- [x] H1: Cochran's Q across personas on safe and on unsafe prompts; exact McNemar tests
      vs neutral with Holm correction.
- [x] H2: logistic model `refused ~ persona * safety_label` (Firth penalty);
      the main H2 test is on the discrimination index, with bootstrap-calibrated p-values
      (a chi-square approximation is unreliable with so few events).
- [x] Robustness: which conclusions hold in all three scenarios; misclassification
      simulation (2-8% random label flips in the unreviewed large cells).
- [x] Save tables and figures to `results/`.

## Phase 5 — Write-up (ACL template; sections in the order required by the course)

- [x] Repository cleanup: README, requirements, data dictionary, fresh-clone check
- [x] References verified (`paper/references.bib`); AI usage declaration drafted
- [ ] Introduction (outline now, final version after the Results)
- [ ] Methods
- [ ] Results (compact: one figure, one main table, one sentence per test; details in the appendix)
- [ ] Discussion and conclusion
- [ ] Abstract (up to 150 words, with the conclusion)
- [x] Repository is public: https://github.com/gursul/persona-safety-calibration
- [ ] Final checks: AI declaration, names and
      matriculation numbers, project title
