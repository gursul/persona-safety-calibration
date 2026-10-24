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
- [ ] Label the fresh sample blind and report that agreement as the final
      judge-reliability number (the validation set is no longer clean after tuning).

## Phase 4 — Analysis
- [ ] Compute refusal rate on safe prompts (over-refusal) and on unsafe
      prompts (under-refusal) per persona
- [ ] Logistic regression: response ~ safety_label * persona (+ prompt
      random effect / clustered SE for repeated prompt use)
- [ ] Prompt-level bootstrap CIs as robustness check
- [ ] Test H1 (persona shifts refusal behavior) and H2 (persona × safety-label
      interaction, i.e. calibration not just rate shift)

## Phase 5 — Write-up
- [ ] Figures: refusal rate by persona × safety label
- [ ] Results tables (regression coefficients, CIs)
- [ ] Final report / slides

## Open decisions
- Exact persona wording (draft + freeze before generation to avoid re-runs)
- New judge prompt design (see Phase 3b)
- Significance threshold / correction for multiple persona comparisons
