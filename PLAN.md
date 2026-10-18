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
- [ ] Run full grid: 450 × 6 personas = 2700 generations
- [ ] Store raw generations in `data/generations/` with persona + prompt_id + safety_label
      (full.csv is committed; test files are gitignored)

## Phase 3 — Judging
- [ ] Implement LLM-as-judge with XSTest 3-way taxonomy (full compliance /
      partial refusal / full refusal)
- [ ] Run judge over all generations → `data/judged/`
- [ ] Draw stratified random subset, manually label, compute judge agreement
      (Cohen's kappa) to validate judge reliability

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
- Judge prompt wording / few-shot examples for the 3-way taxonomy
- Significance threshold / correction for multiple persona comparisons
