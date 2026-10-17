# Execution Plan

Status legend: [ ] todo · [~] in progress · [x] done

## Phase 0 — Setup
- [x] Create GitHub repo, local project scaffold
- [ ] Pin model access (confirm Qwen2.5-7B-Instruct availability — local GPU / API / HF Inference)
- [ ] Pick + pin LLM-as-judge model (should differ from the model under test)
- [ ] `requirements.txt` / environment pinned

## Phase 1 — Data
- [ ] Pull XSTest (450 prompts: 250 safe, 200 unsafe) into `data/xstest/`
- [ ] Verify prompt categories / metadata match the published taxonomy
- [ ] Finalize 5–6 persona system prompts in `personas/` (neutral baseline,
      children's tutor, medical professional, cybersecurity professional,
      creative fiction writer, + optional 6th) — roles/style only, no
      explicit safety steering

## Phase 2 — Generation
- [ ] Build generation script: persona × XSTest prompt → model response
      (deterministic / low-temperature, 1 generation per condition)
- [ ] Run full grid: 450 × 5(6) generations
- [ ] Store raw generations in `data/generations/` with persona + prompt_id + safety_label

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
- Model hosting: local inference vs. API vs. HF endpoint
- Judge model choice and prompt
- Significance threshold / correction for multiple persona comparisons
