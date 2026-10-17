# Does Persona Prompting Change Safety Calibration?

Understanding Language Models (Tübingen, Summer 2026) — project by Yusuf Nar and Arda Gürsul.

## Research question

Does prompting an instruction-tuned LLM to adopt a persona change its refusal
behavior — not just the overall refusal rate, but its ability to tell apart
genuinely unsafe requests from benign-but-scary-sounding ones?

We evaluate a single fixed model (planned: Qwen2.5-7B-Instruct) on the
[XSTest](https://github.com/paul-rottger/exaggerated-safety) benchmark (450
prompts: 250 safe/benign, 200 unsafe) under 5–6 persona system prompts,
classify each response with an LLM-as-judge into the XSTest three-way
taxonomy (full compliance / partial refusal / full refusal), and test for a
persona × safety-label interaction via logistic regression.

See [PLAN.md](PLAN.md) for the step-by-step execution plan and
[docs/proposal.pdf](docs/proposal.pdf) for the original proposal.

## Repo layout

- `personas/` — persona system prompt definitions
- `prompts/` — XSTest prompt data / judge prompt templates
- `src/` — generation, judging, and analysis code
- `data/` — raw and processed datasets (generations, judge labels)
- `results/` — figures, tables, regression outputs
- `notebooks/` — exploratory analysis
- `docs/` — proposal and write-ups

## Status

Project setup in progress. See [PLAN.md](PLAN.md) for current stage.
