# Does Persona Prompting Change Safety Calibration?

Project for *Understanding Language Models* (University of Tübingen, Summer 2026) by
Yusuf Nar and Arda Gürsul.

We test whether prompting an instruction-tuned language model with a role persona changes
how often it refuses benign-but-scary-sounding requests (over-refusal) and how often it
complies with genuinely unsafe ones (under-refusal). One fixed model, Qwen2.5-7B-Instruct,
answers the 450 prompts of [XSTest](https://github.com/paul-rottger/exaggerated-safety)
(250 safe, 200 unsafe) under six personas (2,700 responses). Responses are labeled as
compliance, refusal or off-topic deflection by an LLM judge (Qwen2.5-32B-Instruct), which we
validate against manual labels and correct where it is unreliable. Details, results and
limitations are in the paper.

## Repository structure

```
README.md             this file
requirements.txt      Python dependencies
personas/             the six persona system prompts (personas.yaml)
prompts/              judge prompts (judge_v2.txt, judge_v2_1.txt = final)
src/                  all code (see "Pipeline")
data/                 generated responses, judge labels, manual labels (see "Data files")
results/              tables and figure produced by src/analysis.py
docs/                 labeling guide, proposal, project plan
```

## Setup

- Python 3.10 or newer (tested with 3.13).
- `pip install -r requirements.txt`.
- Generation and judging use [MLX](https://github.com/ml-explore/mlx) and need an Apple Silicon Mac
  (we used an M3 Pro with 36 GB RAM). The two models are downloaded automatically from the
  Hugging Face Hub on first use: `mlx-community/Qwen2.5-7B-Instruct-4bit` (subject model) and
  `mlx-community/Qwen2.5-32B-Instruct-4bit` (judge, about 18 GB). No API keys are needed.
- The analysis needs only `pandas numpy scipy statsmodels matplotlib pyyaml` and runs on any machine.

## Quick reproduction (analysis only, about 30 seconds)

All data needed for the analysis is in the repository.

```bash
python3 src/analysis.py        # writes results/*.csv and results/fig_rates.png
```

## Full pipeline

Run from the repository root. Runtimes are for the hardware above.

```bash
python3 src/data_prep.py                                       # download XSTest -> data/xstest/xstest_v2.csv
python3 src/generate.py --out data/generations/full.csv        # 2,700 responses, temperature 0 (~4.5 h)
python3 src/judge.py --in data/generations/full.csv \
    --out data/judged/full_v2_1.csv --prompt-file prompts/judge_v2_1.txt   # final judge (~4 h)
python3 src/build_review_set.py                                # blind sheet for the review step
python3 src/apply_review.py                                    # corrected labels -> data/judged/full_corrected.csv
python3 src/analysis.py                                        # tables and figure -> results/
```

Notes:
- `generate.py` and `judge.py` write row by row and resume if interrupted.
- The first judge (single-label prompt, `data/judged/full.csv`) is run with
  `python3 src/judge.py --in data/generations/full.csv --out data/judged/full.csv`.
- Manual and LLM labeling steps are not scripts. Their outputs are committed
  (`data/reliability*/`); the rules are in [docs/labeling_guide.md](docs/labeling_guide.md).
- `src/sample_reliability.py`, `src/reliability_report.py` and `src/final_reliability.py` draw
  the reliability samples and score the judge against the manual labels.
- Random seeds: analysis 2026; samples 42, 7 and 11 (see the script arguments).

## Code overview

| Script | Purpose |
|---|---|
| `src/data_prep.py` | download XSTest and add the safe/unsafe label |
| `src/personas.py` | load the persona definitions |
| `src/generate.py` | generate one response per persona and prompt |
| `src/judge.py` | label responses with the LLM judge (two prompt versions) |
| `src/sample_reliability.py` | draw stratified samples for manual labeling |
| `src/reliability_report.py`, `src/final_reliability.py` | agreement between judge and manual labels |
| `src/build_review_set.py`, `src/apply_review.py` | review the risky judge cells and build the corrected labels |
| `src/analysis.py` | rates, tests, robustness checks, figure |

## Data files

| File | Content |
|---|---|
| `data/generations/full.csv` | `persona_id, xstest_id, xstest_type, safety_label, prompt, response` (2,700 rows) |
| `data/judged/full.csv` | the same plus `judge_raw, judge_label` from the first judge |
| `data/judged/full_v2_1.csv` | the same from the final judge (`prompts/judge_v2_1.txt`) |
| `data/judged/full_corrected.csv` | final labels: `judge2` (judge label, partial merged into refusal), `cell`, `label_source` (`judge` or `reviewed`), `final_label`, `final_label3` (compliance, refusal or deflection), `close_call` |
| `data/reliability/` | first 120-row reliability sample: `sample_key.csv`, blind `annotation_sheet.csv` with human labels, `claude_labels.csv`, judge outputs on the development and validation rows |
| `data/reliability_final/` | fresh 62-row sample drawn from the final judge's labels |
| `data/reliability_review/` | review of the risky judge cells: sheets, keys, labels, `deflection_tags.csv` |
| `results/` | `rates_by_scenario.csv`, `tests_by_scenario.csv`, `type_rates_primary.csv`, `type_tests.csv`, `firth_coefficients.csv`, `misclassification_sim.csv`, `fig_rates.png` |

`persona_id` takes the values `neutral, childrens_tutor, medical_professional,
cybersecurity_professional, creative_fiction_writer, customer_support_agent`.

## Label scheme

Responses are labeled `full_compliance`, `full_refusal` or `partial_refusal` (XSTest taxonomy;
partial refusals are merged into refusal in the analysis), plus an additional `deflection`
category for responses that give no answer for reasons of scope or lack of clarity. The analysis
is run under three label scenarios (primary, strict, lenient). See
[docs/labeling_guide.md](docs/labeling_guide.md) and the paper.

## AI use

Claude (Anthropic) was used as a coding assistant and as an LLM annotator for the response
labels in the review step. See the AI usage declaration in the paper.

## References

- Röttger et al. (2024). XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models. NAACL 2024.
- Plaza-del-Arco et al. (2025). No for Some, Yes for Others: Persona Prompts and Other Sources of False Refusal in Language Models.
- Qwen Team (2024). Qwen2.5 Technical Report.
