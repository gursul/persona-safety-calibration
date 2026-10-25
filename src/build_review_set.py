"""Build the review set for the risky judge cells plus blind control rows.

Cells reviewed in full (judge labels from data/judged/full_v2_1.csv, partial merged into refusal):
  A: judge says refusal on a SAFE prompt  (possible false over-refusal)
  B: judge says compliance on an UNSAFE prompt  (possible false under-refusal)
Control rows (to keep the reviewer blind to which cell a row comes from and to measure
judge accuracy in the large cells): 5 per persona from each of
  C: judge compliance on SAFE prompts,  D: judge refusal on UNSAFE prompts.

Rows that already have a Claude label from the earlier samples are not re-labeled;
their labels are reused (same generations, so the responses are identical).

Outputs in data/reliability_review/:
  review_sheet.csv  blind sheet (review_id, prompt, response, label, notes), shuffled
  review_key.csv    review_id -> persona, xstest_id, cell, judge label, safety label
  reused_labels.csv labels reused from earlier samples for rows in the cells
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "reliability_review"
SEED = 11


def main() -> None:
    f = pd.read_csv(ROOT / "data/judged/full_v2_1.csv", keep_default_na=False)
    f["judge2"] = f.judge_label.replace({"partial_refusal": "full_refusal"})
    f["cell"] = f.judge2.map({"full_refusal": "refusal", "full_compliance": "compliance"}) + "|" + f.safety_label

    prior = []
    for key_path, lab_path in [("data/reliability/sample_key.csv", "data/reliability/claude_labels.csv"),
                               ("data/reliability_final/sample_key.csv", "data/reliability_final/claude_labels.csv")]:
        k = pd.read_csv(ROOT / key_path, keep_default_na=False)[["sample_id", "persona_id", "xstest_id"]]
        l = pd.read_csv(ROOT / lab_path, keep_default_na=False)
        col = "claude_label_final" if "claude_label_final" in l.columns else "claude_label"
        prior.append(k.merge(l[["sample_id", col]].rename(columns={col: "claude_label"}), on="sample_id")[["persona_id", "xstest_id", "claude_label"]])
    prior = pd.concat(prior).drop_duplicates(["persona_id", "xstest_id"], keep="last")
    prior["claude_label"] = prior.claude_label.replace({"partial_refusal": "full_refusal"})
    f = f.merge(prior, on=["persona_id", "xstest_id"], how="left")

    A = f[f.cell == "refusal|safe"].copy(); A["source"] = "A"
    B = f[f.cell == "compliance|unsafe"].copy(); B["source"] = "B"
    controls = []
    for cell, tag in [("compliance|safe", "C"), ("refusal|unsafe", "D")]:
        pool = f[(f.cell == cell) & f.claude_label.isna()]
        for _, g in pool.groupby("persona_id"):
            s = g.sample(n=5, random_state=SEED); s["source"] = tag; controls.append(s)
    reviewed = pd.concat([A, B] + controls)

    todo = reviewed[reviewed.claude_label.isna()].sample(frac=1, random_state=SEED).reset_index(drop=True)
    todo.insert(0, "review_id", range(1, len(todo) + 1))
    OUT.mkdir(parents=True, exist_ok=True)
    todo[["review_id", "persona_id", "xstest_id", "xstest_type", "safety_label", "judge_label", "judge2", "source"]].to_csv(OUT / "review_key.csv", index=False)
    sheet = todo[["review_id", "prompt", "response"]].copy(); sheet["claude_label"] = ""; sheet["notes"] = ""
    sheet.to_csv(OUT / "review_sheet.csv", index=False)
    reviewed[reviewed.claude_label.notna()][["persona_id", "xstest_id", "source", "judge2", "claude_label"]].to_csv(OUT / "reused_labels.csv", index=False)

    print("cell sizes:", f.cell.value_counts().to_dict())
    print("A rows:", len(A), "(already labeled:", int(A.claude_label.notna().sum()), ")  B rows:", len(B), "(already labeled:", int(B.claude_label.notna().sum()), ")")
    print("controls:", len(pd.concat(controls)))
    print("to label now:", len(todo), "| by source:", todo.source.value_counts().to_dict())


if __name__ == "__main__":
    main()
