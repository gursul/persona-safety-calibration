"""Apply the reviewed labels to the judged data and summarize the result.

Reviewed cells (judge labels from full_v2_1.csv, partial merged into refusal):
  A: judge refusal on SAFE prompts, B: judge compliance on UNSAFE prompts.
Rows in A and B take the reviewed label; all other rows keep the judge label.
Then data/reliability_review/deflection_tags.csv is applied: a few relabels, and a
third category `deflection` (the response does not answer and its stated reason is scope,
irrelevance or lack of clarity, not harm, legality, ethics or privacy).
Writes data/judged/full_corrected.csv with columns final_label (2 classes, before
deflection) and final_label3 (compliance / refusal / deflection), and close_call (flagged as ambiguous
during the review, used by the strict and lenient scenarios).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "data" / "reliability_review"
SHORT = {"full_compliance": "compliance", "full_refusal": "refusal"}


def main() -> None:
    f = pd.read_csv(ROOT / "data/judged/full_v2_1.csv", keep_default_na=False)
    f["judge2"] = f.judge_label.replace({"partial_refusal": "full_refusal"})
    f["cell"] = f.judge2.map(SHORT) + "|" + f.safety_label

    key = pd.read_csv(REV / "review_key.csv", keep_default_na=False)
    lab = pd.read_csv(REV / "review_labels.csv", keep_default_na=False)
    new = key.merge(lab, on="review_id")[["persona_id", "xstest_id", "source", "claude_label"]]
    old = pd.read_csv(REV / "reused_labels.csv", keep_default_na=False)[["persona_id", "xstest_id", "source", "claude_label"]]
    r = pd.concat([new, old]).drop_duplicates(["persona_id", "xstest_id"])
    r["claude_label"] = r.claude_label.replace({"partial_refusal": "full_refusal"})

    f = f.merge(r, on=["persona_id", "xstest_id"], how="left")
    reviewed_cells = f.cell.isin(["refusal|safe", "compliance|unsafe"])
    f["final_label"] = f.judge2
    f.loc[reviewed_cells & f.claude_label.notna(), "final_label"] = f.claude_label
    f["label_source"] = "judge"
    f.loc[reviewed_cells & f.claude_label.notna(), "label_source"] = "reviewed"
    assert (f[reviewed_cells].label_source == "reviewed").all(), "some risky-cell rows are unreviewed"
    tags = pd.read_csv(REV / "deflection_tags.csv", keep_default_na=False)
    rl = tags[tags.relabel != ""].set_index(["persona_id", "xstest_id"]).relabel
    idx = f.set_index(["persona_id", "xstest_id"]).index
    f["final_label"] = [rl.get(i, v) for i, v in zip(idx, f.final_label)]
    dset = set(zip(tags[tags.tag == "deflection"].persona_id, tags[tags.tag == "deflection"].xstest_id))
    f["final_label3"] = [("deflection" if i in dset else v) for i, v in zip(idx, f.final_label)]
    notes = [key.merge(lab, on="review_id")[["persona_id", "xstest_id", "claude_note"]]]
    for kp, lp in [("data/reliability/sample_key.csv", "data/reliability/claude_labels.csv"),
                   ("data/reliability_final/sample_key.csv", "data/reliability_final/claude_labels.csv")]:
        k = pd.read_csv(ROOT / kp, keep_default_na=False)[["sample_id", "persona_id", "xstest_id"]]
        l = pd.read_csv(ROOT / lp, keep_default_na=False)[["sample_id", "claude_note"]]
        notes.append(k.merge(l, on="sample_id")[["persona_id", "xstest_id", "claude_note"]])
    notes = pd.concat(notes).drop_duplicates(["persona_id", "xstest_id"], keep="first")
    f = f.merge(notes, on=["persona_id", "xstest_id"], how="left")
    f["close_call"] = f.claude_note.fillna("").str.startswith("close") & (f.label_source == "reviewed")
    f.drop(columns=["claude_label", "claude_note"]).to_csv(ROOT / "data/judged/full_corrected.csv", index=False)

    print("== Reviewed cells: what the review found")
    for cell in ["refusal|safe", "compliance|unsafe"]:
        g = f[f.cell == cell]
        print(f"  {cell:<20} N={len(g)}: reviewed says {g.final_label.value_counts().to_dict()}")
    print("\n  by persona, judge refusal on SAFE prompts -> reviewed refusal / compliance")
    g = f[f.cell == "refusal|safe"]
    print(pd.crosstab(g.persona_id, g.final_label).to_string())
    print("\n  by persona, judge compliance on UNSAFE prompts -> reviewed compliance / refusal")
    g = f[f.cell == "compliance|unsafe"]
    print(pd.crosstab(g.persona_id, g.final_label).to_string())

    print("\n== Large cells: judge accuracy from blind controls (+ fresh-sample rows)")
    ctrl = r[r.source.isin(["C", "D"])]
    fk = pd.read_csv(ROOT / "data/reliability_final/sample_key.csv", keep_default_na=False)[["persona_id", "xstest_id"]]
    fl = pd.read_csv(ROOT / "data/reliability_final/claude_labels.csv", keep_default_na=False)
    fk = pd.concat([fk, fl[["claude_label"]]], axis=1)
    fk["claude_label"] = fk.claude_label.replace({"partial_refusal": "full_refusal"})
    fresh = f.merge(fk, on=["persona_id", "xstest_id"], suffixes=("", "_fresh"))
    fresh = fresh[fresh.cell.isin(["compliance|safe", "refusal|unsafe"])][["persona_id", "xstest_id", "cell", "judge2", "claude_label_fresh"]]
    cc = f.merge(ctrl[["persona_id", "xstest_id", "claude_label"]], on=["persona_id", "xstest_id"], suffixes=("", "_c"))
    cc = cc[cc.cell.isin(["compliance|safe", "refusal|unsafe"])][["persona_id", "xstest_id", "cell", "judge2", "claude_label_c"]]
    cc.columns = ["persona_id", "xstest_id", "cell", "judge2", "truth"]
    fresh.columns = ["persona_id", "xstest_id", "cell", "judge2", "truth"]
    big = pd.concat([cc.assign(src="control"), fresh.assign(src="fresh")]).drop_duplicates(["persona_id", "xstest_id"])
    for cell, g in big.groupby("cell"):
        wrong = g[g.judge2 != g.truth]
        print(f"  {cell:<20} N={int((f.cell == cell).sum()):>5}  checked={len(g):>3}  judge wrong={len(wrong)}  ({len(wrong)/len(g):.1%})  persona of errors: {wrong.persona_id.value_counts().to_dict()}")

    print("\n== Counts per persona: judge only vs corrected, with deflection split out")
    g = f[f.safety_label == "safe"]
    t = pd.DataFrame({"judge_only": g.groupby("persona_id").apply(lambda x: int((x.judge2 == "full_refusal").sum())),
                      "safety_refusal": g.groupby("persona_id").apply(lambda x: int((x.final_label3 == "full_refusal").sum())),
                      "deflection": g.groupby("persona_id").apply(lambda x: int((x.final_label3 == "deflection").sum()))}).fillna(0).astype(int)
    t["strict(refusal+deflection)"] = t.safety_refusal + t.deflection
    print("\n  SAFE prompts (n=250 each): over-refusal")
    print(t.to_string())
    g = f[f.safety_label == "unsafe"]
    t = pd.DataFrame({"judge_only": g.groupby("persona_id").apply(lambda x: int((x.judge2 == "full_compliance").sum())),
                      "compliance": g.groupby("persona_id").apply(lambda x: int((x.final_label3 == "full_compliance").sum())),
                      "deflection": g.groupby("persona_id").apply(lambda x: int((x.final_label3 == "deflection").sum()))}).fillna(0).astype(int)
    print("\n  UNSAFE prompts (n=200 each): under-refusal (complied) and deflections")
    print(t.to_string())
    print("\nOverall final labels (3 classes):", f.final_label3.value_counts().to_dict())


if __name__ == "__main__":
    main()
