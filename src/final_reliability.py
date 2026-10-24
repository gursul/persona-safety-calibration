"""Score the final judge on the fresh reliability sample (data/reliability_final).

Compares judge labels with Claude's labels and, when filled in, the human labels
in annotation_sheet.csv. Reports agreement, kappa, stratum-weighted agreement,
per-class and per-persona agreement, and every disagreement.

Usage: python3 src/final_reliability.py [--dir data/reliability_final] [--human-col human_label]
"""

import argparse
from pathlib import Path

import pandas as pd

from reliability_report import LABELS, kappa, wilson

ROOT = Path(__file__).resolve().parents[1]


def summarize(name: str, df: pd.DataFrame, x: str, y: str) -> None:
    d = df[(df[x] != "") & (df[y] != "")]
    n, k = len(d), int((d[x] == d[y]).sum())
    if n == 0:
        return
    lo, hi = wilson(k, n)
    wacc = ((d[x] == d[y]) * d.weight).sum() / d.weight.sum()
    print(f"\n== {name}: n={n}  agreement={k}/{n}={k/n:.1%} (95% CI {lo:.0%}-{hi:.0%})  "
          f"kappa={kappa(d[x], d[y]):.2f}  stratum-weighted agreement={wacc:.1%}")
    print(pd.crosstab(d[x], d[y], rownames=[x], colnames=[y]).reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "reliability_final"))
    ap.add_argument("--human-col", default="human_label")
    args = ap.parse_args()
    d = Path(args.dir)

    key = pd.read_csv(d / "sample_key.csv", keep_default_na=False)
    cl = pd.read_csv(d / "claude_labels.csv", keep_default_na=False)
    sheet = pd.read_csv(d / "annotation_sheet.csv", keep_default_na=False)[["sample_id", args.human_col, "notes"]]
    sheet = sheet.rename(columns={args.human_col: "human_label"})
    df = key.merge(cl, on="sample_id").merge(sheet, on="sample_id")
    df["stratum"] = df.judge_label + "|" + df.safety_label
    df["weight"] = df.stratum_size.astype(float) / df.groupby("stratum").sample_id.transform("count")

    print(f"Rows: {len(df)} | claude-labeled: {(df.claude_label != '').sum()} | human-labeled: {(df.human_label != '').sum()}")
    print("Judge:", df.judge_label.value_counts().to_dict())
    print("Claude:", df.claude_label.value_counts().to_dict())

    summarize("JUDGE vs CLAUDE", df, "judge_label", "claude_label")
    summarize("JUDGE vs HUMAN", df, "judge_label", "human_label")
    summarize("HUMAN vs CLAUDE", df, "human_label", "claude_label")

    print("\n== Per judge label: how often does Claude agree?")
    for l in LABELS:
        s = df[df.judge_label == l]
        print(f"  judge={l:<16} n={len(s):>2}  agree {int((s.claude_label == l).sum())}/{len(s)}")

    print("\n== Per persona (judge vs Claude)")
    for p, s in df.groupby("persona_id"):
        print(f"  {p:<28} n={len(s):>2}  agree {int((s.judge_label == s.claude_label).sum())}/{len(s)}")
    print("\n== Per safety label (judge vs Claude)")
    for p, s in df.groupby("safety_label"):
        print(f"  {p:<28} n={len(s):>2}  agree {int((s.judge_label == s.claude_label).sum())}/{len(s)}")

    close = df[df.claude_note.str.startswith("close")]
    print(f"\n== Claude close calls: {len(close)} rows; judge agrees with Claude on {int((close.judge_label == close.claude_label).sum())} of them")

    err = df[df.judge_label != df.claude_label]
    print(f"\n== Disagreements (judge vs Claude): {len(err)}")
    for _, r in err.iterrows():
        print(f"  #{r.sample_id} [{r.persona_id}] {r.xstest_type} | judge={r.judge_label} claude={r.claude_label} {r.claude_note}\n     Q: {r.prompt}")


if __name__ == "__main__":
    main()
