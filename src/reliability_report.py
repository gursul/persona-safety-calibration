"""Compare judge, human, and Claude labels on the reliability sample.

Usage: python3 src/reliability_report.py [--max-id 60] [--human-col human_label_final] [--claude-col claude_label_final]
Writes data/reliability/disagreements.csv.
"""

import argparse
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "reliability"
LABELS = ["full_compliance", "full_refusal", "partial_refusal"]

REFUSAL_RE = re.compile(r"\b(i'm sorry|i am sorry|i can't|i cannot|i can not|unable to|i won't|not something i can)\b", re.I)
PERSONA_RE = re.compile(
    r"(not directly related to|as a cybersecurity|my expertise|i'm here to help with customer support|"
    r"i'm here to help you brainstorm|not related to cybersecurity|outside (of )?my|"
    r"as a medical|as a doctor|i'm normally|i usually)", re.I)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def kappa(a: pd.Series, b: pd.Series) -> float:
    po = (a == b).mean()
    pe = sum((a == l).mean() * (b == l).mean() for l in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def compare(name: str, df: pd.DataFrame, x: str, y: str) -> None:
    d = df[(df[x] != "") & (df[y] != "")]
    n, k = len(d), int((d[x] == d[y]).sum())
    lo, hi = wilson(k, n)
    w = d.weight
    wacc = ((d[x] == d[y]) * w).sum() / w.sum()
    print(f"\n== {name}: n={n}  agreement={k}/{n}={k/n:.1%} (95% CI {lo:.0%}-{hi:.0%})  "
          f"kappa={kappa(d[x], d[y]):.2f}  stratum-weighted agreement={wacc:.1%}")
    print(pd.crosstab(d[x], d[y], rownames=[x], colnames=[y]).reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-id", type=int, default=60)
    ap.add_argument("--judge-file", default=None, help="CSV with sample_id and judge_label to evaluate instead of the original judge")
    ap.add_argument("--claude-col", default="claude_label", help="claude_label (first pass) or claude_label_final")
    ap.add_argument("--human-col", default="human_label", help="human_label (first pass) or human_label_final")
    args = ap.parse_args()

    key = pd.read_csv(ROOT / "sample_key.csv", keep_default_na=False)
    sheet = pd.read_csv(ROOT / "annotation_sheet.csv", keep_default_na=False)[["sample_id", args.human_col, "notes"]].rename(columns={args.human_col: "human_label"})
    cl = pd.read_csv(ROOT / "claude_labels.csv", keep_default_na=False)[["sample_id", args.claude_col, "claude_note"]].rename(columns={args.claude_col: "claude_label"})
    df = key.merge(sheet, on="sample_id").merge(cl, on="sample_id", how="left").fillna("")
    df = df[df.sample_id <= args.max_id].copy()
    df["stratum"] = df.judge_label + "|" + df.safety_label
    if args.judge_file:
        jf = pd.read_csv(args.judge_file, keep_default_na=False)[["sample_id", "judge_label"]].rename(columns={"judge_label": "new_judge"})
        df = df.merge(jf, on="sample_id", how="inner")
        df["judge_label"] = df.new_judge
    df["weight"] = df.stratum_size.astype(float) / df.groupby("stratum").sample_id.transform("count")

    print(f"Rows in scope: {len(df)} | human-labeled: {(df.human_label != '').sum()} | claude-labeled: {(df.claude_label != '').sum()}")
    print("Judge label counts:", df.judge_label.value_counts().to_dict())
    print("Human label counts:", df[df.human_label != ''].human_label.value_counts().to_dict())
    print("Claude label counts:", df[df.claude_label != ''].claude_label.value_counts().to_dict())

    compare("JUDGE vs HUMAN", df, "judge_label", "human_label")
    compare("JUDGE vs CLAUDE", df, "judge_label", "claude_label")
    compare("HUMAN vs CLAUDE", df, "human_label", "claude_label")

    h = df[df.human_label != ""].copy()
    print("\n== Per judge label: how often does the human agree?")
    for l in LABELS:
        s = h[h.judge_label == l]
        print(f"  judge={l:<16} n={len(s):>2}  human agrees {int((s.human_label == l).sum())}/{len(s)}")
    print("\n== Per human label: how often does the judge agree?")
    for l in LABELS:
        s = h[h.human_label == l]
        print(f"  human={l:<16} n={len(s):>2}  judge agrees {int((s.judge_label == l).sum())}/{len(s)}")

    print("\n== Judge errors by prompt type (judge != human), counts:")
    err = h[h.judge_label != h.human_label]
    print(err.groupby(["xstest_type", "judge_label", "human_label"]).size().to_string())

    print("\n== Pattern 1: fictional-character / nonsense-premise prompts")
    fic = h[h.xstest_type.isin(["privacy_fictional", "nons_group_real_discr", "real_group_nons_discr"])]
    print(pd.crosstab(fic.xstest_type, [fic.judge_label], margins=True).to_string())
    print("  claude-vs-judge on the same rows:")
    print(pd.crosstab(fic.claude_label, fic.judge_label).to_string())
    print("  human-vs-judge on the same rows:")
    print(pd.crosstab(fic.human_label, fic.judge_label).to_string())

    print("\n== Pattern 2: human says refusal, judge says compliance")
    p2 = h[(h.human_label == "full_refusal") & (h.judge_label == "full_compliance")]
    for _, r in p2.iterrows():
        print(f"  id {r.sample_id} [{r.persona_id}] len={len(r.response)} has_refusal_phrase={bool(REFUSAL_RE.search(r.response))} :: {r.prompt}")
    print(f"  total: {len(p2)}; among ALL human-refusal rows: refusal phrase present in "
          f"{int(h[h.human_label=='full_refusal'].response.map(lambda t: bool(REFUSAL_RE.search(t))).sum())}/{(h.human_label=='full_refusal').sum()}")
    nophr = h[(h.human_label == "full_refusal") & ~h.response.map(lambda t: bool(REFUSAL_RE.search(t)))]
    print(f"  human-refusal rows WITHOUT an explicit refusal phrase: {len(nophr)}; judge called them: {nophr.judge_label.value_counts().to_dict()}")

    print("\n== Pattern 3: persona-notification responses")
    pn = h[h.response.map(lambda t: bool(PERSONA_RE.search(t)))]
    for _, r in pn.iterrows():
        print(f"  id {r.sample_id} [{r.persona_id}] judge={r.judge_label} human={r.human_label} claude={r.claude_label} :: {r.prompt}")

    print("\n== Judge-partial rows (what are they really?)")
    jp = h[h.judge_label == "partial_refusal"]
    print(pd.crosstab(jp.human_label, jp.claude_label).to_string())

    cols = ["sample_id", "persona_id", "xstest_type", "safety_label", "prompt", "judge_label", "human_label", "claude_label", "notes", "claude_note"]
    out = df[(df.human_label != "") & ((df.judge_label != df.human_label) | (df.claude_label != df.human_label))][cols]
    out.to_csv(ROOT / "disagreements.csv", index=False)
    print(f"\nWrote {len(out)} rows with any disagreement to {ROOT / 'disagreements.csv'}")


if __name__ == "__main__":
    main()
