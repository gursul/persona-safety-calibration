"""Draw a stratified random sample of judged rows for manual reliability checking.

Strata: judge_label x safety_label (6 strata), n_per_stratum rows each (all rows
if a stratum is smaller), oversampling rare classes. Because of the oversampling,
overall agreement must be re-weighted by stratum size (see stratum_size in the
key file); per-class agreement and Cohen's kappa are reported unweighted.

Examples:
  python3 src/sample_reliability.py                      # original sample (20 per stratum)
  python3 src/sample_reliability.py --judged data/judged/full_v2_1.csv \\
      --out-dir data/reliability_final --n-per-stratum 12 --seed 7 \\
      --exclude data/reliability/sample_key.csv          # fresh sample for the final judge

Outputs (in --out-dir):
  annotation_sheet.csv  blind sheet for annotators (no judge label, no persona,
                        no safety label), with empty human_label and notes columns
  sample_key.csv        same rows + judge label, strata, and stratum sizes
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default=str(ROOT / "data" / "judged" / "full.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "reliability"))
    ap.add_argument("--n-per-stratum", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", default=None, help="sample_key.csv of rows that must not be drawn again")
    args = ap.parse_args()

    df = pd.read_csv(args.judged, keep_default_na=False)
    sizes = df.groupby(["judge_label", "safety_label"]).size().rename("stratum_size")

    pool = df
    if args.exclude:
        used = pd.read_csv(args.exclude, keep_default_na=False)
        taken = set(zip(used.persona_id, used.xstest_id))
        pool = df[[(p, x) not in taken for p, x in zip(df.persona_id, df.xstest_id)]]

    parts = []
    for _, grp in pool.groupby(["judge_label", "safety_label"]):
        parts.append(grp.sample(n=min(args.n_per_stratum, len(grp)), random_state=args.seed))
    sample = pd.concat(parts).merge(sizes.reset_index(), on=["judge_label", "safety_label"])

    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample.insert(0, "sample_id", range(1, len(sample) + 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_cols = [
        "sample_id", "persona_id", "xstest_id", "xstest_type", "safety_label",
        "prompt", "response", "judge_label", "stratum_size",
    ]
    sample[key_cols].to_csv(out_dir / "sample_key.csv", index=False)

    sheet = sample[["sample_id", "prompt", "response"]].copy()
    sheet["human_label"] = ""
    sheet["notes"] = ""
    sheet.to_csv(out_dir / "annotation_sheet.csv", index=False)

    print(f"Sampled {len(sample)} rows")
    print(sample.groupby(["judge_label", "safety_label"]).size().to_string())
    print("\npersonas in sample:")
    print(sample.persona_id.value_counts().to_string())


if __name__ == "__main__":
    main()
