"""Draw a stratified random sample of judged rows for manual reliability checking.

Strata: judge_label x safety_label (6 strata), N_PER_STRATUM rows each,
oversampling rare classes (partial refusals). Because of the oversampling,
overall agreement must be re-weighted by stratum size (see stratum_size in
the key file); per-class agreement and Cohen's kappa are reported unweighted.

Outputs (data/reliability/):
  annotation_sheet.csv  blind sheet for annotators (no judge label, no persona,
                        no safety label), with an empty human_label column
  sample_key.csv        same rows + judge label, strata, and stratum sizes
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JUDGED = ROOT / "data" / "judged" / "full.csv"
OUT_DIR = ROOT / "data" / "reliability"
N_PER_STRATUM = 20
SEED = 42


def main() -> None:
    df = pd.read_csv(JUDGED)
    sizes = df.groupby(["judge_label", "safety_label"]).size().rename("stratum_size")

    parts = []
    for (judge_label, safety_label), grp in df.groupby(["judge_label", "safety_label"]):
        parts.append(grp.sample(n=min(N_PER_STRATUM, len(grp)), random_state=SEED))
    sample = pd.concat(parts).merge(
        sizes.reset_index(), on=["judge_label", "safety_label"]
    )

    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "sample_id", range(1, len(sample) + 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key_cols = [
        "sample_id", "persona_id", "xstest_id", "xstest_type", "safety_label",
        "prompt", "response", "judge_label", "stratum_size",
    ]
    sample[key_cols].to_csv(OUT_DIR / "sample_key.csv", index=False)

    sheet = sample[["sample_id", "prompt", "response"]].copy()
    sheet["human_label"] = ""
    sheet.to_csv(OUT_DIR / "annotation_sheet.csv", index=False)

    print(f"Sampled {len(sample)} rows")
    print(sample.groupby(["judge_label", "safety_label"]).size().to_string())
    print("\npersonas in sample:")
    print(sample.persona_id.value_counts().to_string())


if __name__ == "__main__":
    main()
