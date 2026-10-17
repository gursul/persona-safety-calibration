"""Pull XSTest v2 prompts and save a labeled CSV to data/xstest/xstest_v2.csv.

Source: natolambert/xstest-v2-copy on the Hugging Face Hub (mirror of the
original XSTest release, Röttger et al. 2024). The 'prompts' split carries
the 450 prompts; 'contrast_*' types are the 200 unsafe contrast prompts,
all other types are the 250 safe prompts.
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "xstest" / "xstest_v2.csv"


def main() -> None:
    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    df = ds.to_pandas()[["id", "type", "prompt"]].copy()
    df["safety_label"] = df["type"].apply(
        lambda t: "unsafe" if t.startswith("contrast_") else "safe"
    )

    assert (df["safety_label"] == "safe").sum() == 250
    assert (df["safety_label"] == "unsafe").sum() == 200

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} prompts to {OUT_PATH}")
    print(df["safety_label"].value_counts())


if __name__ == "__main__":
    main()
