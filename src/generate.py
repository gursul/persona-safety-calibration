"""Generate model responses for each (persona, XSTest prompt) pair.

Usage:
    python3 src/generate.py --out data/generations/test.csv --limit 5
    python3 src/generate.py --out data/generations/full.csv

Decoding is greedy (temperature = 0), one generation per condition, per
the proposal. Model is loaded once and reused across all conditions.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from mlx_lm import generate, load

from personas import load_personas

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
XSTEST_PATH = Path(__file__).resolve().parents[1] / "data" / "xstest" / "xstest_v2.csv"
MAX_TOKENS = 256


def build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit to first N XSTest prompts (for a quick test run)"
    )
    args = parser.parse_args()

    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)

    personas = load_personas()
    prompts_df = pd.read_csv(XSTEST_PATH)
    if args.limit:
        prompts_df = prompts_df.head(args.limit)

    rows = []
    total = len(personas) * len(prompts_df)
    done = 0
    t0 = time.time()

    for persona in personas:
        for _, row in prompts_df.iterrows():
            messages = build_messages(persona["system_prompt"], row["prompt"])
            chat_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            response = generate(
                model, tokenizer, prompt=chat_prompt, max_tokens=MAX_TOKENS
            )
            rows.append(
                {
                    "persona_id": persona["id"],
                    "xstest_id": row["id"],
                    "xstest_type": row["type"],
                    "safety_label": row["safety_label"],
                    "prompt": row["prompt"],
                    "response": response,
                }
            )
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"  {done}/{total} generations ({elapsed:.1f}s elapsed)")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
