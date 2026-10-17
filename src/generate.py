"""Generate model responses for each (persona, XSTest prompt) pair.

Usage:
    python3 src/generate.py --out data/generations/test.csv --limit 5
    python3 src/generate.py --out data/generations/full.csv

Output is appended row by row; rerunning with the same --out resumes.
Decoding is greedy (temperature = 0), one generation per condition, per
the proposal. Model is loaded once and reused across all conditions.
"""

import argparse
import csv
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["persona_id", "xstest_id", "xstest_type", "safety_label", "prompt", "response"]

    finished = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        finished = set(zip(prev["persona_id"], prev["xstest_id"]))
        print(f"Resuming: {len(finished)} generations already in {out_path}")

    total = len(personas) * len(prompts_df)
    done = len(finished)
    t0 = time.time()

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not finished:
            writer.writeheader()
        for persona in personas:
            for _, row in prompts_df.iterrows():
                if (persona["id"], row["id"]) in finished:
                    continue
                messages = build_messages(persona["system_prompt"], row["prompt"])
                chat_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                response = generate(
                    model, tokenizer, prompt=chat_prompt, max_tokens=MAX_TOKENS
                )
                writer.writerow(
                    {
                        "persona_id": persona["id"],
                        "xstest_id": row["id"],
                        "xstest_type": row["type"],
                        "safety_label": row["safety_label"],
                        "prompt": row["prompt"],
                        "response": response,
                    }
                )
                f.flush()
                done += 1
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - t0
                    print(f"  {done}/{total} generations ({elapsed:.1f}s elapsed)", flush=True)

    print(f"Done: {done}/{total} rows in {out_path}")


if __name__ == "__main__":
    main()
