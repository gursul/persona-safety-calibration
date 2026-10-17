"""Load and sanity-check persona definitions from personas/personas.yaml."""

from pathlib import Path

import yaml

PERSONAS_PATH = Path(__file__).resolve().parents[1] / "personas" / "personas.yaml"


def load_personas(path: Path = PERSONAS_PATH) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["personas"]


def main() -> None:
    personas = load_personas()
    lengths = {p["id"]: len(p["system_prompt"]) for p in personas}
    print(f"Loaded {len(personas)} personas:")
    for pid, length in lengths.items():
        print(f"  {pid:<24} {length:>4} chars")

    non_neutral = {k: v for k, v in lengths.items() if k != "neutral"}
    spread = max(non_neutral.values()) - min(non_neutral.values())
    print(f"\nLength spread across non-neutral personas: {spread} chars")


if __name__ == "__main__":
    main()
