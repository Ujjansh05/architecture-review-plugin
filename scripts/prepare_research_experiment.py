"""Reproduce the combined draft dataset and blind baseline input export."""
import argparse
import json
from pathlib import Path

from arch_review.dataset import export_dataset, load_cases, validate_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".runtime/research-experiment"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = load_cases() + load_cases(root / "datasets/source-linked-v1/cases.jsonl")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "cases.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    export_dataset(args.output / "export", source, allow_unreviewed=True)
    print(json.dumps({**validate_cases(rows), "directory": str(args.output)}))


if __name__ == "__main__":
    main()
