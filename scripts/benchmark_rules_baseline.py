"""Replay published synthetic cases with deterministic rules; no model or download.

This is a regression baseline, not evidence of real-world production accuracy.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def run(source):
    from arch_review.research_fact_input import CACHE_NAMES
    from arch_review.rules_v2 import evaluate_check
    from arch_review.schema import Verdict
    from arch_review.schema_v2 import CheckInput

    results = []
    identifiers = set()
    digest = hashlib.sha256()
    counts = defaultdict(lambda: {"count": 0, "correct": 0, "incorrect_approvals": 0})
    reverse = {v: k for k, v in CACHE_NAMES.items()}
    with source.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError("Every benchmark case requires a nonempty string ID")
            if row["id"] in identifiers:
                raise ValueError("Benchmark case IDs must be unique")
            identifiers.add(row["id"])
            if not isinstance(row.get("questions"), dict) or len(row["questions"]) != 1:
                raise ValueError("Every benchmark case must contain exactly one decision question")
            check = next(iter(row["questions"]))
            if (not isinstance(row.get("expected"), dict) or set(row["expected"]) != {check}
                    or row["expected"][check] not in {label.value for label in Verdict}):
                raise ValueError("Expected verdict must match the single question and supported verdict labels")
            facts = json.loads(row["state"])["facts"]
            if check == "cache_freshness":
                facts = {reverse.get(key, key): value for key, value in facts.items()}
            expected = row["expected"][check]
            finding = evaluate_check(CheckInput(check_id=check, facts=facts))
            if finding.verdict is None:
                raise ValueError("Benchmark case does not request an applicable decision")
            predicted = finding.verdict.value
            result = {"id": row["id"], "check_id": check, "expected": expected, "predicted": predicted}
            results.append(result)
            counts[check]["count"] += 1
            counts[check]["correct"] += predicted == expected
            counts[check]["incorrect_approvals"] += predicted == "supported" and expected != "supported"
    correct = sum(row["predicted"] == row["expected"] for row in results)
    return {"format": "rules", "input_sha256": digest.hexdigest(),
            "count": len(results), "correct": correct, "accuracy": correct / len(results) if results else None,
            "incorrect_approvals": sum(c["incorrect_approvals"] for c in counts.values()),
            "per_check": dict(counts), "results": results,
            "dataset_role": "public_regression", "independent_real_world_accuracy": None,
            "note": "Finite synthetic mechanisms closely match the rules. This does not establish general architecture accuracy or model advantage."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "docs/benchmarks/2026-10-final-test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
