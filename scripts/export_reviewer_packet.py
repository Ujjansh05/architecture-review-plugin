"""Export a blind human-labeling packet. Never fills reviewer names or verdicts."""
import argparse
import csv
import json
from pathlib import Path

from arch_review.dataset import load_cases
from arch_review.decisions import DEFINITIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_cases(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "scenario_family", "split", "check_id", "decision_definition", "request_json",
                                                  "source_url", "source_chunk_ids", "reviewer_verdict", "reviewer_rationale", "ambiguous_facts",
                                                  "reviewer", "reviewed_at"])
        writer.writeheader()
        for row in rows:
            for check in row["expected"]:
                writer.writerow({"id": row["id"], "scenario_family": row["scenario_family"], "split": row["split"],
                                 "check_id": check, "decision_definition": json.dumps(DEFINITIONS[check]),
                                 "request_json": json.dumps(row["request"], ensure_ascii=False),
                                 "source_url": row.get("attribution", {}).get("url", ""),
                                 "source_chunk_ids": json.dumps(row.get("source_chunk_ids", []))})
    print(json.dumps({"cases": len(rows), "gold_verdicts_included": False, "reviewer_fields_blank": True}))


if __name__ == "__main__":
    main()
