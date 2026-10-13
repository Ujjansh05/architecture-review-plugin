"""Build 240 transparent diagnostics, or audit/export independently collected cases."""
import argparse
import json

from arch_review.backend_dataset import export_cases, make_diagnostic_cases
from arch_review.evidence_v2 import PersistentEvidence
from arch_review.settings import runtime_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", help="Optional source-linked JSONL cases in the diagnostic case schema")
    args = parser.parse_args()
    if args.source:
        from pathlib import Path
        rows = [json.loads(line) for line in Path(args.source).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = make_diagnostic_cases()
    audit = export_cases(rows, args.output, PersistentEvidence(runtime_dir() / "evidence-index"))
    print(json.dumps({k: v for k, v in audit.items() if k not in {"near_duplicate_pairs", "coverage"}}, indent=2))
    print(json.dumps({"near_duplicate_pairs": len(audit["near_duplicate_pairs"])}))


if __name__ == "__main__":
    main()
