"""Publish a compact reproducible web experiment without local paths or host logs."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from arch_review.web_corpus import sha, write_json
from arch_review.web_training import load_corpus, validate_web_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    corpus, documents, _ = load_corpus(args.corpus)
    manifest, splits = validate_web_export(args.data)
    result = json.loads(args.training_report.read_text(encoding="utf-8"))
    if result["dataset_manifest"] != manifest:
        raise ValueError("Training report does not belong to this export")
    if sha((args.corpus / "manifest.json").read_bytes()) != manifest["corpus_manifest_sha256"]:
        raise ValueError("Export does not belong to this corpus snapshot")
    args.snapshot.mkdir(parents=True, exist_ok=False)
    for source, target, names in [
        (args.corpus, args.snapshot / "corpus", list(corpus["files"]) + ["manifest.json"]),
        (args.data, args.snapshot / "export", list(manifest["files"]) + ["cases.jsonl", "manifest.json", "rejections.json"]),
    ]:
        target.mkdir()
        for name in names:
            if Path(name).name != name:
                raise ValueError("Unexpected manifest filename")
            shutil.copyfile(source / name, target / name)
    keys = ("status", "device", "epochs", "selected_epoch", "train_cases", "development_cases", "calibration_cases",
            "test_evaluated", "expert_reviewed_labels", "encoder_unchanged_verified", "base_checkpoint_unchanged_verified",
            "artifact_round_trip_verified", "base_weights_sha256", "head_sha256", "head_bytes", "history", "before",
            "after_unfitted", "after", "calibration", "elapsed_seconds", "paid_gpu_spend_usd", "artifact_format")
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "model": {"name": "Laya Multilingual", "approximate_parameters": 322_000_000,
                        "approximate_trained_parameters": 15_000_000, "mode": "frozen_encoder_head_update"},
              "corpus": corpus, "source_licenses": sorted({doc["license"] for doc in documents.values()}),
              "dataset": manifest, "split_counts": {split: len(rows) for split, rows in splits.items()},
              "split_check_coverage": {split: sorted({key for row in rows for key in row["expected"]})
                                       for split, rows in splits.items()},
              "training": {key: result[key] for key in keys},
              "limitations": ["Automatically authored labels, not human-verified correctness",
                              "Rules define and gate the labels; rule agreement is not an independent baseline",
                              "Check split counts and per-decision coverage before interpreting metrics",
                              "Calibration on a small sample is unvalidated",
                              "Final test remains untouched; no final quality claim",
                              "Peak RAM and full-model training were not measured",
                              "Earlier datasets were not used in this run, but source topics may overlap older experiments",
                              "The head is experimental and is not installed as the default plugin checkpoint"]}
    report["interpretation"] = ("Development label agreement: "
                                f"{result['before']['overall']['accuracy']:.4f} before, "
                                f"{result['after']['overall']['accuracy']:.4f} after; inspect scope before drawing conclusions.")
    write_json(args.report, report)
    print(json.dumps({"snapshot_documents": len(documents), "snapshot_cases": sum(map(len, splits.values()))}))


if __name__ == "__main__":
    main()
