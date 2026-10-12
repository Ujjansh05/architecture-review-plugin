import hashlib
import json
from collections import Counter
from pathlib import Path

from .decisions import DEFINITIONS, question
from .evidence import EvidenceStore, data_path
from .model import build_state
from .schema import ReviewRequest, Verdict

SPLITS = ("train", "development", "calibration", "test")


def load_cases(path: Path | None = None) -> list[dict]:
    source = path or data_path("pilot.jsonl")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_cases(rows)
    return rows


def expert_reviewed(row):
    return (row.get("review_status") == "expert_reviewed" and bool(row.get("reviewer"))
            and bool(row.get("reviewed_at")))


def validate_cases(rows):
    identifiers, families, inputs = set(), {}, {}
    for row in rows:
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate case ID: {row['id']}")
        identifiers.add(row["id"])
        if row["split"] not in SPLITS:
            raise ValueError("Invalid dataset split")
        family = row["scenario_family"]
        if families.setdefault(family, row["split"]) != row["split"]:
            raise ValueError(f"Scenario family crosses splits: {family}")
        request = ReviewRequest.model_validate(row["request"])
        if not row["expected"]:
            raise ValueError("Every labeled case needs at least one decision")
        for check, label in row["expected"].items():
            if check not in DEFINITIONS:
                raise ValueError("Unsupported decision ID")
            Verdict(label)
            signature = json.dumps([request.model_dump(mode="json"), check], sort_keys=True)
            if signature in inputs and inputs[signature] != row["split"]:
                raise ValueError("Identical review input leaked across splits")
            inputs[signature] = row["split"]
    return {"cases": len(rows), "families": len(families),
            "splits": dict(Counter(row["split"] for row in rows)),
            "expert_reviewed": sum(expert_reviewed(row) for row in rows)}


def render_case(row, store):
    req = ReviewRequest.model_validate(row["request"])
    for check, expected in row["expected"].items():
        records = store.search(check, req.design.model_dump_json())
        yield {"id": row["id"], "scenario_family": row["scenario_family"],
               "state": build_state(req, check, records), "questions": question(check),
               "expected": {check: expected}, "evidence_ids": [r["id"] for r in records]}


def export_dataset(destination: Path, source: Path | None = None, allow_unreviewed=False):
    rows = load_cases(source)
    if not allow_unreviewed and any(not expert_reviewed(row) for row in rows):
        raise ValueError("Dataset awaits expert review. Use --allow-unreviewed only for an explicitly experimental export.")
    destination.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore()
    manifest = {"version": "export-1", "evidence_version": store.version,
                "expert_reviewed": all(expert_reviewed(row) for row in rows),
                "experimental": not all(expert_reviewed(row) for row in rows), "files": {},
                "family_splits": {row["scenario_family"]: row["split"] for row in rows}}
    prompts = []
    for split in SPLITS:
        items = [item for row in rows if row["split"] == split for item in render_case(row, store)]
        content = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
        (destination / f"{split}.jsonl").write_bytes(content.encode("utf-8"))
        manifest["files"][f"{split}.jsonl"] = {"sha256": hashlib.sha256(content.encode()).hexdigest(), "count": len(items)}
        prompts += [{"id": item["id"], "split": split, "state": item["state"], "questions": item["questions"]}
                    for item in items]
    # General-AI comparison inputs intentionally exclude gold labels and rationales.
    (destination / "general_ai_inputs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
