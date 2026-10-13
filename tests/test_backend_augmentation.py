"""Train-only views inherit frozen labels and never create independent evidence."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from arch_review.evaluation_v2 import validate_backend_export

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "backend-diagnostics-v1"
SPEC = importlib.util.spec_from_file_location("augment_backend_training", ROOT / "scripts" / "augment_backend_training.py")
augmentation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(augmentation)


@pytest.fixture(scope="module")
def augmented(tmp_path_factory):
    directory = tmp_path_factory.mktemp("augmentation") / "export"
    augmentation.write_export(SOURCE, directory)
    return directory


def test_counts_and_held_out_bytes_remain_exactly_frozen(augmented):
    manifest, splits, cases, audit = validate_backend_export(augmented)
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 360, "development": 36, "calibration": 36, "test": 48}
    assert len(cases) == 480 and audit["families"] == 80
    assert (augmented / "train.jsonl").read_bytes().startswith((SOURCE / "train.jsonl").read_bytes())
    for split in augmentation.HELD_OUT:
        assert (augmented / (split + ".jsonl")).read_bytes() == (SOURCE / (split + ".jsonl")).read_bytes()
    original = json.loads((SOURCE / "manifest.json").read_text())
    assert manifest["family_splits"] == original["family_splits"]
    assert manifest["augmentation"]["source_manifest_sha256"] == augmentation.digest((SOURCE / "manifest.json").read_bytes())


def test_every_variant_inherits_labels_families_questions_and_exact_evidence(augmented):
    _, original_splits, original_cases, _ = validate_backend_export(SOURCE)
    _, splits, cases, _ = validate_backend_export(augmented)
    original_rows = {row["id"]: row for rows in original_splits.values() for row in rows}
    variants = []
    for row in splits["train"]:
        case = cases[row["id"]]
        if "augmentation" not in case:
            assert row == original_rows[row["id"]]
            assert case == original_cases[row["id"]]
            continue
        variants.append(case)
        parent_id = case["augmentation"]["parent_case_id"]
        original = original_rows[parent_id]
        parent = original_cases[parent_id]
        assert case["split"] == parent["split"] == "train"
        assert case["scenario_family"] == row["scenario_family"] == parent["scenario_family"]
        assert row["questions"] == original["questions"]
        assert row["expected"] == original["expected"] and row["answers"] == original["answers"]
        assert case["expected"] == parent["expected"]
        for field in ("check", "label_provenance", "review_status", "rationale", "source_groups", "critical"):
            assert case[field] == parent[field]
        state = json.loads(row["state"])
        assert state["evidence"] == json.loads(original["state"])["evidence"]
        assert state["design"] == case["description"]
        assert not any(label in row["state"] for label in augmentation.VERDICTS)
        if case["augmentation"]["representation"] == "json_facts_v1":
            assert json.loads(state["design"]) == parent["check"]["facts"]
        assert case["augmentation"]["new_independent_case"] is False
        assert case["augmentation"]["label_recomputed"] is False
    assert len(variants) == 240
    assert set(Counter(case["augmentation"]["parent_case_id"] for case in variants).values()) == {2}
    for split in augmentation.HELD_OUT:
        assert splits[split] == original_splits[split]
        assert all("augmentation" not in cases[row["id"]] for row in splits[split])


def test_prose_preserves_units_negation_and_missing_fields():
    _, rows, indexed, _ = validate_backend_export(SOURCE)
    chosen = indexed["diagnostic-retry_safety-00-insufficient_information"]
    prose = augmentation.prose_description(chosen)
    assert "7200 seconds" in prose and "3600 seconds" in prose
    assert "At least one business effect is not inherently idempotent." in prose
    assert "atomic commit" not in prose
    assert augmentation.sentence("design.retry.dedup_atomic_with_effect", False) == (
        "Duplicate detection and the protected effect do not share one atomic commit.")
    assert augmentation.sentence("per_attempt_timeout_ms", 1000000.0) == (
        "Each dependency attempt times out after 1000000.0 milliseconds.")
    assert augmentation.sentence("design.queue.max_messages", 300) == "The queue holds at most 300 messages."
    with pytest.raises(ValueError, match="do not guess"):
        augmentation.sentence("invented.guarantee", True)
    with pytest.raises(ValueError, match="explicit boolean"):
        augmentation.sentence("jitter", "true")
    held_out = rows["development"][0]
    with pytest.raises(ValueError, match="Only aligned training"):
        augmentation.make_variants(indexed[held_out["id"]], held_out)


def test_export_never_calls_rules_or_models_to_assign_labels(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Representation augmentation must not recompute labels")

    monkeypatch.setattr("arch_review.backend_dataset.evaluate_check", forbidden)
    monkeypatch.setattr("arch_review.rules_v2.evaluate_check", forbidden)
    manifest = augmentation.write_export(SOURCE, tmp_path / "export")
    assert manifest["augmentation"]["new_independent_cases"] == 0
    assert manifest["augmentation"]["original_source_cases"] == 240
    assert manifest["augmentation"]["added_training_representations"] == 240
    assert manifest["expert_reviewed"] is False
    assert manifest["quality_release_approved"] is False
    assert manifest["augmentation"]["exact_runtime_wrapper_used"] is False


def test_export_is_reproducible_and_immutable(augmented, tmp_path):
    other = tmp_path / "export"
    augmentation.write_export(SOURCE, other)
    for path in augmented.iterdir():
        assert path.read_bytes() == (other / path.name).read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        augmentation.write_export(SOURCE, other)


def test_published_augmented_snapshot_matches_reproduction(augmented):
    published = ROOT / "datasets" / "backend-augmented-v1"
    validate_backend_export(published)
    assert {path.name for path in published.iterdir()} == {path.name for path in augmented.iterdir()}
    for path in augmented.iterdir():
        assert path.read_bytes() == (published / path.name).read_bytes()
