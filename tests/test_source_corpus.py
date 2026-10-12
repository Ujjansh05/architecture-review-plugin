import copy
import hashlib
import json
from pathlib import Path

import pytest

from arch_review.dataset import load_cases, render_case, validate_cases
from arch_review.evidence import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]


def test_source_corpus_integrity_and_attribution():
    directory = ROOT / "datasets/open-reference-v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    sources = {row["id"]: row for row in json.loads((directory / "sources.json").read_text(encoding="utf-8"))}
    chunks = [json.loads(line) for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(chunks) == manifest["chunks"]
    assert len(sources) == manifest["documents"]
    for row in chunks:
        assert row["source_id"] in sources
        assert row["authors"] and row["source_url"]
        assert row["license"] in {"CC-BY-3.0", "CC-BY-4.0"}
        assert hashlib.sha256(row["text"].encode()).hexdigest() == row["text_sha256"]
        assert row["end_character"] - row["start_character"] == len(row["text"])
        assert row["usage"] == "reference_only_not_gold_label"


def test_source_cases_keep_variants_and_source_documents_together():
    directory = ROOT / "datasets/source-linked-v1"
    rows = load_cases(directory / "cases.jsonl")
    assert validate_cases(rows)["expert_reviewed"] == 0
    modified = copy.deepcopy(rows)
    modified[-1]["source_group"] = modified[0]["source_group"]
    with pytest.raises(ValueError, match="Source document crosses"):
        validate_cases(modified)


def test_model_input_excludes_label_and_reviewer_metadata():
    row = copy.deepcopy(load_cases(ROOT / "datasets/source-linked-v1/cases.jsonl")[0])
    row["rationale"] = "PRIVATE_GOLD_REASON_SENTINEL"
    row["reviewer"] = "PRIVATE_REVIEWER_SENTINEL"
    rendered = list(render_case(row, EvidenceStore()))[0]
    assert "PRIVATE_" not in rendered["state"]
    evidence = json.loads(rendered["state"])["evidence"]
    assert evidence[0] == row["background_evidence"][0]
    bad = copy.deepcopy(row)
    bad["background_evidence"][0]["expected"] = "supported"
    with pytest.raises(ValueError, match="Background evidence"):
        validate_cases([bad])
