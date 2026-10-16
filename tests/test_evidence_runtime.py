import hashlib
import json

import pytest

from arch_review.evidence_v2 import PersistentEvidence


def lesson(identifier, checks, text):
    return {"id": identifier, "problem": text, "lesson": text, "conditions": [text],
            "tags": ["cache"], "checks": checks, "source_url": "https://example.org/lesson",
            "publication_date": None, "review_status": "author_reviewed", "license": "MIT",
            "provenance": "test fixture"}


def test_filtered_limit_does_not_lose_applicable_lessons(tmp_path, monkeypatch):
    # High-ranking unrelated lessons must not consume the requested result limit.
    rows = [lesson(f"unrelated-{i}", ["retry_safety"], "cache freshness bound") for i in range(40)]
    rows += [lesson("wanted-a", ["cache_freshness"], "cache"),
             lesson("wanted-b", ["cache_freshness"], "cache")]
    monkeypatch.setattr("arch_review.evidence_v2.read_jsonl",
                        lambda name: rows if name == "evidence.jsonl" else [])
    store = PersistentEvidence(tmp_path)
    assert [r["id"] for r in store.search("cache_freshness", {}, limit=2)] == ["wanted-a", "wanted-b"]
    assert len(store.search("cache_freshness", {}, limit=1)) == 1
    assert store.search("cache_freshness", {}, limit=0) == []
    expected = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
    assert store.version == expected
    assert PersistentEvidence(tmp_path).search("cache_freshness", {}, limit=2) == store.search("cache_freshness", {}, 2)


def test_streamed_extra_records_reject_duplicate_ids(tmp_path):
    extra = tmp_path / "extra.jsonl"
    row = lesson("duplicate", ["cache_freshness"], "cache")
    extra.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="globally unique"):
        PersistentEvidence(tmp_path / "index", extra)


def test_source_change_during_index_build_is_not_marked_complete(tmp_path, monkeypatch):
    calls = []
    def changing(name):
        if name != "evidence.jsonl":
            return []
        calls.append(name)
        return [lesson("same-id", ["cache_freshness"], "cache" if len(calls) == 1 else "changed")]
    monkeypatch.setattr("arch_review.evidence_v2.read_jsonl", changing)
    with pytest.raises(ValueError, match="changed while"):
        PersistentEvidence(tmp_path)
    # An interrupted index can be rebuilt; a false completion marker must not survive.
    monkeypatch.setattr("arch_review.evidence_v2.read_jsonl",
                        lambda name: [lesson("same-id", ["cache_freshness"], "cache")] if name == "evidence.jsonl" else [])
    assert PersistentEvidence(tmp_path).search("cache_freshness", {})[0]["lesson"] == "cache"


def test_equivalent_fact_order_does_not_change_bounded_search_terms(tmp_path, monkeypatch):
    rows = [lesson("wanted-first", ["cache_freshness"], "term000"),
            lesson("wanted-last", ["cache_freshness"], "term099")]
    monkeypatch.setattr("arch_review.evidence_v2.read_jsonl",
                        lambda name: rows if name == "evidence.jsonl" else [])
    store = PersistentEvidence(tmp_path)
    facts = {f"term{i:03d}": True for i in range(100)}
    forward = store.search("cache_freshness", facts)
    backward = store.search("cache_freshness", dict(reversed(list(facts.items()))))
    assert forward == backward
