"""Persistent, versioned lexical retrieval. Never index evaluation cases or answers."""
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from filelock import FileLock

from .evidence import read_jsonl
from .registry import REGISTRY


def validate_lessons(records):
    identifiers = set()
    required = {"id", "problem", "lesson", "conditions", "tags", "checks", "source_url", "publication_date",
                "review_status", "license", "provenance"}
    for row in records:
        if not required.issubset(row) or not row["source_url"].startswith("https://"):
            raise ValueError("Evidence requires source URL, date (null if unknown), provenance, license and applicability")
        if row["id"] in identifiers or set(row["checks"]) - REGISTRY.keys():
            raise ValueError("Duplicate evidence ID or unsupported check")
        if not row["provenance"] or row["license"] not in {"MIT", "CC-BY-3.0", "CC-BY-4.0", "Apache-2.0", "original_summary_link_only"}:
            raise ValueError("Unsupported or missing evidence provenance/reuse metadata")
        if {"expected", "answers", "split", "request"} & row.keys():
            raise ValueError("Evaluation/training cases must not be indexed as evidence")
        identifiers.add(row["id"])
    return records


class PersistentEvidence:
    def __init__(self, directory, extra=None):
        self.records = read_jsonl("evidence.jsonl")
        try:
            self.records += validate_lessons(read_jsonl("backend_evidence.jsonl"))
        except FileNotFoundError:
            pass
        if extra:
            self.records += validate_lessons([json.loads(line) for line in Path(extra).read_text(encoding="utf-8").splitlines() if line.strip()])
        self.by_id = {r["id"]: r for r in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError("Evidence IDs must be globally unique")
        self.version = hashlib.sha256(json.dumps(self.records, sort_keys=True).encode()).hexdigest()[:16]
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"evidence-{self.version}.sqlite3"
        with FileLock(str(self.path) + ".lock"):
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS lessons USING fts5(id UNINDEXED, checks UNINDEXED, body)")
                if db.execute("SELECT count(*) FROM lessons").fetchone()[0] != len(self.records):
                    db.execute("DELETE FROM lessons")
                    db.executemany("INSERT INTO lessons VALUES (?, ?, ?)", [
                        (r["id"], json.dumps(r["checks"]), " ".join([r["problem"], r["lesson"],
                            " ".join(r["conditions"]), " ".join(r["tags"])])) for r in self.records])

    def search(self, check_id, facts, limit=3):
        text = " ".join(REGISTRY[check_id].tags) + " " + json.dumps(facts)
        words = list(dict.fromkeys(re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower())))[:64]
        query = " OR ".join('"' + word + '"' for word in words)
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT id, checks FROM lessons WHERE lessons MATCH ? ORDER BY rank, id", (query,))
            selected = [key for key, checks in rows if check_id in json.loads(checks)][:limit]
        return [dict(self.by_id[key], relationship="background_lesson_not_deployment_proof") for key in selected]
