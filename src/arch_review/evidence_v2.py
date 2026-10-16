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
        # Hash the same canonical list as older versions, without retaining source
        # bodies in memory. Only the small identifier set grows with the corpus.
        self.version, count = self._fingerprint(extra)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"evidence-{self.version}-v2.sqlite3"
        with FileLock(str(self.path) + ".lock"):
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS lessons USING fts5(id UNINDEXED, body)")
                db.execute("CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, record TEXT NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS lesson_checks(check_id TEXT, id TEXT, PRIMARY KEY(check_id,id))")
                db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                stored = db.execute("SELECT value FROM metadata WHERE key='complete_version'").fetchone()
                if not stored or stored[0] != self.version or db.execute("SELECT count(*) FROM records").fetchone()[0] != count:
                    db.execute("DELETE FROM lessons")
                    db.execute("DELETE FROM records")
                    db.execute("DELETE FROM lesson_checks")
                    digest = hashlib.sha256(b"[")
                    inserted = 0
                    for row in self._records(extra):
                        encoded = json.dumps(row, sort_keys=True)
                        if inserted:
                            digest.update(b", ")
                        digest.update(encoded.encode())
                        db.execute("INSERT INTO records VALUES (?,?)", (row["id"], encoded))
                        db.execute("INSERT INTO lessons VALUES (?,?)", (row["id"], " ".join([
                            row["problem"], row["lesson"], " ".join(row["conditions"]), " ".join(row["tags"])])))
                        db.executemany("INSERT INTO lesson_checks VALUES (?,?)",
                                       ((check, row["id"]) for check in set(row["checks"])))
                        inserted += 1
                    digest.update(b"]")
                    if digest.hexdigest()[:16] != self.version or inserted != count:
                        raise ValueError("Evidence changed while its index was being built; retry with a stable source")
                    db.execute("INSERT OR REPLACE INTO metadata VALUES ('complete_version',?)", (self.version,))

    @staticmethod
    def _records(extra):
        yield from read_jsonl("evidence.jsonl")
        try:
            yield from validate_lessons(read_jsonl("backend_evidence.jsonl"))
        except FileNotFoundError:
            pass
        if extra:
            with Path(extra).open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield from validate_lessons([json.loads(line)])

    @classmethod
    def _fingerprint(cls, extra):
        identifiers = set()
        digest = hashlib.sha256(b"[")
        for row in cls._records(extra):
            if row["id"] in identifiers:
                raise ValueError("Evidence IDs must be globally unique")
            if identifiers:
                digest.update(b", ")
            identifiers.add(row["id"])
            digest.update(json.dumps(row, sort_keys=True).encode())
        digest.update(b"]")
        return digest.hexdigest()[:16], len(identifiers)

    def search(self, check_id, facts, limit=3):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Evidence limit must be a nonnegative integer")
        if limit == 0:
            return []
        text = " ".join(REGISTRY[check_id].tags) + " " + json.dumps(facts, sort_keys=True)
        words = list(dict.fromkeys(re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower())))[:64]
        query = " OR ".join('"' + word + '"' for word in words)
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                "SELECT records.record FROM lessons "
                "JOIN lesson_checks ON lesson_checks.id=lessons.id "
                "JOIN records ON records.id=lessons.id "
                "WHERE lessons MATCH ? AND lesson_checks.check_id=? "
                "ORDER BY lessons.rank, lessons.id LIMIT ?", (query, check_id, limit))
            return [dict(json.loads(row[0]), relationship="background_lesson_not_deployment_proof") for row in rows]
