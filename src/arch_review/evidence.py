import hashlib
import json
import re
import sqlite3
from importlib.resources import files


def data_path(name: str):
    return files("arch_review").joinpath("data", name)


def read_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in data_path(name).read_text(encoding="utf-8").splitlines() if line.strip()]


class EvidenceStore:
    """Small local full-text index. Evaluation cases are never indexed here."""

    def __init__(self, records=None):
        self.records = records if records is not None else read_jsonl("evidence.jsonl")
        self.version = hashlib.sha256(json.dumps(self.records, sort_keys=True).encode()).hexdigest()[:16]
        self.by_id = {r["id"]: r for r in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError("Duplicate evidence IDs")

    def search(self, check_id: str, text: str = "", limit=3) -> list[dict]:
        # Per-call connections keep SQLite access safe across HTTP worker threads.
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE VIRTUAL TABLE evidence USING fts5(id UNINDEXED, body)")
            candidates = [r for r in self.records if check_id in r.get("checks", [])]
            for r in candidates:
                db.execute("INSERT INTO evidence VALUES (?, ?)", (r["id"], " ".join([
                    r["problem"], r["lesson"], " ".join(r["tags"]), " ".join(r["conditions"])])))
            tokens = list(dict.fromkeys(re.findall(r"[a-zA-Z0-9_]+", text.lower())))[:32]
            selected = []
            if tokens:
                query = " OR ".join('"' + token + '"' for token in tokens)
                selected = [r[0] for r in db.execute(
                    "SELECT id FROM evidence WHERE evidence MATCH ? ORDER BY rank, id LIMIT ?", (query, limit))]
            # The check tag gives a useful baseline even when the design vocabulary differs.
            selected += [r["id"] for r in candidates if r["id"] not in selected]
        return [dict(self.by_id[key], relationship="background_lesson_not_deployment_proof")
                for key in selected[:limit]]
