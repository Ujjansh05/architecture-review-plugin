import json
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from arch_review.api import create_app
from arch_review.backend_dataset import audit_cases, make_diagnostic_cases, positive_facts
from arch_review.backends import LlamaBackend, NoModelBackend
from arch_review.documents import chunks, extract_documents
from arch_review.evaluation_v2 import summarize
from arch_review.evidence_v2 import PersistentEvidence, validate_lessons
from arch_review.jobs import ReviewJobs
from arch_review.registry import REGISTRY
from arch_review.review_v2 import DesignReviewer
from arch_review.rules_v2 import evaluate_check
from arch_review.schema_v2 import CheckInput, DesignReviewRequest, Document
from arch_review.settings import Settings


@pytest.fixture
def evidence(tmp_path):
    return PersistentEvidence(tmp_path / "index")


@pytest.mark.parametrize("check", list(REGISTRY))
def test_diagnostic_decisions_and_unknowns(check):
    assert evaluate_check(CheckInput(check_id=check, facts=positive_facts(check))).verdict == "supported"
    assert evaluate_check(CheckInput(check_id=check, facts={})).verdict == "insufficient_information"
    skipped = evaluate_check(CheckInput(check_id=check, applicable=False))
    assert skipped.evaluation_status == "not_evaluated" and skipped.verdict is None


def test_dataset_has_all_checks_labels_and_exposes_weak_holdout():
    rows = make_diagnostic_cases()
    audit = audit_cases(rows)
    assert len(rows) == 240 and audit["split_counts"] == {"train": 120, "development": 36, "calibration": 36, "test": 48}
    for split in audit["coverage"].values():
        assert all(set(counts) == {"supported", "contradicted", "insufficient_information"} for counts in split.values())
    assert not audit["release_eligible"] and not audit["final_test_minimum_counts_met"]
    rows[0]["source_groups"], rows[-1]["source_groups"] = ["same-incident"], ["same-incident"]
    with pytest.raises(ValueError, match="Source group"):
        audit_cases(rows)


def test_actual_database_failure_schedule_preserves_only_transactional_changes():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE balances (id INTEGER PRIMARY KEY, value INTEGER)")
    db.executemany("INSERT INTO balances VALUES (?,?)", [(1, 100), (2, 100)])
    db.commit()
    db.execute("UPDATE balances SET value=value-10 WHERE id=1")
    db.rollback()  # crash/failure before completing the operation
    assert db.execute("SELECT sum(value) FROM balances").fetchone()[0] == 200
    db.execute("UPDATE balances SET value=value-10 WHERE id=1")
    db.commit()  # independent write persisted before the second write
    assert db.execute("SELECT sum(value) FROM balances").fetchone()[0] == 190
    db.close()


def test_component_names_do_not_establish_guarantees():
    with pytest.raises(ValueError):
        CheckInput(check_id="transaction_atomicity", facts={"database": "PostgreSQL"})
    result = evaluate_check(CheckInput(check_id="transaction_atomicity", facts={"boundary": "single_transaction"}))
    assert result.verdict == "insufficient_information"


def test_saga_is_not_assumed_to_provide_atomic_visibility():
    facts = positive_facts("transaction_atomicity")
    facts["boundary"] = "saga"
    assert evaluate_check(CheckInput(check_id="transaction_atomicity", facts=facts)).verdict == "insufficient_information"


def test_cache_error_path_and_deadline_edge():
    facts = positive_facts("cache_freshness")
    facts.update(stale_on_error=True, stale_on_error_bound_ms=2000)
    assert evaluate_check(CheckInput(check_id="cache_freshness", facts=facts)).verdict == "contradicted"
    facts = positive_facts("dependency_resilience")
    facts["deadline_ms"] = 500
    assert evaluate_check(CheckInput(check_id="dependency_resilience", facts=facts)).verdict == "supported"
    facts.update(deadline_ms=499, deadline_propagated=False)
    assert evaluate_check(CheckInput(check_id="dependency_resilience", facts=facts)).verdict == "contradicted"


def test_document_chunks_cover_every_character():
    doc = Document(id="long", text=("# Topic\n\nUnicode 東京 🙂 and design details.\n\n" * 500))
    covered = set()
    for start, end, text in chunks(doc, size=100, overlap=10):
        assert text == doc.text[start:end]
        covered.update(range(start, end))
    assert len(covered) == len(doc.text)


class Extractor:
    name = "test_double"
    def tokenize(self, text):
        return len(text.split())
    def complete(self, instructions, data, schema):
        section = data["section"]
        value = "old writer can commit" in section
        return {"facts": [{"check_id": "failover_safety", "path": "old_writer_can_commit_after_promotion",
                           "value": value, "quote": section}]}
    def metadata(self):
        return {"name": self.name}
    def close(self):
        pass


def test_distant_conflicts_are_not_hidden_by_merge(evidence):
    req = DesignReviewRequest(documents=[Document(id="a", text="old writer can commit"),
                                        Document(id="b", text="old writer cannot commit")],
                              requested_checks=["failover_safety"])
    items, coverage, conflicts = extract_documents(req, Extractor())
    assert not coverage["unprocessed"]
    assert "old_writer_can_commit_after_promotion" in conflicts[("design", "failover_safety")]
    assert "old_writer_can_commit_after_promotion" not in items[0].facts
    result = DesignReviewer(Extractor(), evidence).review(req)
    assert result.checks[0].conflict and result.checks[0].verdict == "insufficient_information"


def test_invalid_citation_and_omitted_documents_prevent_approval(evidence):
    class BadExtractor(Extractor):
        def complete(self, *args, **kwargs):
            return {"facts": [{"check_id": "failover_safety", "path": "single_writer_required", "value": True,
                               "quote": "invented passage"}]}
    req = DesignReviewRequest(documents=[Document(id="a", text="a real passage")], requested_checks=["failover_safety"],
        checks=[CheckInput(check_id="failover_safety", facts=positive_facts("failover_safety"))])
    for backend in (BadExtractor(), NoModelBackend()):
        result = DesignReviewer(backend, evidence).review(req)
        assert result.coverage["unprocessed"] and result.checks[0].verdict == "insufficient_information"


def test_model_cannot_overrule_concrete_failure(evidence):
    class Approver(NoModelBackend):
        def judge(self, *args):
            return {"verdict": "supported", "probabilities": None}
    facts = positive_facts("failover_safety")
    facts["old_writer_can_commit_after_promotion"] = True
    req = DesignReviewRequest(checks=[CheckInput(check_id="failover_safety", facts=facts)], requested_checks=["failover_safety"])
    result = DesignReviewer(Approver(), evidence).review(req)
    assert result.checks[0].verdict == "contradicted" and result.checks[0].conflict
    assert result.capacity["evidence_status"] == "unknown"


def test_persistent_index_has_each_check_and_rejects_case_import(evidence):
    for check in REGISTRY:
        assert evidence.search(check, positive_facts(check))
    record = evidence.search("cache_freshness", {})[0]
    record["expected"] = "supported"
    with pytest.raises(ValueError, match="Evaluation"):
        validate_lessons([record])
    reopened = PersistentEvidence(evidence.path.parent)
    assert reopened.version == evidence.version


def test_models_must_be_loopback_and_no_self_reported_probabilities():
    with pytest.raises(ValueError, match="local HTTP"):
        LlamaBackend(Settings(llama_endpoint="https://example.com"))
    assert LlamaBackend(Settings()).metadata()["probabilities_available"] is False


def test_failed_predictions_are_not_correct_unknown_answers():
    rows = [{"check_id": "cache_freshness", "scenario_family": str(i), "expected": "insufficient_information",
             "predicted": None, "error": "timeout", "critical": False} for i in range(2)]
    report = summarize(rows, {})
    assert report["overall"]["accuracy"] == 0 and not any(report["numeric_gates"].values())
    assert not report["independent_quality_release_approved"]


def test_jobs_bound_queue_cancel_and_release_resources(evidence):
    started, release = threading.Event(), threading.Event()
    class SlowReviewer(DesignReviewer):
        def review(self, request, cancelled, progress):
            started.set()
            release.wait(3)
            if cancelled():
                raise InterruptedError()
            return super().review(request, cancelled, progress)
    manager = ReviewJobs(SlowReviewer(NoModelBackend(), evidence), max_pending=1)
    req = DesignReviewRequest(checks=[CheckInput(check_id="cache_freshness")])
    try:
        job = manager.submit(req)
        assert started.wait(2)
        with pytest.raises(RuntimeError, match="queue"):
            manager.submit(req)
        assert manager.cancel(job["job_id"])["status"] == "cancelling"
        release.set()
        deadline = time.monotonic() + 3
        while manager.get(job["job_id"])["status"] == "cancelling" and time.monotonic() < deadline:
            time.sleep(.01)
        assert manager.get(job["job_id"])["status"] == "cancelled"
    finally:
        release.set()
        manager.close()


def test_v2_http_auth_jobs_and_original_size_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH_REVIEW_HOME", str(tmp_path))
    headers = {"Authorization": "Bearer test-v2"}
    payload = {"checks": [{"check_id": "cache_freshness", "facts": positive_facts("cache_freshness")}],
               "requested_checks": ["cache_freshness"]}
    with TestClient(create_app(Settings(), "test-v2")) as client:
        assert client.post("/v2/reviews", json=payload).status_code == 401
        assert client.post("/v1/reviews", headers=headers, content="x" * 100001).status_code == 413
        assert client.post("/v2/reviews", headers=headers, content="x" * (2 * 1024 * 1024 + 1)).status_code == 413
        response = client.post("/v2/reviews", headers=headers, json=payload)
        assert response.status_code == 202
        key = response.json()["job_id"]
        for _ in range(100):
            job = client.get("/v2/reviews/" + key, headers=headers).json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(.01)
        assert job["status"] == "completed", json.dumps(job)
        assert job["result"]["checks"][0]["verdict"] == "supported"
        assert len(client.get("/v2/schema", headers=headers).json()["decisions"]) == 8
        assert client.get("/v2/reviews/missing", headers=headers).status_code == 404
def test_evidence_closes_connections_after_build_and_search(tmp_path, monkeypatch):
    connections = []
    connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr("arch_review.evidence_v2.sqlite3.connect", tracked_connect)
    evidence = PersistentEvidence(tmp_path / "closed-index")
    assert evidence.search("retry_safety", {})
    assert len(connections) == 2
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
