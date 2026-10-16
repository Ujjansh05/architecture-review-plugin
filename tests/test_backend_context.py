import json
import time

import pytest

from arch_review.backend_dataset import positive_facts
from arch_review.backends import LlamaBackend, NoModelBackend
from arch_review.documents import extract_documents, fact_fields, validate_spans
from arch_review.jobs import ReviewJobs
from arch_review.rules_v2 import evaluate_check
from arch_review.schema_v2 import CheckInput, DesignReviewRequest, Document, Span
from arch_review.settings import Settings
from arch_review.training import reserve_budget


class RequirementsExtractor:
    def tokenize(self, text):
        return len(text.split())

    def complete(self, instructions, data, schema):
        if data["document_id"] == "__requirements__":
            return {"facts": [{"check_id": "cache_freshness", "path": "max_staleness_ms", "value": 100,
                               "quote": data["section"]}]}
        return {"facts": [{"check_id": "cache_freshness", "scope_id": "catalog", "path": "worst_case_staleness_ms",
                           "value": 500, "quote": data["section"]}]}


def test_requirements_are_cited_and_apply_to_named_flows():
    requirement = json.dumps({"check_id": "cache_freshness", "scope_id": "design", "facts": {"max_staleness_ms": 100}})
    implementation = json.dumps({"check_id": "cache_freshness", "scope_id": "catalog",
                                 "facts": {"worst_case_staleness_ms": 500}})
    request = DesignReviewRequest(requirements=requirement,
        documents=[Document(id="implementation", text=implementation)],
        requested_checks=["cache_freshness"])
    items, coverage, _ = extract_documents(request, RequirementsExtractor())
    flow = next(item for item in items if item.scope_id == "catalog")
    assert flow.facts["max_staleness_ms"] == 100
    assert evaluate_check(flow).verdict == "contradicted"
    assert not coverage["unprocessed"]
    span = flow.fact_spans["max_staleness_ms"][0]
    assert span.document_id == "__requirements__"
    assert request.requirements[span.start:span.end] == request.requirements
    validate_spans(request.model_copy(update={"checks": [flow]}))


def test_free_text_requirements_are_not_silently_ignored_in_rules_mode():
    request = DesignReviewRequest(requirements="Values must be at most 5 ms old.",
        checks=[CheckInput(check_id="cache_freshness", facts=positive_facts("cache_freshness"))])
    _, coverage, _ = extract_documents(request, NoModelBackend())
    assert "__requirements__" in coverage["unprocessed"]


def test_reserved_requirements_id_and_source_bounds():
    with pytest.raises(ValueError, match="reserved"):
        DesignReviewRequest(documents=[Document(id="__requirements__", text="fake")])
    request = DesignReviewRequest(requirements="Short", checks=[CheckInput(check_id="cache_freshness",
        facts={"max_staleness_ms": 10}, fact_spans={"max_staleness_ms": [Span(document_id="__requirements__", start=0, end=20)]})])
    with pytest.raises(ValueError, match="outside"):
        validate_spans(request)


def test_schemas_are_scoped_and_global_implementation_not_inherited():
    fields = fact_fields("retry_safety")
    assert "requirements.prevent_duplicate_effects" in fields
    assert "design.retry.dedup_atomic_with_effect" in fields
    assert not any("queue" in key or "workload" in key for key in fields)
    class ScopedExtractor(RequirementsExtractor):
        def complete(self, instructions, data, schema):
            return {"facts": [{"check_id": "cache_freshness", "scope_id": "other-flow",
                               "path": "max_staleness_ms", "value": 100, "quote": "Second flow exists."}]}
    request = DesignReviewRequest(checks=[CheckInput(check_id="cache_freshness", facts=positive_facts("cache_freshness"))],
        documents=[Document(id="d", text="Second flow exists.")], requested_checks=["cache_freshness"])
    items, _, _ = extract_documents(request, ScopedExtractor())
    other = next(item for item in items if item.scope_id == "other-flow")
    assert "worst_case_staleness_ms" not in other.facts
    assert evaluate_check(other).verdict == "insufficient_information"


def test_internally_inconsistent_read_guarantees_are_unknown():
    facts = positive_facts("replica_consistency")
    facts["stale_reads_possible"] = True
    assert evaluate_check(CheckInput(check_id="replica_consistency", facts=facts)).verdict == "insufficient_information"


def test_long_document_cancellation_checked_before_tokenization():
    request = DesignReviewRequest(documents=[Document(id="d", text="large document")])
    with pytest.raises(InterruptedError):
        extract_documents(request, RequirementsExtractor(), cancelled=lambda: True)


def test_model_handoff_precedes_server_access(monkeypatch):
    calls = []
    backend = LlamaBackend(Settings(), before_start=lambda: calls.append("unload_laya"))
    monkeypatch.setattr(backend, "_ready", lambda: calls.append("server_ready") or True)
    backend._start()
    assert calls == ["unload_laya", "server_ready"]


def test_job_deadline_is_failed_and_completed_results_expire():
    class Slow:
        backend = NoModelBackend()
        def review(self, request, cancelled, progress):
            time.sleep(.03)
            cancelled()
    jobs = ReviewJobs(Slow(), review_timeout_seconds=.01, ttl_seconds=.05)
    try:
        job = jobs.submit(None)
        limit = time.monotonic() + 2
        while time.monotonic() < limit:
            result = jobs.get(job["job_id"])
            if result["status"] == "failed":
                break
            time.sleep(.002)
        assert result["status"] == "failed" and "time budget" in result["error"]
        time.sleep(.15)
        with jobs.lock:
            assert job["job_id"] not in jobs.jobs  # expiry happens without a GET
    finally:
        jobs.close()


def test_existing_budget_ledger_obeys_reduced_fifteen_dollar_cap(tmp_path):
    (tmp_path / "training-budget.json").write_text('{"cap_usd":25,"reserve_usd":5,"runs":[{"maximum_compute_usd":9}]}')
    reserve_budget(1, 3600, tmp_path)
    with pytest.raises(ValueError, match="USD 15"):
        reserve_budget(.01, 3600, tmp_path)
