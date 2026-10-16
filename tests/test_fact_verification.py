import json

import pytest

from arch_review.backend_dataset import positive_facts
from arch_review.backends import NoModelBackend
from arch_review.documents import extract_documents, fact_fields, flatten
from arch_review.evidence_v2 import PersistentEvidence
from arch_review.review_v2 import DesignReviewer
from arch_review.schema_v2 import DesignReviewRequest

PATH = "bound_covers_failures"


def supplied(text, scope="catalog", *, quoted=None, origin="host_extracted"):
    facts = positive_facts("cache_freshness")
    return DesignReviewRequest(input_mode="supplied_facts", requested_checks=["cache_freshness"],
        documents=[{"id": "design-notes", "text": text}], checks=[{
            "check_id": "cache_freshness", "scope_id": scope, "facts": facts,
            "fact_origins": {p: origin if p == PATH else "user_declared" for p in facts},
            "fact_quotes": {PATH: [{"document_id": "design-notes", "quote": quoted or text}]}}])


def declaration(value=True, scope="catalog", check="cache_freshness"):
    return {"check_id": check, "scope_id": scope, "facts": {PATH: value}}


def run(tmp_path, request, backend=None):
    return DesignReviewer(backend or NoModelBackend(), PersistentEvidence(tmp_path)).review(request)


def test_valid_quote_cannot_validate_wrong_meaning_or_reach_model(tmp_path):
    class Spy(NoModelBackend):
        def judge(self, check, facts, evidence, **kwargs):
            assert PATH not in facts
            return {"verdict": "supported"}
    source = "Prices may be arbitrarily old while the database is unavailable."
    result = run(tmp_path, supplied(source), Spy())
    finding = result.checks[0]
    assert finding.verdict == "insufficient_information"
    assert PATH not in finding.facts
    assert finding.fact_origins[PATH] == "host_extracted"
    proposal = result.coverage["unverified_facts"][0]
    assert proposal["proposed_value"] is True and proposal["status"] == "unverified"
    assert finding.supporting_spans[PATH][0].document_id == "design-notes"
    question = next(q for q in finding.clarification_questions if q["fact_path"] == PATH)
    assert question["requires_confirmation"] and question["proposed_facts"] == [proposal]


def test_explicit_user_declaration_stays_a_declaration(tmp_path):
    result = run(tmp_path, supplied("External supporting discussion.", origin="user_declared"))
    assert result.checks[0].verdict == "supported"
    assert not result.coverage["unverified_facts"]
    assert result.coverage["fact_verification"]["deployment_verified"] is False


def test_complete_typed_declaration_inside_json_source_is_matched(tmp_path):
    quoted = json.dumps(declaration())
    text = '{"checks": [' + quoted + ', ' + json.dumps(declaration(scope="other")) + ']}'
    result = run(tmp_path, supplied(text, quoted=quoted))
    assert result.checks[0].verdict == "supported"
    assert result.checks[0].fact_origins[PATH] == "host_extracted"
    assert result.coverage["fact_verification"]["source_matched"][0]["status"] == "source_matched"
    assert not result.coverage["unverified_facts"]


@pytest.mark.parametrize("record", [declaration(scope="payments"), declaration(check="failover_safety"),
                                    declaration(value=False), declaration(value=1), declaration(value="true")])
def test_scope_check_value_and_types_cannot_be_substituted(tmp_path, record):
    result = run(tmp_path, supplied(json.dumps(record)))
    assert result.checks[0].verdict == "insufficient_information"
    assert PATH not in result.checks[0].facts
    assert result.coverage["unverified_facts"]


def test_numeric_declaration_does_not_accept_boolean_as_integer(tmp_path):
    record = {"check_id": "cache_freshness", "scope_id": "catalog", "facts": {"max_staleness_ms": True}}
    request = supplied(json.dumps(record))
    check = request.checks[0]
    check.facts["max_staleness_ms"] = 1
    check.fact_origins[PATH] = "user_declared"
    check.fact_origins["max_staleness_ms"] = "host_extracted"
    check.fact_quotes["max_staleness_ms"] = check.fact_quotes.pop(PATH)
    result = run(tmp_path, request)
    assert "max_staleness_ms" not in result.checks[0].facts
    assert result.checks[0].verdict == "insufficient_information"


def test_duplicate_keys_and_json_embedded_in_prose_remain_unverified(tmp_path):
    for source in ('{"check_id":"cache_freshness","scope_id":"catalog","facts":'
                   '{"bound_covers_failures":false,"bound_covers_failures":true}}',
                   "Example only, not our architecture: " + json.dumps(declaration())):
        result = run(tmp_path, supplied(source))
        assert result.checks[0].verdict == "insufficient_information"
        assert result.coverage["unverified_facts"][0]["status"] == "unverified"


def test_conflicting_source_declarations_are_not_cherry_picked(tmp_path):
    quoted = json.dumps(declaration())
    source = '[' + quoted + ', ' + json.dumps(declaration(False)) + ']'
    result = run(tmp_path, supplied(source, quoted=quoted))
    assert result.checks[0].verdict == "insufficient_information"
    assert result.checks[0].conflict
    assert result.coverage["unverified_facts"][0]["status"] == "conflicting"


def test_uncited_conflicting_structured_source_also_blocks_matching_fact(tmp_path):
    request = supplied(json.dumps(declaration()))
    from arch_review.schema_v2 import Document
    request.documents.append(Document(id="other-notes", text=json.dumps(declaration(False))))
    result = run(tmp_path, request)
    assert result.checks[0].conflict and result.checks[0].verdict == "insufficient_information"


def test_no_extractor_and_legacy_mode_cannot_bypass_verification(tmp_path):
    request = supplied("A cache exists.")
    request.input_mode = "extract_documents"
    result = run(tmp_path, request)
    assert PATH not in result.checks[0].facts
    assert result.checks[0].verdict == "insufficient_information"


def test_legacy_nested_schema_does_not_coerce_numeric_protection_to_boolean(tmp_path):
    facts = positive_facts("retry_safety")
    path = "design.retry.dedup_enforced"
    facts["design"]["retry"]["dedup_enforced"] = 1
    declaration = json.dumps({"check_id": "retry_safety", "scope_id": "design", "facts": facts})
    request = DesignReviewRequest(input_mode="supplied_facts", requested_checks=["retry_safety"],
        documents=[{"id": "notes", "text": declaration}], checks=[{"check_id": "retry_safety", "facts": facts,
        "fact_origins": {p: "host_extracted" if p == path else "user_declared"
                         for p in flatten(facts) if p in fact_fields("retry_safety")},
        "fact_quotes": {path: [{"document_id": "notes", "quote": declaration}]}}])
    result = run(tmp_path, request)
    assert result.checks[0].verdict == "insufficient_information"
    assert "dedup_enforced" not in result.checks[0].facts["design"]["retry"]


def test_machine_document_extraction_is_proposal_only_for_prose():
    class Extractor:
        def tokenize(self, text):
            return len(text.split())
        def complete(self, instructions, data, schema):
            return {"facts": [{"check_id": "cache_freshness", "scope_id": "catalog",
                               "path": PATH, "value": True, "quote": data["section"]}]}
    request = DesignReviewRequest(documents=[{"id": "notes", "text": "Use Redis."}],
                                  requested_checks=["cache_freshness"])
    items, coverage, _ = extract_documents(request, Extractor())
    assert items[0].facts == {}
    assert coverage["unverified_facts"][0]["proposed_value"] is True


def test_stage_timings_are_present_and_nonnegative(tmp_path):
    result = run(tmp_path, supplied(json.dumps(declaration())))
    timing = result.coverage["timings_ms"]
    assert set(timing) == {"validate_facts", "extract_and_verify", "rules", "evidence_retrieval",
                           "model", "workload", "aggregate"}
    assert all(value >= 0 for value in timing.values())
    assert result.coverage["model_metrics"]["cuda_peak_allocated_bytes"] is None


def test_worker_allocator_peaks_are_aggregated_without_gpu_probing(tmp_path):
    class MeasuredBackend(NoModelBackend):
        calls = 0
        def judge(self, check, facts, evidence, **kwargs):
            self.calls += 1
            return {"verdict": "supported"}
        def metadata(self):
            return {"runtime_metrics": {"startup": {"cuda_peak_allocated_bytes": 400},
                    "last_request": {"cuda_peak_allocated_bytes": 1000 // max(1, self.calls),
                                     "cuda_peak_reserved_bytes": 2000 // max(1, self.calls)}}}
    request = supplied(json.dumps(declaration()))
    second = request.checks[0].model_copy(deep=True)
    second.scope_id = "other"
    second.fact_origins[PATH] = "user_declared"
    request.checks.append(second)
    result = run(tmp_path, request, MeasuredBackend())
    assert result.coverage["model_metrics"]["cuda_peak_allocated_bytes"] == 1000
    assert result.coverage["model_metrics"]["cuda_peak_reserved_bytes"] == 2000
