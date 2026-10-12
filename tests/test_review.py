import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arch_review.dataset import load_cases, validate_cases
from arch_review.evidence import EvidenceStore, read_jsonl
from arch_review.rules import check_rules
from arch_review.schema import ModelDecision, ReviewRequest, Verdict
from arch_review.service import ReviewService
from arch_review.settings import Settings
from arch_review.training import reserve_budget
from arch_review.workload import calculate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def complete():
    return ReviewRequest.model_validate_json((ROOT / "examples/complete.json").read_text())


@pytest.mark.parametrize("row", read_jsonl("regressions.jsonl"), ids=lambda r: r["id"])
def test_ten_contract_regressions(row):
    actual = {f.check_id: f.verdict.value for f in check_rules(ReviewRequest.model_validate(row["request"]))}
    for check, expected in row["expected"].items():
        assert actual[check] == expected


def test_atomic_database_dedup_does_not_cover_external_effect(complete):
    complete.design.retry.external_effects = True
    complete.design.retry.external_effects_idempotent = False
    assert check_rules(complete)[0].verdict == Verdict.CONTRADICTED


def test_dedup_retention_must_cover_maximum_retry_age(complete):
    complete.design.retry.dedup_retention_seconds = 60
    assert check_rules(complete)[0].verdict == Verdict.CONTRADICTED


def test_exactly_once_claim_in_notes_cannot_replace_missing_mechanisms():
    req = ReviewRequest(requirements={"prevent_duplicate_effects": True},
                        design={"notes": "Famous-company architecture with exactly-once delivery."})
    assert check_rules(req)[0].verdict == Verdict.INSUFFICIENT


def test_different_ack_boundary_is_not_a_durability_proof(complete):
    complete.design.acknowledgement.acknowledged_operation = "consumer_finished"
    assert check_rules(complete)[1].verdict == Verdict.INSUFFICIENT


def test_host_survival_does_not_imply_zone_survival(complete):
    complete.requirements.survive_failures.append("zone_loss")
    assert check_rules(complete)[1].verdict == Verdict.CONTRADICTED


def test_missing_queue_details_stay_unknown(complete):
    complete.design.queue.retry_backoff = None
    assert check_rules(complete)[2].verdict == Verdict.INSUFFICIENT


def test_arithmetic_preserves_units_and_storage_copy_semantics(complete):
    result, capacity = calculate(complete)
    estimates = result["estimates"]
    assert estimates["peak_messages"]["value"] == 500
    assert estimates["peak_deliveries"]["value"] == 1500
    assert estimates["daily_payload"]["value"] == 8_847_360_000
    assert estimates["retained_storage"]["value"] == 955_514_880_000
    complete.workload.average_recipients = 100
    second, _ = calculate(complete)
    assert second["estimates"]["retained_storage"]["value"] == estimates["retained_storage"]["value"]
    assert capacity["evidence_status"] == "unknown"
    assert capacity["supported_capacity"] is None


def test_unknown_inputs_do_not_get_invented_defaults():
    result, capacity = calculate(ReviewRequest(requirements={}, design={}))
    assert result["estimates"] == {}
    assert "replication_factor" in result["missing_inputs"]
    assert capacity["headroom_ratio"] is None


def test_zero_traffic_and_extreme_growth_return_finite_json(complete):
    complete.workload.average_messages_per_second = 0
    estimates, _ = calculate(complete)
    assert estimates["estimates"]["peak_messages"]["value"] == 0
    complete.workload.average_messages_per_second = 100
    complete.workload.monthly_growth_fraction = 1e100
    estimates, _ = calculate(complete)
    json.dumps(estimates, allow_nan=False)
    assert "projected_average_messages" not in estimates["estimates"]


def test_mismatched_benchmark_cannot_establish_capacity(complete):
    from arch_review.schema import Benchmark
    complete.deployment.benchmark = Benchmark(provenance="direct_measurement", source="internal-load-test-1",
        measured_messages_per_second=1000, p95_latency_ms=200, error_fraction=0,
        duration_seconds=600, workload_matches=False, configuration_matches=True)
    assert calculate(complete)[1]["evidence_status"] == "unknown"
    complete.deployment.benchmark.workload_matches = True
    assert calculate(complete)[1]["headroom_ratio"] == 2


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_numeric_input_rejected(value):
    with pytest.raises(ValidationError):
        ReviewRequest(requirements={}, design={}, workload={"average_messages_per_second": value})


def test_oversized_input_rejected():
    with pytest.raises(ValidationError):
        ReviewRequest(requirements={}, design={"flows": ["x" * 8000] * 20})


class DisagreeingWorker:
    metadata = {"role": "test_double", "calibration_status": "unvalidated"}
    pid = None
    def predict(self, payloads):
        return {key: ModelDecision(verdict=Verdict.SUPPORTED, checkpoint="test-only") for key in payloads}, {}
    def close(self):
        pass


def test_model_cannot_erase_a_concrete_contradiction(complete):
    complete.design.retry.dedup_enforced = False
    service = ReviewService(worker=DisagreeingWorker())
    result = service.review(complete)
    assert result.checks[0].verdict == Verdict.CONTRADICTED
    assert result.checks[0].conflict
    assert result.review_status == "partial"


def test_model_cannot_approve_missing_required_facts(complete):
    complete.design.retry.dedup_enforced = None
    result = ReviewService(worker=DisagreeingWorker()).review(complete)
    assert result.checks[0].verdict == Verdict.INSUFFICIENT


def test_missing_model_is_explicitly_partial(complete):
    service = ReviewService(Settings())
    result = service.review(complete)
    assert result.review_mode == "rules_only"
    assert result.review_status == "partial"
    assert any("No Laya checkpoint" in warning for warning in result.warnings)
    assert all(f.model_decision is None for f in result.checks)


def test_evidence_provenance_and_fts_query_safety():
    store = EvidenceStore()
    records = store.search("retry_safety", '" OR * DROP TABLE evidence; --')
    assert records
    assert len(store.records) == 60
    assert len({r["source_id"] for r in store.records}) == 20
    assert all("expected" not in r and r["review_status"] == "pending_expert" for r in records)
    assert all(r["source_url"].startswith("https://") for r in records)


def test_scenario_variants_cannot_cross_splits():
    rows = load_cases()
    assert len(rows) == 100
    tampered = copy.deepcopy(rows)
    tampered[1]["split"] = "test"
    with pytest.raises(ValueError, match="crosses splits"):
        validate_cases(tampered)


def test_budget_reserves_overhead_and_rejects_overspend(tmp_path):
    first = reserve_budget(10, 3600, tmp_path)
    assert first["maximum_compute_usd"] == 10
    reserve_budget(10, 3600, tmp_path)
    with pytest.raises(ValueError, match="exceed"):
        reserve_budget(.4, 3600, tmp_path)


def test_training_export_requires_explicit_unreviewed_mode(tmp_path):
    from arch_review.dataset import export_dataset
    from arch_review.training import validate_export
    with pytest.raises(ValueError, match="expert review"):
        export_dataset(tmp_path)
    manifest = export_dataset(tmp_path, allow_unreviewed=True)
    assert manifest["experimental"]
    assert '"expected"' not in (tmp_path / "general_ai_inputs.jsonl").read_text()
    validate_export(tmp_path, allow_unreviewed=True)
    with pytest.raises(ValueError, match="reviewed data"):
        validate_export(tmp_path)
    with (tmp_path / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_export(tmp_path, allow_unreviewed=True)
