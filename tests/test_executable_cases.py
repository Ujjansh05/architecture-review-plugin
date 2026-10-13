"""Exercise real local counterexamples; no model or plugin rule oracle is used."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

from arch_review.schema_v2 import DesignReviewRequest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "build_executable_cases.py"
SPEC = importlib.util.spec_from_file_location("build_executable_cases", SOURCE)
packet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packet)


def test_actual_unique_constraint_prevents_duplicate_counter_effects():
    good = packet.retry_observation(True)
    flawed = packet.retry_observation(False)
    assert good["effect_count"] == good["ledger_rows"] == 1
    assert good["deliveries"][1]["action"] == "duplicate_rejected_by_unique_constraint"
    assert flawed["effect_count"] == flawed["ledger_rows"] == 2
    assert good["constraint_satisfied"] is True
    assert flawed["constraint_satisfied"] is False


def test_actual_process_exit_preserves_committed_append_but_loses_uncommitted_ack():
    good = packet.ack_observation(True)
    flawed = packet.ack_observation(False)
    assert good["child_exit_code"] == flawed["child_exit_code"] == packet.CRASH_EXIT_CODE
    assert good["acknowledgement"]["event"] == flawed["acknowledgement"]["event"] == "acknowledged"
    assert good["persisted_after_process_exit"] == ["append-1"]
    assert flawed["persisted_after_process_exit"] == []
    assert good["constraint_satisfied"] is True
    assert flawed["constraint_satisfied"] is False
    assert good["failure_scope"] == "writer_process_only"
    assert packet.SUBPROCESS_TIMEOUT_SECONDS == 5


def test_real_bounded_queue_rejection_and_silent_deque_eviction():
    good = packet.queue_observation(True)
    flawed = packet.queue_observation(False)
    assert good["rejected_before_accept"] == ["message-3"]
    assert good["accepted"] == good["retained_before_consumption"] == ["message-1", "message-2"]
    assert flawed["accepted"] == ["message-1", "message-2", "message-3"]
    assert flawed["missing_accepted_messages"] == ["message-1"]
    assert max(good["depth_after_each_arrival"]) == max(flawed["depth_after_each_arrival"]) == 2
    assert good["constraint_satisfied"] is True
    assert flawed["constraint_satisfied"] is False


def test_independent_sqlite_reader_sees_partial_independent_commit():
    good = packet.atomicity_observation(True)
    flawed = packet.atomicity_observation(False)
    assert good["independent_reader_between_writes"] == [100, 0]
    assert good["independent_reader_after_failure"] == [100, 0]
    assert good["successful_full_transfer"] == [90, 10]
    assert flawed["independent_reader_between_writes"] == [90, 0]
    assert flawed["independent_reader_after_failure"] == [90, 0]
    assert good["constraint_satisfied"] is True
    assert flawed["constraint_satisfied"] is False


def test_packet_keeps_oracle_labels_and_missing_case_traces_out_of_inputs(tmp_path):
    output = tmp_path / "packet"
    manifest = packet.write_packet(output)
    cases = [json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()]
    assert manifest["cases"] == 12 and manifest["measured_runs"] == 8
    assert not manifest["training_eligible"] and not manifest["final_test_eligible"]
    assert {case["split"] for case in cases} == {"development_regression"}
    for check in packet.CHECKS:
        group = [case for case in cases if case["check_id"] == check]
        assert len({case["scenario_family"] for case in group}) == 1
        assert {case["expected"] for case in group} == {"supported", "contradicted", "insufficient_information"}
    for case in cases:
        DesignReviewRequest.model_validate(case["input"])
        text = json.dumps(case["input"])
        assert "constraint_satisfied" not in text
        assert "expected" not in text
        assert case["expert_reviewed"] is False
        assert case["provenance"] == "executable_local_regression"
        if case["expected"] == "insufficient_information":
            assert "Measured local trace" not in text
            assert "checks" not in case["input"]
            assert case["observation_id"] is None and case["command"] is None
    assert packet.verify_packet(output)["status"] == "verified"
    with pytest.raises(ValueError, match="immutable"):
        packet.write_packet(output)
    (output / "cases.jsonl").write_text("modified")
    with pytest.raises(ValueError, match="Packet hash mismatch"):
        packet.verify_packet(output)


def test_regression_generator_does_not_import_plugin_rules_or_models():
    imported = set()
    for node in ast.walk(ast.parse(SOURCE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert all(not name.startswith(("arch_review", "laya", "torch", "transformers")) for name in imported)


def test_published_packet_reproduces_its_frozen_observations():
    result = packet.verify_packet(ROOT / "datasets" / "executable-backend-v1")
    assert result == {"status": "verified", "cases": 12, "measured_runs": 8,
                      "split": "development_regression"}
