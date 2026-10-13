import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def evaluator():
    spec = importlib.util.spec_from_file_location("adapter_regressions_test", ROOT / "scripts/evaluate_adapter_regressions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_inputs_exclude_labels_provenance_and_case_ids(evaluator):
    cases, _ = evaluator.load_cases(ROOT / "datasets/executable-backend-v1")
    evidence = SimpleNamespace(search=lambda check, facts: [{"id": "lesson-1", "lesson": "Ordinary reference lesson."}])
    for case in cases:
        payload = evaluator.render_payload(case, evidence, True)
        changed = copy.deepcopy(case)
        for name in ("id", "expected", "provenance", "scenario_family", "command", "observation_id", "notice"):
            changed[name] = "secret-label-marker"
        assert evaluator.render_payload(changed, evidence, True) == payload
        assert case["id"] not in json.dumps(payload)
        assert "expected" not in json.loads(payload["state"])
        assert set(payload["questions"][case["check_id"]]["criteria"]) == {"A", "B", "C"}
        if case["expected"] == "insufficient_information":
            assert "Measured local trace" not in payload["state"]
            assert "Explicit known facts" not in payload["state"]
        else:
            assert "Measured local trace" in payload["state"]
            assert "Explicit known facts" in payload["state"]


def test_packet_tampering_is_rejected(evaluator, tmp_path):
    directory = tmp_path / "packet"
    shutil.copytree(ROOT / "datasets/executable-backend-v1", directory)
    with (directory / "cases.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="frozen packet"):
        evaluator.load_cases(directory)


def test_real_retrieval_is_repeatable_and_temporary_index_closes(evaluator):
    cases, _ = evaluator.load_cases(ROOT / "datasets/executable-backend-v1")
    first, version = evaluator.prepare_inputs(cases, True)
    second, other_version = evaluator.prepare_inputs(cases, True)
    assert first == second
    assert version == other_version
    assert len(first) == 12
    assert all(json.loads(payload["state"])["evidence"] for payload in first)


def test_preprocessing_checks_every_case_and_rejects_truncation(evaluator):
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text))
    payloads = [{"state": "small", "questions": {"retry_safety": {}}}] * 12
    def builder(*args):
        return [1, 2, 3], [0, 1, 2], {"options_distinct": 3, "options": 3}
    assert evaluator.check_preprocessing(payloads, tokenizer, 1024, builder)["cases_preprocessed"] == 12
    with pytest.raises(ValueError, match="state budget"):
        evaluator.check_preprocessing([{"state": "x" * 900, "questions": {}}], tokenizer, 1024, builder)
    with pytest.raises(ValueError, match="collapsed"):
        evaluator.check_preprocessing(payloads, tokenizer, 1024,
                                      lambda *args: ([1], [0], {"options_distinct": 1, "options": 3}))


def test_scoring_keeps_failures_separate_from_unknown(evaluator):
    rows = [{"expected": "contradicted", "predicted": None, "error": "failure", "scenario_family": "family"},
            {"expected": "insufficient_information", "predicted": "supported", "scenario_family": "family"},
            {"expected": "supported", "predicted": "insufficient_information", "scenario_family": "family"}]
    result = evaluator.score(rows)["overall"]
    assert result["missed_defects"] == result["false_approvals"] == result["inference_failures"] == 1
    assert result["unknown_predictions"] == result["expected_unknowns"] == 1


def test_base_tampering_is_rejected_before_model_import(evaluator, tmp_path):
    base, experiment = tmp_path / "base", tmp_path / "experiment"
    base.mkdir()
    experiment.mkdir()
    (base / "model.safetensors").write_bytes(b"tampered")
    (experiment / "report.json").write_text(json.dumps({
        "status": "experimental_encoder_adapter_training_completed", "training_mode": "encoder_low_rank_adapters",
        "original_encoder_weights_frozen": True, "frozen_parameters_unchanged_verified": True,
        "state_format": "narrative", "base_weights_sha256": "wrong"}), encoding="utf-8")
    helper = SimpleNamespace(checkpoint_plan=lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="Base weight hash"):
        evaluator.verified_experiment(base, experiment, helper)


@pytest.mark.parametrize("changed", ["config", "tokenizer", "delta", "scope", "calibration"])
def test_experiment_integrity_is_bound_to_report(evaluator, tmp_path, changed):
    base, experiment = tmp_path / "base", tmp_path / "experiment"
    base.mkdir()
    experiment.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    (experiment / "experimental-delta.safetensors").write_bytes(b"delta")
    configurations = {"base_config_sha256": "config", "base_tokenizer_sha256": "tokenizer"}
    plan = {"trainable_tensor_names": ["layer.lora_A"], "selected_encoder_layers": [0],
            "adapter_targets": {"layer": {"in_features": 2, "out_features": 3}}}
    report = {"status": "experimental_encoder_adapter_training_completed",
              "training_mode": "encoder_low_rank_adapters", "original_encoder_weights_frozen": True,
              "frozen_parameters_unchanged_verified": True, "state_format": "narrative",
              "base_weights_sha256": evaluator.file_digest(base / "model.safetensors"),
              "delta_sha256": evaluator.file_digest(experiment / "experimental-delta.safetensors"),
              **configurations, **plan,
              "adapter_configuration": {"rank": 8, "alpha": 16, "targets": plan["adapter_targets"],
                                        "merged_into_base": False},
              "implementation_sha256": {"scripts/train_partial_encoder.py":
                                        evaluator.file_digest(ROOT / "scripts/train_partial_encoder.py")},
              "calibration": {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}},
              "context_tokens": 1024, "head_tokens": 256, "neutral_labels": True}
    helper = SimpleNamespace(checkpoint_plan=lambda *args, **kwargs: plan,
                             configuration_hashes=lambda *args: configurations)
    path = experiment / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    evaluator.verified_experiment(base, experiment, helper)
    if changed in ("config", "tokenizer"):
        configurations["base_" + changed + "_sha256"] = "tampered"
    elif changed == "delta":
        (experiment / "experimental-delta.safetensors").write_bytes(b"tampered")
    elif changed == "scope":
        report["trainable_tensor_names"] = ["encoder.weight"]
        path.write_text(json.dumps(report), encoding="utf-8")
    else:
        report["calibration"]["temperature"] = [-1]
        path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluator.verified_experiment(base, experiment, helper)


def test_module_and_help_do_not_import_torch():
    code = ("import importlib.util,sys; "
            "spec=importlib.util.spec_from_file_location('regression','scripts/evaluate_adapter_regressions.py'); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "module.trainer_helpers(); assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=10, capture_output=True)
