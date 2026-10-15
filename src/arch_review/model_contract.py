"""Artifact-bound fact formatting and separate model-assessment provenance."""
import hashlib
import json
from pathlib import Path

from .decision_input import decision_input
from .research_fact_input import render

VERSION = "archrivet-facts-2"


def format_input(check_id, facts, evidence, context=""):
    payload = render(check_id, facts, evidence, context, "explicit")
    state = json.loads(payload["state"])
    state["input_contract"] = VERSION
    payload["state"] = json.dumps(state, ensure_ascii=False, sort_keys=True)
    criteria = payload["questions"][check_id]["criteria"]
    payload["questions"][check_id]["criteria"] = {key: criteria[key] for key in sorted(criteria)}
    return payload


def formatter_identity():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest() for name in
            ("model_contract.py", "research_fact_input.py", "decision_input.py", "registry.py", "schema.py")}


def verify_contract(checkpoint, context_tokens):
    root = Path(checkpoint)
    path = root / "decision-contract.json"
    if not path.exists():
        return None
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (contract.get("input_contract") != VERSION or contract.get("context_tokens") != context_tokens
            or contract.get("formatter") != formatter_identity()):
        raise ValueError("Model formatter/context differs from its evaluated artifact contract")
    files = contract.get("files", {})
    required = {"model.safetensors", "rl_agent_config.json", "calibration.json"}
    if not required <= files.keys() or not any(p.startswith("tokenizer/") for p in files):
        raise ValueError("Model contract must bind weights, tokenizer, configuration and calibration")
    for name, expected in files.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError("Model contract path is outside the checkpoint or missing")
        with target.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Model artifact differs from its evaluated contract")
    return contract


def payload_for(contract, check_id, facts, evidence, context=""):
    if contract is None:
        return decision_input(check_id, facts, evidence, context)
    return format_input(check_id, facts, evidence, context)


def assessment_metadata(decision, contract):
    if contract is None:
        return decision
    # Abstention is optional and separate from the semantic missing-facts label.
    threshold = contract.get("abstention_threshold")
    if threshold is not None and (not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1):
        raise ValueError("Invalid abstention threshold")
    confidence = max(decision.get("probabilities", {}).values(), default=0.)
    abstained = threshold is not None and confidence < threshold
    return {**decision, "assessment_status": "abstained" if abstained else "assessed",
            "abstention_reason": "Below the artifact's model-confidence threshold" if abstained else None,
            "input_contract": contract["input_contract"],
            "calibration_provenance": {"status": contract.get("calibration_status", "fitted_unvalidated"),
                                       "sha256": contract["files"]["calibration.json"],
                                       "verification_split": "calibration_verify"},
            "production_failure_probability": None}
