"""Per-check decision quality, family bootstrap intervals and explicit eligibility gates."""
import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .evaluation import metrics
from .registry import REGISTRY
from .schema import Verdict

LABELS = tuple(label.value for label in Verdict)
CONTEXT_LENGTHS = (1024, 2048, 4096)
HEAD_TOKENS = 256


def ordered_questions(row, order=None, neutral_labels=False):
    """Render identical criteria with a reversible label map; never change evidence."""
    questions = copy.deepcopy(row["questions"])
    if len(questions) != 1:
        raise ValueError("Diagnostic rows must contain exactly one decision")
    check = next(iter(questions))
    criteria = questions[check]["criteria"]
    order = list(criteria) if order is None else list(order)
    if set(criteria) != set(LABELS) or len(order) != len(LABELS) or set(order) != set(LABELS):
        raise ValueError("Every decision must contain each of the three verdicts exactly once")
    mapping = {chr(65 + index) if neutral_labels else label: label for index, label in enumerate(order)}
    questions[check]["criteria"] = {key: criteria[label] for key, label in mapping.items()}
    return questions, mapping


def training_row(row, neutral_labels=False):
    """Only presentation labels change; source state and semantic gold labels do not."""
    result = copy.deepcopy(row)
    result["questions"], mapping = ordered_questions(row, neutral_labels=neutral_labels)
    reverse = {value: key for key, value in mapping.items()}
    for field in ("expected", "answers"):
        if field in result:
            result[field] = {check: reverse[label] for check, label in result[field].items()}
    if "gold" in result:
        raise ValueError("Head diagnostics require explicit expected labels, not a second gold representation")
    return result


def checked_prediction(choice, probabilities=None, label_map=None):
    mapping = label_map or {label: label for label in LABELS}
    if choice not in mapping:
        raise ValueError("Inference returned a choice outside the presented options")
    verdict = Verdict(mapping[choice]).value
    if probabilities is None:
        return verdict, None
    if set(probabilities) != set(mapping):
        raise ValueError("Inference probabilities do not match the presented options")
    decoded = {mapping[key]: value for key, value in probabilities.items()}
    if (set(decoded) != set(LABELS)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in decoded.values())
            or abs(sum(decoded.values()) - 1) > .01):
        raise ValueError("Inference returned invalid verdict probabilities")
    return verdict, decoded


def check_laya_usage(result, check):
    usage = result.get("usage", {})
    if usage.get("truncated") or usage.get("state_tokens_dropped", 0):
        raise ValueError("Inference truncated the model input")
    options = usage.get("options", {}).get(check, {})
    if "distinct" in options and options["distinct"] != len(LABELS):
        raise ValueError("Inference collapsed answer options")


def selection_score(score):
    """Select on development macro-F1, then prefer fewer unsafe positive answers."""
    value = score["macro_f1"]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Development selection requires a finite macro-F1")
    return value, -score["false_supported"]


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def candidate_signature(candidate):
    return hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_frozen_selection(selection, candidate):
    """A backend name alone cannot identify the model selected before opening test."""
    if (selection.get("version") != "backend-selection-1"
            or selection.get("selected_from_split") != "development"
            or selection.get("test_predictions_used") is not False):
        raise ValueError("Final test requires a candidate frozen from development without test predictions")
    if (selection.get("candidate") != candidate
            or selection.get("candidate_sha256") != candidate_signature(candidate)):
        raise ValueError("Frozen candidate differs from the dataset, model artifacts, or inference configuration")
    if not candidate.get("artifacts_verified"):
        raise ValueError("Final test requires verifiable local model artifacts or the pinned rules implementation")
    if candidate.get("backend") == "imported_blind_predictions":
        raise ValueError("Imported final-test predictions need independently bound model provenance; this runner cannot verify it")


def validate_backend_export(directory):
    """Bind source labels to the already hashed model inputs, without rendering new evidence."""
    from .backend_dataset import audit_cases, digest
    from .training import validate_export

    directory = Path(directory)
    manifest, exported = validate_export(directory, allow_unreviewed=True)
    if manifest.get("version") != "backend-diagnostics-1":
        raise ValueError("Use an explicit backend diagnostic export")
    cases = [json.loads(line) for line in (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if digest(json.dumps(cases, sort_keys=True).encode()) != manifest.get("source_cases_sha256"):
        raise ValueError("Source cases differ from the frozen dataset manifest")
    audit = audit_cases(cases)
    indexed = {case["id"]: case for case in cases}
    found = set()
    for split, rows in exported.items():
        if len(rows) != manifest["files"][split + ".jsonl"]["count"]:
            raise ValueError("Exported split count differs from its manifest")
        for row in rows:
            if row["id"] not in indexed or row["id"] in found:
                raise ValueError("Model input case IDs are missing or duplicated")
            found.add(row["id"])
            case = indexed[row["id"]]
            check = case["check"]["check_id"]
            if (case["split"] != split or case["scenario_family"] != row["scenario_family"]
                    or row["expected"] != {check: case["expected"]}
                    or set(row["questions"]) != {check}
                    or row["questions"][check].get("instructions") != REGISTRY[check].definition
                    or json.loads(row["state"]).get("design") != case["description"]):
                raise ValueError("Source cases and exported model inputs disagree")
            if "answers" in row and row["answers"] != row["expected"]:
                raise ValueError("Exported answers and expected labels disagree")
            ordered_questions(row)
    if found != set(indexed):
        raise ValueError("Source cases and exported input IDs differ")
    return manifest, exported, indexed, audit


def summarize(results, audit, bootstrap_samples=300):
    if bootstrap_samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    results = copy.deepcopy(results)
    grouped = defaultdict(list)
    for row in results:
        if row["check_id"] not in REGISTRY or row["expected"] not in LABELS:
            raise ValueError("Unexpected check or expected verdict in evaluation")
        # Do not let a stale prediction/probability vector make a failed call appear correct.
        if row.get("error") or row.get("predicted") not in LABELS:
            row["error"] = row.get("error") or "Missing or invalid inference verdict"
            row["predicted"] = None
            row.pop("probabilities", None)
        elif row.get("probabilities") is not None:
            try:
                checked_prediction(row["predicted"], row["probabilities"])
            except ValueError as exc:
                row.update(error=str(exc), predicted=None)
                row.pop("probabilities", None)
        grouped[row["check_id"]].append(row)
    reports, gates = {}, {}
    for check in REGISTRY:
        items = grouped[check]
        score = metrics(items)
        families = defaultdict(list)
        for row in items:
            families[row["scenario_family"]].append(row)
        draws = []
        if len(families) >= 2:
            rng = random.Random(17)
            keys = list(families)
            for _ in range(bootstrap_samples):
                sample = [row for key in rng.choices(keys, k=len(keys)) for row in families[key]]
                draws.append(metrics(sample)["macro_f1"])
        draws.sort()
        score["macro_f1_family_bootstrap_95"] = ([draws[int(.025 * len(draws))], draws[min(len(draws)-1, int(.975 * len(draws)))]]
                                                  if draws else None)
        score["scenario_family_count"] = len(families)
        score["independent_family_count"] = len(families) if audit.get("independently_reviewed") else None
        score["primary_inference_failures"] = sum(bool(row.get("error")) for row in items)
        score["perturbation_inference_failures"] = sum(len(row.get("perturbation_errors", [])) for row in items)
        score["inference_failures"] = score["primary_inference_failures"] + score["perturbation_inference_failures"]
        score["critical_false_supported"] = sum(bool(row.get("critical")) and row["predicted"] == "supported"
                                                and row["expected"] != "supported" for row in items)
        eligible_perturbations = [r for r in items if r.get("perturbation_complete") is True
                                 and r.get("perturbation_orders_scored") == 6 and not r.get("error")
                                 and not r.get("perturbation_errors")]
        score["perturbation_cases"] = len(eligible_perturbations)
        score["perturbation_coverage"] = len(eligible_perturbations) / len(items) if items else 0
        score["changed_fraction"] = (sum(bool(r.get("changed")) for r in eligible_perturbations) / len(eligible_perturbations)
                                     if eligible_perturbations else None)
        score["perturbation_gate_passed"] = bool(items and len(eligible_perturbations) == len(items)
                                                and score["changed_fraction"] <= .05)
        labels = score["per_label"]
        gates[check] = bool(items and not score["missing_labels"] and score["macro_f1"] >= .85
            and (labels["contradicted"]["recall"] or 0) >= .9
            and (labels["insufficient_information"]["recall"] or 0) >= .9
            and labels["supported"]["precision"] >= .95 and score["critical_false_supported"] == 0
            and score["inference_failures"] == 0 and score["perturbation_gate_passed"])
        reports[check] = score
    overall = metrics(results)
    overall["inference_failures"] = sum(value["inference_failures"] for value in reports.values())
    return {"per_check": reports, "overall": overall, "numeric_gates": gates,
            "independent_quality_release_approved": False,
            "eligibility": audit, "results": results,
            "gate_requirements": {"macro_f1_minimum": .85, "contradicted_recall_minimum": .9,
                "insufficient_information_recall_minimum": .9, "supported_precision_minimum": .95,
                "critical_false_supported_maximum": 0, "inference_failures_maximum": 0,
                "perturbation_coverage_required": 1.0, "perturbation_change_fraction_maximum": .05},
            "notice": "Template families are not independent real-world examples. Numeric gates require complete answer-order testing. "
                      "Failed primary calls are scored as failures, not unknown answers; failed perturbations block promotion."}
