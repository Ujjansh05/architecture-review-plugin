import math
import time
from collections import Counter

from .dataset import expert_reviewed, load_cases, render_case
from .decisions import DEFINITIONS
from .evidence import EvidenceStore
from .model import ModelWorker
from .rules import check_rules
from .schema import ReviewRequest, Verdict
from .settings import Settings

LABELS = [v.value for v in Verdict]


def metrics(items):
    counts = Counter((r["expected"], r["predicted"]) for r in items)
    by_label = {}
    for label in LABELS:
        tp = counts[label, label]
        actual = sum(v for (truth, _), v in counts.items() if truth == label)
        predicted = sum(v for (_, guess), v in counts.items() if guess == label)
        precision = tp / predicted if predicted else 0
        recall = tp / actual if actual else None
        f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0
        by_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    scored = [r for r in items if r.get("probabilities")]
    ece = brier = None
    if scored:
        ece = 0
        brier = sum(sum((r["probabilities"].get(label, 0) - int(label == r["expected"])) ** 2
                        for label in LABELS) for r in scored) / len(scored)
        for index in range(10):
            bucket = [r for r in scored if min(int(max(r["probabilities"].values()) * 10), 9) == index]
            if bucket:
                confidence = sum(max(r["probabilities"].values()) for r in bucket) / len(bucket)
                accuracy = sum(r["expected"] == r["predicted"] for r in bucket) / len(bucket)
                ece += len(bucket) / len(scored) * abs(confidence - accuracy)
    return {"count": len(items), "accuracy": sum(r["expected"] == r["predicted"] for r in items) / len(items) if items else None,
            "macro_f1": sum(d["f1"] for d in by_label.values()) / len(LABELS), "per_label": by_label,
            "false_supported": sum(r["predicted"] == "supported" and r["expected"] != "supported" for r in items),
            "ece_10_bins": ece, "multiclass_brier": brier, "scored_items": len(scored),
            "missing_labels": [k for k, v in by_label.items() if v["support"] == 0]}


def evaluate(source=None, split="development", backend="rules", checkpoint=None, predictions=None, perturb=False):
    all_rows = load_cases(source)
    rows = [row for row in all_rows if split == "all" or row["split"] == split]
    if not rows:
        raise ValueError("Requested evaluation split is empty")
    supplied = None
    if predictions is not None:
        import json
        prediction_rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
        supplied = {row["id"]: row for row in prediction_rows}
        if len(supplied) != len(prediction_rows) or set(supplied) != {r["id"] for r in rows}:
            raise ValueError("Imported predictions must contain each selected case ID exactly once")
    store = EvidenceStore()
    worker = ModelWorker(Settings(checkpoint=checkpoint))
    results, changes = [], []
    started = time.perf_counter()
    try:
        for row in rows:
            if backend == "rules" and supplied is None:
                guesses = {f.check_id: f.verdict.value for f in check_rules(ReviewRequest.model_validate(row["request"]))}
                probabilities = {}
                errors = {}
            elif supplied is not None:
                guesses = supplied[row["id"]]["answers"]
                probabilities = supplied[row["id"]].get("probabilities", {})
                errors = {}
                if set(guesses) != set(row["expected"]):
                    raise ValueError("Imported answer IDs do not match expected decisions")
            else:
                items = list(render_case(row, store))
                payloads = {next(iter(item["questions"])): {"state": item["state"], "questions": item["questions"]}
                            for item in items}
                answers, errors = worker.predict(payloads)
                guesses = {key: answer.verdict.value for key, answer in answers.items()}
                probabilities = {key: {label.value: score for label, score in answer.probabilities.items()}
                                 for key, answer in answers.items()}
                if perturb:
                    import copy
                    changed = copy.deepcopy(payloads)
                    for key, item in changed.items():
                        item["questions"][key]["criteria"] = dict(reversed(list(item["questions"][key]["criteria"].items())))
                    altered, altered_errors = worker.predict(changed)
                    changes.append({"id": row["id"], "changed": any(key in altered and altered[key].verdict.value != guess
                                                                                for key, guess in guesses.items()),
                                    "errors": altered_errors})
            for key, expected in row["expected"].items():
                predicted = guesses.get(key)
                if predicted is not None:
                    Verdict(predicted)
                scores = probabilities.get(key)
                if scores is not None and (set(scores) != set(LABELS) or any(not math.isfinite(v) or v < 0 or v > 1 for v in scores.values())
                                           or abs(sum(scores.values()) - 1) > 0.01):
                    raise ValueError("Invalid imported probabilities")
                results.append({"id": row["id"], "check_id": key, "expected": expected, "predicted": predicted,
                                "probabilities": scores, "error": errors.get(key), "critical": row.get("critical", False)})
    finally:
        worker.close()
    by_check = {key: metrics([r for r in results if r["check_id"] == key]) for key in DEFINITIONS}
    reviewed = all(expert_reviewed(row) for row in rows)
    gates = {key: value["count"] > 0 and not value["missing_labels"] and value["macro_f1"] >= .85
                  and (value["per_label"]["contradicted"]["recall"] or 0) >= .9
                  and (value["per_label"]["insufficient_information"]["recall"] or 0) >= .9
             for key, value in by_check.items()}
    return {"backend": "imported_general_ai" if supplied is not None else backend, "checkpoint": checkpoint,
            "split": split, "expert_reviewed": reviewed, "experimental": not reviewed,
            "notice": "Synthetic draft-label agreement is not expert validation or evidence of production safety.",
            "per_check": by_check, "overall": metrics(results), "pilot_numeric_gates": gates,
            "quality_release_approved": False,
            "release_note": "Requires reviewed independent test cases, base comparison, critical regressions and a human pilot.",
            "critical_false_supported": sum(r["critical"] and r["predicted"] == "supported" and r["expected"] != "supported" for r in results),
            "option_order_changes": changes, "elapsed_seconds": time.perf_counter() - started,
            "evidence_version": store.version, "results": results}
