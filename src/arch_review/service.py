import time

from . import __version__
from .evidence import EvidenceStore
from .model import ModelWorker, build_state, payload
from .rules import check_rules
from .schema import ReviewRequest, ReviewResponse, Verdict
from .settings import Settings
from .workload import calculate


class ReviewService:
    def __init__(self, settings: Settings | None = None, evidence=None, worker=None):
        self.settings = settings or Settings()
        self.evidence = evidence or EvidenceStore()
        self.worker = worker or ModelWorker(self.settings)

    def close(self):
        self.worker.close()

    def review(self, req: ReviewRequest, *, rules_only=False) -> ReviewResponse:
        start = time.perf_counter()
        findings = check_rules(req)
        selected = {}
        payloads = {}
        for finding in findings:
            records = self.evidence.search(finding.check_id, req.design.model_dump_json())
            finding.evidence_ids = [r["id"] for r in records]
            selected.update({r["id"]: r for r in records})
            payloads[finding.check_id] = payload(finding.check_id, build_state(req, finding.check_id, records))
        predictions, errors = ({}, {}) if rules_only else self.worker.predict(payloads)
        warnings = ["Experimental decision support; supported means conditional on supplied facts.",
                    "Retrieved lessons are background evidence, not verification of this deployment.",
                    "Model probabilities are not production failure probabilities."]
        if any(r.get("review_status") != "expert_reviewed" for r in selected.values()):
            warnings.append("Retrieved source summaries await independent expert review.")
        for f in findings:
            model = predictions.get(f.check_id)
            if model is not None:
                f.model_decision = model
                if model.verdict != f.rule_verdict:
                    f.conflict = True
                    # A concrete counterexample cannot be erased by a model's reassurance.
                    if f.rule_verdict != Verdict.CONTRADICTED:
                        f.verdict = Verdict.INSUFFICIENT
                        f.missing_information.append("Resolve the rule/model disagreement against the stated facts.")
                    f.reason += " The model disagreed; inspect both judgments before using this finding."
                    f.risk["severity"] = "high" if f.verdict == Verdict.CONTRADICTED else "unknown"
            if f.check_id in errors:
                warnings.append(f"{f.check_id}: {errors[f.check_id]}")
        metadata = self.worker.metadata
        if metadata.get("role") != "specialized":
            warnings.append("No specialized architecture checkpoint is active.")
        estimates, capacity = calculate(req)
        complete = bool(predictions) and not errors and not rules_only and all(
            f.verdict != Verdict.INSUFFICIENT and not f.conflict for f in findings)
        return ReviewResponse(
            review_status="complete" if complete else "partial",
            review_mode="laya_assisted" if predictions else "rules_only", checks=findings,
            workload_estimates=estimates, capacity=capacity, evidence=list(selected.values()), warnings=warnings,
            versions={"service": __version__, "evidence": self.evidence.version,
                      "checkpoint": self.settings.checkpoint, "model_metadata": metadata,
                      "definitions": "1.0", "dataset": "pilot-1"},
            elapsed_ms=round((time.perf_counter() - start) * 1000, 3),
        )
