"""Evidence-aware orchestration; neither model approval nor long input proves completeness."""
import time

from .documents import extract_documents, validate_spans
from .registry import REGISTRY_VERSION
from .resources import ResourceSampler
from .rules_v2 import evaluate_check
from .schema import ReviewRequest, Verdict
from .schema_v2 import CheckInput, DesignReviewResponse
from .workload import calculate


class DesignReviewer:
    def __init__(self, backend, evidence):
        self.backend, self.evidence = backend, evidence

    def review(self, request, cancelled=lambda: False, progress=lambda **_: None):
        with ResourceSampler() as sampler:
            result = self._review(request, cancelled, progress)
        result.resources = sampler.report()
        return result

    def _review(self, request, cancelled, progress):
        started = time.monotonic()
        validate_spans(request)
        items, coverage, conflicts = extract_documents(request, self.backend, cancelled, progress)
        covered_checks = {item.check_id for item in items}
        items += [CheckInput(check_id=check_id, facts={}) for check_id in request.requested_checks
                  if check_id not in covered_checks]
        findings, selected, failures = [], {}, []
        for item in items:
            if cancelled():
                raise InterruptedError("Review cancelled")
            if item.check_id not in request.requested_checks:
                continue
            finding = evaluate_check(item)
            records = self.evidence.search(item.check_id, item.facts)
            selected.update({r["id"]: r for r in records})
            finding.evidence_ids = [r["id"] for r in records]
            if finding.evaluation_status == "evaluated" and hasattr(self.backend, "judge"):
                try:
                    evidence = [{"id": r["id"], "conditions": r["conditions"], "lesson": r["lesson"]} for r in records]
                    model = self.backend.judge(item.check_id, item.facts, evidence)
                    finding.model_decision = model
                    if model["verdict"] != finding.rule_verdict:
                        finding.conflict = True
                        if finding.rule_verdict != Verdict.CONTRADICTED:
                            finding.verdict = Verdict.INSUFFICIENT
                        finding.missing_information.append("Resolve the rule/model disagreement against explicit facts.")
                except Exception as exc:
                    failures.append(f"{item.scope_id}/{item.check_id}: {type(exc).__name__}: {exc}")
            disputed = conflicts.get((item.scope_id, item.check_id), set())
            if disputed:
                finding.verdict = Verdict.INSUFFICIENT
                finding.conflict = True
                finding.missing_information.extend("Conflicting fact: " + path for path in sorted(disputed))
            if coverage["unprocessed"] and finding.verdict == Verdict.SUPPORTED:
                finding.verdict = Verdict.INSUFFICIENT
                finding.missing_information.append("Some document sections were not reviewed; a supported verdict is withheld.")
            findings.append(finding)
        coverage.update(requested_checks=request.requested_checks,
                        evaluated_scopes=len(findings), registry_version=REGISTRY_VERSION,
                        evidence_version=self.evidence.version, inference_errors=failures)
        estimates, capacity = calculate(ReviewRequest(requirements={}, design={},
            workload=request.workload, deployment=request.deployment))
        incomplete = (coverage["unprocessed"] or failures
                      or any(f.verdict == Verdict.INSUFFICIENT or f.conflict for f in findings))
        return DesignReviewResponse(review_status="partial" if incomplete else "complete", checks=findings,
            coverage=coverage, evidence=list(selected.values()), workload_estimates=estimates, capacity=capacity,
            backend=self.backend.metadata(), elapsed_ms=round((time.monotonic() - started) * 1000, 2),
            warnings=["Experimental review of supplied facts; no production certification or outage probability.",
                      "Source lessons are background evidence; extracted facts and labels await independent review.",
                      "Complete means the requested checks finished, not that the entire architecture is safe."])
