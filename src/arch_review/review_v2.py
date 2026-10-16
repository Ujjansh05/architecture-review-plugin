"""Evidence-aware orchestration; neither model approval nor long input proves completeness."""
import time

from .clarifications import enrich, next_questions
from .documents import extract_documents, validate_spans
from .registry import REGISTRY_VERSION
from .resources import ResourceSampler
from .review_summary import render_summary
from .rules_v2 import evaluate_check
from .schema import ReviewRequest, Verdict
from .schema_v2 import CheckInput, DesignReviewResponse, Span
from .workload import calculate


class DesignReviewer:
    def __init__(self, backend, evidence):
        self.backend, self.evidence = backend, evidence

    def review(self, request, cancelled=lambda: False, progress=lambda **_: None):
        with ResourceSampler() as sampler:
            result = self._review(request, cancelled, progress)
        result.resources = sampler.report()
        result.review_summary = render_summary(result.model_dump(mode="json"))
        return result

    def _review(self, request, cancelled, progress):
        started = time.monotonic()
        timings = {key: 0.0 for key in ("validate_facts", "extract_and_verify", "rules",
                                        "evidence_retrieval", "model", "workload", "aggregate")}
        model_metrics = {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
                         "measurement_scope": "Maximum of worker-recorded allocator peaks across review calls."}
        stage = time.monotonic()
        validate_spans(request)
        timings["validate_facts"] = (time.monotonic() - stage) * 1000
        stage = time.monotonic()
        items, coverage, conflicts = extract_documents(request, self.backend, cancelled, progress)
        timings["extract_and_verify"] = (time.monotonic() - stage) * 1000
        covered_checks = {item.check_id for item in items}
        items += [CheckInput(check_id=check_id, facts={}) for check_id in request.requested_checks
                  if check_id not in covered_checks]
        findings, selected, failures = [], {}, []
        for item in items:
            if cancelled():
                raise InterruptedError("Review cancelled")
            if item.check_id not in request.requested_checks:
                continue
            stage = time.monotonic()
            finding = evaluate_check(item)
            timings["rules"] += (time.monotonic() - stage) * 1000
            finding.fact_origins = dict(item.fact_origins)
            stage = time.monotonic()
            records = self.evidence.search(item.check_id, item.facts)
            timings["evidence_retrieval"] += (time.monotonic() - stage) * 1000
            selected.update({r["id"]: r for r in records})
            finding.evidence_ids = [r["id"] for r in records]
            if finding.evaluation_status == "evaluated" and hasattr(self.backend, "judge"):
                stage = time.monotonic()
                try:
                    evidence = [{"id": r["id"], "conditions": r["conditions"], "lesson": r["lesson"]} for r in records]
                    model = self.backend.judge(item.check_id, item.facts, evidence,
                                               **({"context": item.context} if item.context else {}))
                    finding.model_decision = model
                    if model.get("assessment_status") != "abstained" and model["verdict"] != finding.rule_verdict:
                        finding.conflict = True
                        if finding.rule_verdict != Verdict.CONTRADICTED:
                            finding.verdict = Verdict.INSUFFICIENT
                        finding.missing_information.append("Resolve the rule/model disagreement against explicit facts.")
                except Exception as exc:
                    failures.append(f"{item.scope_id}/{item.check_id}: {type(exc).__name__}: {exc}")
                finally:
                    timings["model"] += (time.monotonic() - stage) * 1000
                    runtime = self.backend.metadata().get("runtime_metrics", {})
                    for sample in (runtime.get("startup", {}), runtime.get("last_request", {})):
                        for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                            value = sample.get(key)
                            if type(value) is int and value >= 0:
                                model_metrics[key] = max(model_metrics[key] or 0, value)
            stage = time.monotonic()
            unverified = [row for row in coverage["unverified_facts"]
                          if (row["scope_id"], row["check_id"]) == (item.scope_id, item.check_id)]
            for row in unverified:
                path = row["fact_path"]
                if path not in finding.missing_information:
                    finding.missing_information.append(path)
                finding.fact_origins.setdefault(path, "host_extracted")
                # Preserve the citation without putting its unverified value back
                # into facts passed to the rules or model.
                references = finding.supporting_spans.setdefault(path, [])
                for reference in row["supporting_spans"]:
                    span = Span.model_validate(reference)
                    if span not in references:
                        references.append(span)
            if unverified and finding.verdict == Verdict.SUPPORTED:
                finding.verdict = Verdict.INSUFFICIENT
            if unverified and finding.verdict == Verdict.INSUFFICIENT:
                finding.reason = "Confirm the proposed facts from the actual design; source quotes alone do not verify their meaning."
            disputed = conflicts.get((item.scope_id, item.check_id), set())
            if disputed:
                finding.verdict = Verdict.INSUFFICIENT
                finding.conflict = True
                finding.missing_information.extend("Conflicting fact: " + path for path in sorted(disputed))
            if coverage["unprocessed"] and finding.verdict == Verdict.SUPPORTED:
                finding.verdict = Verdict.INSUFFICIENT
                finding.missing_information.append("Some document sections were not reviewed; a supported verdict is withheld.")
            enrich(finding)
            for question in finding.clarification_questions:
                proposals = [row for row in unverified if row["fact_path"] == question["fact_path"]]
                if proposals:
                    question["proposed_facts"] = proposals
                    question["requires_confirmation"] = True
                    question["question"] = "Confirm this detail from the actual design. " + question["question"]
            findings.append(finding)
            timings["aggregate"] += (time.monotonic() - stage) * 1000
        coverage.update(requested_checks=request.requested_checks,
                        evaluated_scopes=len(findings), registry_version=REGISTRY_VERSION,
                        evidence_version=self.evidence.version, inference_errors=failures)
        coverage["operations"] = sorted({f.scope_id for f in findings})
        coverage["whole_architecture_reviewed"] = False
        stage = time.monotonic()
        estimates, capacity = calculate(ReviewRequest(requirements={}, design={},
            workload=request.workload, deployment=request.deployment))
        timings["workload"] = (time.monotonic() - stage) * 1000
        coverage["timings_ms"] = {key: round(value, 3) for key, value in timings.items()}
        coverage["model_metrics"] = model_metrics
        coverage["timing_scope"] = "Review stages only; excludes resource sampling and final summary rendering."
        incomplete = (coverage["unprocessed"] or failures or coverage["unverified_facts"]
                      or any(f.verdict == Verdict.INSUFFICIENT or f.conflict for f in findings))
        return DesignReviewResponse(review_status="partial" if incomplete else "complete", checks=findings,
            coverage=coverage, evidence=list(selected.values()), workload_estimates=estimates, capacity=capacity,
            backend=self.backend.metadata(), elapsed_ms=round((time.monotonic() - started) * 1000, 2),
            next_questions=next_questions(findings),
            warnings=["Experimental review of supplied facts; no production certification or outage probability.",
                      "Source lessons are background evidence; extracted facts and labels await independent review.",
                      "Complete means the requested checks finished, not that the entire architecture is safe."])
