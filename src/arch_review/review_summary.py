"""Deterministic findings for hosts to preserve; no second model summarizes risks."""
import hashlib
import json


def quoted(value):
    # Keep caller-controlled labels and text on one line and visibly quoted.
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def summary_items(review):
    items = []

    def add(key, text):
        items.append({"id": key, "text": text})

    add("scope", "Only the listed requirements were reviewed; the whole architecture is not certified.")
    add("facts", "Findings depend on supplied facts. Source interpretation and deployment behavior are not independently verified.")
    add("changes", "Suggested changes are proposals, not evidence of an implemented fix.")
    for finding in sorted(review["checks"], key=lambda f: (f["scope_id"], f["check_id"])):
        scope = quoted([finding["scope_id"], finding["check_id"]])
        key = hashlib.sha256(scope.encode()).hexdigest()[:16]
        verdict = finding["verdict"] if finding["evaluation_status"] == "evaluated" else "not_evaluated"
        add(key, f"Scope/check {scope}: {verdict}. {quoted(finding['reason'])}")
        for index, missing in enumerate(finding.get("missing_information", [])):
            add(f"{key}.missing.{index}", f"Missing or unresolved for {scope}: {quoted(missing)}")
        if finding.get("conflict"):
            add(f"{key}.conflict", f"Unresolved conflicting evidence or judgments for {scope}.")
        for index, evidence in enumerate(finding.get("evidence_ids", [])):
            add(f"{key}.evidence.{index}", f"Background evidence for {scope}: {quoted(evidence)}")
    capacity = review["capacity"]
    if capacity.get("evidence_status") == "unknown":
        add("capacity", "Application capacity is unknown; workload arithmetic is not a capacity benchmark.")
    else:
        add("capacity", "Capacity applies only to supplied benchmark conditions; provenance is not independently verified: "
            + quoted({k: capacity.get(k) for k in ("evidence_status", "supported_capacity", "latency_slo_status")}))
    for index, value in enumerate(capacity.get("missing_evidence", [])):
        add(f"capacity.missing.{index}", "Missing capacity evidence: " + quoted(value))
    for index, value in enumerate(review.get("workload_estimates", {}).get("missing_inputs", [])):
        add(f"workload.missing.{index}", "Missing workload input: " + quoted(value))
    for name in ("unprocessed", "extraction_errors", "inference_errors"):
        if review.get("coverage", {}).get(name):
            add("coverage." + name, f"Review coverage limitation ({name}): " + quoted(review["coverage"][name]))
    for index, warning in enumerate(review.get("warnings", [])):
        add(f"warning.{index}", "Limitation: " + quoted(warning))
    return items


def render_summary(review):
    return "ArchRivet review summary\n" + "\n".join("- " + item["text"] for item in summary_items(review))


def audit_summary(review, answer):
    """Literal retention only. This does not certify other free-form answer claims."""
    def normalize(text):
        return " ".join(text.split())
    text = normalize(answer)
    items = summary_items(review)
    missing = [item for item in items if normalize(item["text"]) not in text]
    return {"required_items": len(items), "retained_items": len(items) - len(missing),
            "missing_items": missing, "literal_retention_passed": not missing,
            "freeform_claims_verified": False,
            "method": "Exact required statements with whitespace normalization; paraphrases require separate review."}
