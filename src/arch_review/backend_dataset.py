"""Diagnostic fixtures and strict export audits; these are not independent gold labels."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from .documents import flatten, near_duplicate_signature
from .registry import REGISTRY
from .rules_v2 import evaluate_check
from .schema_v2 import CheckInput

LABELS = ("supported", "contradicted", "insufficient_information")
SPLITS = ("train", "development", "calibration", "test")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def positive_facts(check, index=0):
    window = (index + 1) * 3600
    if check == "retry_safety":
        return {"requirements": {"prevent_duplicate_effects": True}, "design": {"retry": {
            "redelivery_possible": True, "all_effects_intrinsically_idempotent": False,
            "stable_message_id": True, "dedup_enforced": True, "dedup_atomic_with_effect": True,
            "dedup_retention_seconds": window * 2, "max_redelivery_age_seconds": window,
            "external_effects": False}}}
    if check == "ack_durability":
        failures = ["process_crash", "host_loss"] if index % 2 else ["process_crash"]
        return {"requirements": {"acknowledged_operation": "store", "survive_failures": failures},
                "design": {"acknowledgement": {"acknowledged_operation": "store", "after_durable_commit": True,
                                               "survives_failures": failures}}}
    if check == "queue_overload":
        return {"requirements": {"bounded_queue_required": True, "preserve_accepted_messages": True},
                "design": {"queue": {"max_messages": 100 * (index + 1), "policy": "reject_before_accept",
                                     "admission_before_accept": True, "preserves_accepted_messages": True, "retry_backoff": True}}}
    if check == "transaction_atomicity":
        return {"atomic_visibility_required": True, "multiple_writes": True, "boundary": "single_transaction",
                "covers_all_writes": True, "partial_state_visible": False, "concurrent_invariant_protected": True}
    if check == "replica_consistency":
        return {"required_consistency": "read_your_writes", "route": "replica", "replication": "asynchronous",
                "read_your_writes_enforced": True, "linearizable_reads_enforced": False, "stale_reads_possible": False}
    if check == "cache_freshness":
        return {"max_staleness_ms": 1000 * (index + 1), "worst_case_staleness_ms": 500 * (index + 1),
                "bound_covers_failures": True, "stale_on_error": False}
    if check == "failover_safety":
        return {"single_writer_required": True, "partition_in_scope": True, "old_writer_can_commit_after_promotion": False,
                "fence_enforced_at_storage": True, "ownership_epoch_monotonic": True}
    return {"deadline_ms": 1000 * (index + 1), "per_attempt_timeout_ms": 100 * (index + 1),
            "max_attempts": 3, "total_backoff_ms": 100, "local_work_ms": 100,
            "deadline_propagated": True, "retry_budget_shared": True, "jitter": True}


CHANGES = {
    "retry_safety": ("design.retry.dedup_atomic_with_effect", False),
    "ack_durability": ("design.acknowledgement.after_durable_commit", False),
    "queue_overload": ("design.queue.preserves_accepted_messages", False),
    "transaction_atomicity": ("partial_state_visible", True),
    "replica_consistency": ("read_your_writes_enforced", False),
    "cache_freshness": ("worst_case_staleness_ms", 1_000_000),
    "failover_safety": ("old_writer_can_commit_after_promotion", True),
    "dependency_resilience": ("per_attempt_timeout_ms", 1_000_000),
}


def changed(facts, path, value=None, remove=False):
    obj = copy.deepcopy(facts)
    target = obj
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    if remove:
        target.pop(parts[-1], None)
    else:
        target[parts[-1]] = value
    return obj


CONTEXTS = [
    "A notification dispatcher records a user-visible delivery operation.",
    "An inventory reservation service handles concurrent checkout requests.",
    "A document collaboration service persists edit operations before showing confirmation.",
    "A telemetry collector processes sensor updates during traffic bursts.",
    "A subscription worker updates entitlements and its processing ledger.",
    "A search indexing pipeline processes document revision events.",
    "A media processing scheduler records job state and completion notifications.",
    "A chat backend stores conversation events and updates unread counters.",
    "An analytics ingestion service maintains aggregate materialized views.",
    "A booking service records seat allocations during concurrent requests.",
]


def make_diagnostic_cases():
    rows = []
    for number, check in enumerate(REGISTRY):
        for index, context in enumerate(CONTEXTS):
            split = "train" if index < 5 else "test" if index >= 8 else (
                "development" if index < (7 if number % 2 == 0 else 6) else "calibration")
            family = f"diagnostic-{check}-{index:02}"
            good = positive_facts(check, index)
            path, bad_value = CHANGES[check]
            for label in LABELS:
                facts = copy.deepcopy(good)
                if label == "contradicted":
                    facts = changed(facts, path, bad_value)
                    if check == "replica_consistency":
                        facts["stale_reads_possible"] = True
                    if check == "dependency_resilience":
                        facts["deadline_propagated"] = False
                elif label == "insufficient_information":
                    facts = changed(facts, path, remove=True)
                item = CheckInput(check_id=check, scope_id="operation", facts=facts)
                # The expected label is assigned by construction, then checked by the implementation.
                finding = evaluate_check(item)
                if finding.verdict != label:
                    raise ValueError(f"Diagnostic construction disagrees with rules: {family}/{label}: {finding}")
                details = "\n".join(f"- {name.replace('_', ' ')}: {json.dumps(value)}" for name, value in flatten(facts).items())
                rows.append({"id": family + "-" + label, "scenario_family": family, "split": split,
                    "description": context + "\nExplicit design declarations:\n" + details,
                    "check": item.model_dump(mode="json"), "expected": label, "rationale": finding.reason,
                    "critical": label == "contradicted", "label_provenance": "synthetic_definition_fixture",
                    "review_status": "not_independently_reviewed", "source_groups": [],
                    "construction": "Template-based diagnostic; not an independent source-derived evaluation."})
    return rows


def audit_cases(rows):
    seen, families, sources, signatures = set(), {}, {}, {}
    near = []
    docs = []
    for row in rows:
        if row["id"] in seen or row["split"] not in SPLITS or row["expected"] not in LABELS:
            raise ValueError("Duplicate case ID, invalid split or label")
        seen.add(row["id"])
        CheckInput.model_validate(row["check"])
        if families.setdefault(row["scenario_family"], row["split"]) != row["split"]:
            raise ValueError("Scenario family crosses splits")
        for source in row.get("source_groups", []):
            if sources.setdefault(source, row["split"]) != row["split"]:
                raise ValueError("Source group crosses splits")
        signature = " ".join(row["description"].lower().split())
        if signatures.setdefault(signature, row["split"]) != row["split"]:
            raise ValueError("Identical descriptions cross splits")
        shingles = near_duplicate_signature(signature)
        for previous, prior in docs:
            union = shingles | prior
            if previous["split"] != row["split"] and union and len(shingles & prior) / len(union) >= .8:
                near.append([previous["id"], row["id"]])
        docs.append((row, shingles))
    coverage = {split: {check: dict(Counter(row["expected"] for row in rows
                    if row["split"] == split and row["check"]["check_id"] == check)) for check in REGISTRY} for split in SPLITS}
    test_counts_met = all(coverage["test"][check].get(label, 0) >= 20 for check in REGISTRY for label in LABELS)
    return {"cases": len(rows), "families": len(families), "coverage": coverage,
            "split_counts": dict(Counter(row["split"] for row in rows)), "near_duplicate_pairs": near,
            "independently_reviewed": all(row.get("review_status") == "expert_reviewed" for row in rows),
            "final_test_minimum_counts_met": test_counts_met,
            "release_eligible": test_counts_met and not near and all(row.get("review_status") == "expert_reviewed" for row in rows),
            "notice": "Template diagnostics expose implementation errors; high agreement is not independent accuracy."}


def export_cases(rows, output, evidence):
    output = Path(output)
    if output.exists():
        raise ValueError("Exports are immutable; choose a new directory")
    audit = audit_cases(rows)
    output.mkdir(parents=True)
    manifest = {"version": "backend-diagnostics-1", "expert_reviewed": audit["independently_reviewed"],
                "family_splits": {row["scenario_family"]: row["split"] for row in rows},
                "files": {}, "source_cases_sha256": digest(json.dumps(rows, sort_keys=True).encode()),
                "evidence_version": evidence.version, "quality_release_approved": False}
    for split in SPLITS:
        exported = []
        for row in rows:
            if row["split"] != split:
                continue
            check = row["check"]["check_id"]
            records = evidence.search(check, row["check"]["facts"])
            state = json.dumps({"design": row["description"],
                                "evidence": [{"id": r["id"], "lesson": r["lesson"]} for r in records]}, ensure_ascii=False)
            question = {check: {"type": "choice", "instructions": REGISTRY[check].definition,
                "criteria": {"supported": "Explicit facts satisfy the requirement.",
                             "contradicted": "Explicit facts establish a violation.",
                             "insufficient_information": "Required facts are missing or conflicting."}}}
            exported.append({"id": row["id"], "scenario_family": row["scenario_family"], "state": state,
                "questions": question, "answers": {check: row["expected"]}, "expected": {check: row["expected"]}})
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported).encode()
        (output / (split + ".jsonl")).write_bytes(content)
        manifest["files"][split + ".jsonl"] = {"sha256": digest(content), "count": len(exported)}
    (output / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8", newline="\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return audit
