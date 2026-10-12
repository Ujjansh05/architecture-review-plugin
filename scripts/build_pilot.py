"""Create reproducible synthetic pilot cases. Labels await independent expert review."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "arch_review" / "data"
EXAMPLES = ROOT / "examples"


def complete_request():
    return {
        "requirements": {"features": ["Store and deliver messages"], "prevent_duplicate_effects": True,
            "acknowledged_operation": "message_accepted", "survive_failures": ["process_crash", "host_loss"],
            "preserve_accepted_messages": True, "bounded_queue_required": True, "latency_target_ms": 500},
        "design": {"components": ["HTTP API", "Durable replicated queue", "Transactional message store", "Worker"],
            "flows": ["Client submits a message to the API.", "Worker consumes queued messages and writes business state."],
            "retry": {"redelivery_possible": True, "all_effects_intrinsically_idempotent": False,
                "stable_message_id": True, "dedup_enforced": True, "dedup_atomic_with_effect": True,
                "dedup_retention_seconds": 172800, "max_redelivery_age_seconds": 86400, "external_effects": False},
            "acknowledgement": {"acknowledged_operation": "message_accepted", "after_durable_commit": True,
                "survives_failures": ["process_crash", "host_loss"],
                "notes": "Acceptance follows a durable quorum commit on independent hosts; failover preserves committed messages."},
            "queue": {"max_messages": 10000, "policy": "reject_before_accept", "admission_before_accept": True,
                "preserves_accepted_messages": True, "retry_backoff": True,
                "notes": "At capacity return a retryable rejection before success; retry clients use bounded exponential backoff with jitter."}},
        "workload": {"average_messages_per_second": 100, "peak_factor": 5, "message_bytes": 1024,
            "average_recipients": 3, "retention_days": 30, "storage_copies_per_message": 1,
            "replication_factor": 3, "storage_overhead_factor": 1.2,
            "monthly_growth_fraction": 0.1, "growth_horizon_months": 6},
        "deployment": {"hardware": "Unknown; requires measurement", "topology": "Replicas on independent hosts"}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def labeled(case_id, family, split, request, check, verdict, rationale, sources):
    return {"id": case_id, "scenario_family": family, "split": split, "origin": "synthetic",
            "review_status": "pending_expert", "reviewer": None, "reviewed_at": None,
            "request": request, "expected": {check: verdict}, "rationale": rationale,
            "evidence_ids": sources, "critical": verdict == "contradicted"}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    base = complete_request()
    write_json(EXAMPLES / "complete.json", base)
    write_json(EXAMPLES / "incomplete.json", {"requirements": {}, "design": {"components": ["API", "Queue", "Database"]},
                                            "workload": {}, "deployment": {}})
    flawed = copy.deepcopy(base)
    flawed["design"]["retry"].update(dedup_enforced=False, dedup_atomic_with_effect=False)
    flawed["design"]["acknowledgement"]["after_durable_commit"] = False
    flawed["design"]["acknowledgement"]["notes"] = "Success is returned before the durable commit."
    flawed["design"]["queue"].update(policy="unbounded", max_messages=None)
    flawed["design"]["queue"]["notes"] = "No limit or admission control under sustained overload."
    write_json(EXAMPLES / "flawed.json", flawed)

    # Variants are defined before materialization and assigned a split as whole families.
    families = [
        ("ledger", "train", "retry_safety", "A worker increments a conversation's unread counter.", "dedup_enforced", False, "dedup_enforced", ["postgres-constraints-1"]),
        ("inventory", "train", "retry_safety", "A worker reserves one stock unit for an accepted order.", "dedup_atomic_with_effect", False, "dedup_atomic_with_effect", ["aws-outbox-1"]),
        ("webhook", "train", "retry_safety", "A worker calls an external webhook with a message identifier.", "external_effects_idempotent", False, "external_effects_idempotent", ["stripe-idempotency-1"]),
        ("expiry", "development", "retry_safety", "A delayed message can be replayed up to a day later.", "dedup_retention_seconds", 60, "max_redelivery_age_seconds", ["stripe-idempotency-2"]),
        ("concurrent-workers", "train", "retry_safety", "Two consumers may process one message concurrently.", "stable_message_id", False, "stable_message_id", ["postgres-constraints-1"]),
        ("billing-events", "test", "retry_safety", "Each consumed usage event increments a billing accumulator.", "dedup_enforced", False, "dedup_enforced", ["sqs-redelivery-1"]),
        ("notification", "calibration", "retry_safety", "A notification worker invokes an external delivery provider.", "external_effects_idempotent", False, "external_effects_idempotent", ["stripe-idempotency-3"]),
        ("consumer-batch", "train", "retry_safety", "A consumer processes multiple message effects before acknowledging a batch.", "dedup_atomic_with_effect", False, "external_effects", ["rabbit-confirms-1"]),
        ("publisher-acceptance", "train", "ack_durability", "The API confirms acceptance to the message sender.", "after_durable_commit", False, "after_durable_commit", ["rabbit-confirms-2"]),
        ("host-loss", "development", "ack_durability", "A machine and its local disk can be permanently lost.", "survives_failures", ["process_crash"], "survives_failures", ["rabbit-quorum-1"]),
        ("consumer-completion", "train", "ack_durability", "The queue acknowledgement represents completed business processing.", "after_durable_commit", False, "after_durable_commit", ["rabbit-confirms-2"]),
        ("zone-loss", "test", "ack_durability", "The deployment must survive losing one availability zone.", "survives_failures", ["process_crash", "host_loss"], "survives_failures", ["rabbit-quorum-3"]),
        ("durable-queue", "train", "ack_durability", "A durable queue accepts persistent messages.", "after_durable_commit", False, "after_durable_commit", ["rabbit-queues-1"]),
        ("failover", "calibration", "ack_durability", "Failover must recover every acknowledged message.", "survives_failures", [], "survives_failures", ["rabbit-quorum-2"]),
        ("worker-overload", "train", "queue_overload", "Arrival rate stays above worker throughput during a traffic spike.", "policy", "unbounded", "policy", ["google-overload-1"]),
        ("accepted-loss", "development", "queue_overload", "Accepted customer messages must remain available during overload.", "preserves_accepted_messages", False, "preserves_accepted_messages", ["google-overload-3"]),
        ("hot-channel", "train", "queue_overload", "One busy conversation dominates a storage partition.", "policy", "unbounded", "max_messages", ["discord-storage-1"]),
        ("tenant-burst", "test", "queue_overload", "A tenant sends a sustained burst into a shared bounded worker service.", "preserves_accepted_messages", False, "admission_before_accept", ["google-overload-2"]),
        ("retry-amplification", "train", "queue_overload", "Many clients retry after a dependency timeout.", "policy", "unbounded", "retry_backoff", ["aws-retries-1"]),
        ("consumer-prefetch", "train", "queue_overload", "The consumer fleet may buffer work faster than it completes it.", "policy", "unbounded", "max_messages", ["rabbit-confirms-3"]),
    ]
    rows = []
    section = {"retry_safety": "retry", "ack_durability": "acknowledgement", "queue_overload": "queue"}
    for family, split, check, context, bad_key, bad_value, missing_key, refs in families:
        for variant in range(5):
            request = copy.deepcopy(base)
            request["design"]["flows"] = [context]
            request["design"][section[check]]["notes"] = ""
            if family in {"webhook", "notification"}:
                request["design"]["retry"].update(external_effects=True, external_effects_idempotent=True)
            if family == "zone-loss":
                request["requirements"]["survive_failures"].append("zone_loss")
                request["design"]["acknowledgement"]["survives_failures"].append("zone_loss")
            if family == "consumer-completion":
                request["requirements"]["acknowledged_operation"] = "business_effect_committed"
                request["design"]["acknowledgement"]["acknowledged_operation"] = "business_effect_committed"
            verdict, rationale = "supported", "The explicit controls cover the stated requirement under the declared failure model."
            if variant in {1, 4}:
                request["design"][section[check]][bad_key] = bad_value
                verdict = "contradicted"
                rationale = f"The explicitly specified {bad_key} permits a requirement-violating scenario."
            elif variant == 2:
                request["design"][section[check]][missing_key] = None
                verdict = "insufficient_information"
                rationale = f"The required mechanism cannot be established without {missing_key}."
            if variant == 3:
                request["design"]["notes"] = "Synthetic example named Atlas Messaging; the name supplies no architectural guarantee."
            if variant == 4:
                request["design"]["flows"] = ["This is a hypothetical workload: " + context]
            rows.append(labeled(f"{family}-{variant}", family, split, request, check, verdict, rationale, refs))
    (DATA / "pilot.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {"version": "pilot-1", "family_splits": {f[0]: f[1] for f in families},
                "status": "draft_pending_expert", "note": "Variants share a split. These are synthetic pilot candidates, not validated final-test evidence.",
                "counts": {split: sum(row["split"] == split for row in rows)
                           for split in ["train", "development", "calibration", "test"]},
                "company_holdouts": {"planned": ["stripe-idempotency", "salesforce-tenants"],
                    "status": "not_implemented_in_synthetic_pilot; commission independently authored company cases before release"}}
    write_json(DATA / "split_manifest.json", manifest)
    selections = [("ledger-1", "retry_safety"), ("concurrent-workers-2", "retry_safety"),
                  ("inventory-0", "retry_safety"), ("webhook-1", "retry_safety"),
                  ("publisher-acceptance-1", "ack_durability"), ("publisher-acceptance-0", "ack_durability"),
                  ("host-loss-1", "ack_durability"), ("worker-overload-1", "queue_overload"),
                  ("worker-overload-0", "queue_overload"), ("worker-overload-2", "queue_overload")]
    lookup = {row["id"]: row for row in rows}
    regressions = []
    for index, (key, check) in enumerate(selections, 1):
        row = copy.deepcopy(lookup[key])
        row.update(id=f"starter-{index:02d}", origin_case=key, usage="development_regression_only")
        regressions.append(row)
    (DATA / "regressions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in regressions), encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic pilot cases and {len(regressions)} development regressions.")


if __name__ == "__main__":
    main()
