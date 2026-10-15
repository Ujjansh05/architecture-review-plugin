"""Deterministic questions: missing knowledge is not a request to invent a guarantee."""
import hashlib

from .documents import fact_fields
from .schema import Verdict

QUESTIONS = {
    "prevent_duplicate_effects": "Must retries avoid repeating the business action?",
    "redelivery_possible": "Can any producer or consumer retry or redeliver this operation?",
    "all_effects_intrinsically_idempotent": "Does repeating every action have the same effect as doing it once?",
    "stable_message_id": "Does the same request keep the same ID on every retry?",
    "dedup_enforced": "What stops two workers from applying the same message twice?",
    "dedup_atomic_with_effect": "Are duplicate protection and the business write committed together, even if a worker crashes?",
    "dedup_retention_seconds": "How many seconds is a processed message ID remembered?",
    "max_redelivery_age_seconds": "How many seconds after the first attempt can a message still be retried?",
    "external_effects": "Does processing also call another service, send a notification, or make a payment?",
    "external_effects_idempotent": "How does each external service prevent the same action from happening twice?",
    "acknowledged_operation": "What exactly has succeeded when the user receives confirmation?",
    "survive_failures": "Must confirmed operations survive a process crash, lost server, zone, or region?",
    "after_durable_commit": "Is the operation safely stored before the user sees the success confirmation?",
    "survives_failures": "Which failures does the storage guarantee cover after confirmation?",
    "preserve_accepted_messages": "Must every accepted message be retained during overload?",
    "bounded_queue_required": "Must the queue have a fixed size limit?",
    "max_messages": "What is the maximum number of messages the queue can hold?",
    "policy": "When the queue fills up, does the system slow producers, reject new work, or drop messages?",
    "admission_before_accept": "Is space checked before telling the sender its message was accepted?",
    "preserves_accepted_messages": "Are all already accepted messages preserved during overload?",
    "retry_backoff": "Do rejected senders wait before retrying?",
    "atomic_visibility_required": "Must readers see all related changes together, with no partial result?",
    "multiple_writes": "Does this operation change more than one record?",
    "boundary": "Are these writes in one transaction, a distributed transaction, a saga, or separate writes?",
    "covers_all_writes": "Does the transaction include every write needed to keep this operation consistent?",
    "partial_state_visible": "Can another request see only some of the related changes?",
    "concurrent_invariant_protected": "What keeps simultaneous requests from breaking the business rule?",
    "required_consistency": "Must reads show the user's own latest write, every latest write, or can they be delayed?",
    "route": "Can reads go to the primary database, a replica, or either?",
    "replication": "Does a write wait for replication before completing?",
    "read_your_writes_enforced": "How does the user always read their own latest update?",
    "linearizable_reads_enforced": "How do reads always reflect the latest completed write?",
    "stale_reads_possible": "Can any allowed read return an older value than the requirement permits?",
    "max_staleness_ms": "How old may a returned value be, in milliseconds?",
    "worst_case_staleness_ms": "What is the oldest cached value the system can return, in milliseconds?",
    "bound_covers_failures": "Does that cache-age limit still hold when invalidation or the source fails?",
    "stale_on_error": "Does the cache return old data when its source is unavailable?",
    "stale_on_error_bound_ms": "During an outage, how old may a cached value become, in milliseconds?",
    "single_writer_required": "Must only one server be allowed to write at a time?",
    "partition_in_scope": "Must failover remain safe when servers cannot communicate?",
    "old_writer_can_commit_after_promotion": "After a replacement takes over, can the old server still save writes?",
    "fence_enforced_at_storage": "Does storage itself reject writes from the previous owner?",
    "ownership_epoch_monotonic": "Does each new owner receive a strictly increasing ownership number?",
    "deadline_ms": "How many milliseconds may the complete request take?",
    "per_attempt_timeout_ms": "How many milliseconds can each dependency attempt take?",
    "max_attempts": "How many total attempts are allowed, including the first?",
    "total_backoff_ms": "How many milliseconds can all retry waits add up to?",
    "local_work_ms": "How many milliseconds are reserved for work outside the dependency call?",
    "deadline_propagated": "Does downstream work stop when the original request deadline expires?",
    "retry_budget_shared": "Do all layers share one retry limit?",
    "jitter": "Are retry waits spread out so clients do not all retry together?",
}

FOLLOWUPS = {
    "retry_safety": "Describe what happens if a worker saves the result, crashes before confirming it, and receives the same message again.",
    "ack_durability": "Clarify which operation is confirmed and what remains saved after each required failure.",
    "queue_overload": "Describe the queue limit, acceptance boundary, and response when no space remains.",
    "transaction_atomicity": "Describe the transaction boundary and what concurrent readers see if a write fails.",
    "replica_consistency": "Explain the required read guarantee and how every allowed read route enforces it.",
    "cache_freshness": "Describe the maximum cache age, including source outages and failed invalidation.",
    "failover_safety": "Explain how storage rejects writes from an old owner after a replacement takes over.",
    "dependency_resilience": "Describe the timeout, total attempts, retry waits, and cancellation across the whole call path.",
}

ACTIONS = {
    "retry_safety": "Consider enforcing a stable request ID atomically with each protected write; cover the entire retry window and external effects, or establish intrinsic idempotence.",
    "ack_durability": "Move success confirmation after the durable commit that covers the exact operation and required failures.",
    "queue_overload": "Set a queue bound and enforce admission before acceptance; preserve accepted work and control retry load.",
    "transaction_atomicity": "Place the required invariant inside an appropriate atomic boundary and protect it against concurrent writes.",
    "replica_consistency": "Choose and verify a read route that enforces the requested consistency guarantee, including failover.",
    "cache_freshness": "Bound returned cache age on both normal and failure paths, or revise the freshness requirement explicitly.",
    "failover_safety": "Enforce monotonically increasing ownership tokens at storage and reject the old writer after promotion.",
    "dependency_resilience": "Fit attempts, retry waits, and local work into the end-to-end deadline; propagate cancellation and share the retry budget.",
}


def questions_for(finding):
    fields = fact_fields(finding.check_id)
    questions = []
    for missing in dict.fromkeys(finding.missing_information):
        path = missing.removeprefix("Conflicting fact: ")
        field = fields.get(path)
        if field is not None:
            question = QUESTIONS.get(path.rsplit(".", 1)[-1], FOLLOWUPS[finding.check_id])
            if missing.startswith("Conflicting fact: "):
                question = "The sources disagree. " + question
        elif missing == "applicable":
            question = "Does this requirement apply to this operation?"
            field = {"type": ["boolean", "null"]}
        elif "disagreement" in missing or "Conflicting" in missing or "inconsistent" in missing:
            question = "Which declaration is correct? Resolve the conflicting facts before changing the design."
        elif "document sections" in missing:
            question = "Can you supply the missing design facts or review the unprocessed document sections?"
        else:
            question = FOLLOWUPS[finding.check_id]
        identity = "\0".join([finding.scope_id, finding.check_id, missing])
        questions.append({"id": hashlib.sha256(identity.encode()).hexdigest()[:20],
            "check_id": finding.check_id, "scope_id": finding.scope_id,
            "fact_path": path if field is not None or path == "applicable" else None,
            "missing_detail": missing, "question": question, "answer_schema": field or {"type": ["string", "null"]},
            "why": finding.reason, "unknown_allowed": True})
    return questions


def enrich(finding):
    finding.clarification_questions = questions_for(finding)
    if finding.verdict == Verdict.CONTRADICTED:
        finding.suggested_action = ACTIONS[finding.check_id]
    elif finding.verdict == Verdict.INSUFFICIENT:
        finding.suggested_action = "Answer the missing questions from the actual design; keep unknown answers unresolved."
    else:
        finding.suggested_action = None


def next_questions(findings):
    # Fair across operations rather than exhausting one long list before another scope is heard.
    groups = [f.clarification_questions for f in findings if f.clarification_questions]
    return [q for index in range(max(map(len, groups), default=0))
            for group in groups if index < len(group) for q in [group[index]]][:3]
