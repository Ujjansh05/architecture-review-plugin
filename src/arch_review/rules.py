from .decisions import TEST_STEPS
from .schema import Finding, ReviewRequest, Verdict


def finding(check_id, verdict, reason, facts, missing=()):
    severity = "high" if verdict == Verdict.CONTRADICTED else "unknown" if missing else "conditional"
    consequences = {
        "retry_safety": "Duplicate business writes or external effects.",
        "ack_durability": "Loss of an operation already reported as successful.",
        "queue_overload": "Unbounded resource use, rejected traffic, or loss of accepted messages.",
    }
    return Finding(
        check_id=check_id, verdict=verdict, rule_verdict=verdict, reason=reason, facts=facts,
        risk={"severity": severity, "consequence": consequences[check_id],
              "trigger": reason, "production_failure_probability": None},
        missing_information=list(missing), verification_steps=TEST_STEPS[check_id],
    )


def retry_safety(req: ReviewRequest) -> Finding:
    r, d = req.requirements, req.design.retry
    facts = {"prevent_duplicate_effects": r.prevent_duplicate_effects, **d.model_dump(exclude_none=True)}
    def answer(v, reason, missing=()):
        return finding("retry_safety", v, reason, facts, missing)
    if r.prevent_duplicate_effects is None:
        return answer(Verdict.INSUFFICIENT, "The required duplicate-effect policy is unknown.",
                      ["requirements.prevent_duplicate_effects"])
    if r.prevent_duplicate_effects is False:
        return answer(Verdict.SUPPORTED, "The stated requirement permits duplicate effects; prevention is not established.")
    if d.all_effects_intrinsically_idempotent is True:
        return answer(Verdict.SUPPORTED, "All business effects are explicitly intrinsically idempotent.")
    if d.redelivery_possible is not True:
        return answer(Verdict.INSUFFICIENT, "Retry/redelivery behaviour must be specified across all producers and consumers.",
                      ["design.retry.redelivery_possible"])
    if d.external_effects is True and d.external_effects_idempotent is False:
        return answer(Verdict.CONTRADICTED, "An external effect can be repeated after a crash and redelivery.")
    if d.all_effects_intrinsically_idempotent is False and (
        d.stable_message_id is False or d.dedup_enforced is False or d.dedup_atomic_with_effect is False
    ):
        return answer(Verdict.CONTRADICTED, "A non-idempotent effect lacks stable, enforced, atomic duplicate protection.")
    if (d.dedup_retention_seconds is not None and d.max_redelivery_age_seconds is not None
            and d.dedup_retention_seconds < d.max_redelivery_age_seconds):
        if d.all_effects_intrinsically_idempotent is False:
            return answer(Verdict.CONTRADICTED, "Deduplication expires before the final permitted redelivery.")
    needed = [name for name in ("stable_message_id", "dedup_enforced", "dedup_atomic_with_effect",
                                "dedup_retention_seconds", "max_redelivery_age_seconds", "external_effects")
              if getattr(d, name) is None]
    if d.external_effects is True and d.external_effects_idempotent is None:
        needed.append("external_effects_idempotent")
    if needed:
        return answer(Verdict.INSUFFICIENT, "An identifier alone does not establish protection for every business effect.",
                      ["design.retry." + name for name in needed])
    if (d.stable_message_id and d.dedup_enforced and d.dedup_atomic_with_effect
            and d.dedup_retention_seconds >= d.max_redelivery_age_seconds
            and (d.external_effects is False or d.external_effects_idempotent is True)):
        return answer(Verdict.SUPPORTED, "Enforced atomic deduplication covers the declared retry window and effect boundaries.")
    return answer(Verdict.INSUFFICIENT, "The stated mechanisms do not establish either protection or a definite violating effect.",
                  ["Whether unprotected effects are non-idempotent and can repeat."])


def ack_durability(req: ReviewRequest) -> Finding:
    r, d = req.requirements, req.design.acknowledgement
    facts = {"required_operation": r.acknowledged_operation, "required_failures": r.survive_failures,
             **d.model_dump(exclude_none=True)}
    def answer(v, reason, missing=()):
        return finding("ack_durability", v, reason, facts, missing)
    needed = []
    if r.acknowledged_operation is None:
        needed.append("requirements.acknowledged_operation")
    if r.survive_failures is None:
        needed.append("requirements.survive_failures")
    if d.acknowledged_operation is None:
        needed.append("design.acknowledgement.acknowledged_operation")
    if needed:
        return answer(Verdict.INSUFFICIENT, "The acknowledgement boundary and required failure model must be specified.", needed)
    if r.acknowledged_operation != d.acknowledged_operation:
        return answer(Verdict.INSUFFICIENT, "The design and requirement describe different acknowledgement boundaries.",
                      ["A matching acknowledged operation in requirements and design."])
    if not r.survive_failures:
        return answer(Verdict.SUPPORTED, "The requirement explicitly imposes no post-acknowledgement failure durability.")
    if d.after_durable_commit is False:
        return answer(Verdict.CONTRADICTED, "Success is acknowledged before durable commit.")
    if d.survives_failures is not None and not set(r.survive_failures).issubset(d.survives_failures):
        return answer(Verdict.CONTRADICTED, "The stated durability guarantee does not cover every required failure domain.")
    needed = ["design.acknowledgement." + name for name in ("after_durable_commit", "survives_failures")
              if getattr(d, name) is None]
    if needed:
        return answer(Verdict.INSUFFICIENT, "Persistence and replication guarantees are incomplete.", needed)
    return answer(Verdict.SUPPORTED, "Acknowledgement follows durable commit covering each explicitly required failure domain.")


def queue_overload(req: ReviewRequest) -> Finding:
    r, d = req.requirements, req.design.queue
    facts = {"preserve_accepted_messages": r.preserve_accepted_messages,
             "bounded_queue_required": r.bounded_queue_required, **d.model_dump(exclude_none=True)}
    def answer(v, reason, missing=()):
        return finding("queue_overload", v, reason, facts, missing)
    needed = ["requirements." + name for name in ("preserve_accepted_messages", "bounded_queue_required")
              if getattr(r, name) is None]
    if needed:
        return answer(Verdict.INSUFFICIENT, "Loss tolerance and queue-bound requirements are unspecified.", needed)
    if r.bounded_queue_required and d.policy == "unbounded":
        return answer(Verdict.CONTRADICTED, "The design explicitly allows unbounded queue growth under overload.")
    if r.preserve_accepted_messages and d.preserves_accepted_messages is False:
        return answer(Verdict.CONTRADICTED, "Overload may discard messages already accepted successfully.")
    needed = ["design.queue." + name for name in ("policy", "admission_before_accept", "retry_backoff")
              if getattr(d, name) is None]
    if r.bounded_queue_required and d.max_messages is None:
        needed.append("design.queue.max_messages")
    if r.preserve_accepted_messages and d.preserves_accepted_messages is None:
        needed.append("design.queue.preserves_accepted_messages")
    if needed:
        return answer(Verdict.INSUFFICIENT, "Queue overload handling is not fully specified.", needed)
    if d.admission_before_accept is False or d.retry_backoff is False:
        return answer(Verdict.INSUFFICIENT, "Admission timing or retry amplification needs an explicit alternative control.",
                      ["A bounded admission and client retry policy consistent with the requirements."])
    if d.policy == "unbounded":
        return answer(Verdict.INSUFFICIENT, "An unbounded policy gives no resource-exhaustion handling mechanism.",
                      ["A resource bound and overload response."])
    return answer(Verdict.SUPPORTED, "The stated queue and admission policies address the declared loss and queue-bound requirements; latency requires testing.")


def check_rules(req: ReviewRequest) -> list[Finding]:
    return [retry_safety(req), ack_durability(req), queue_overload(req)]
