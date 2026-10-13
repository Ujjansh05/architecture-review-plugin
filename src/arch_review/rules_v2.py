"""Conservative checks over explicit declarations, with concrete counterexamples."""
from .registry import REGISTRY
from .rules import check_rules
from .schema import ReviewRequest, Verdict
from .schema_v2 import BackendFinding, CheckInput


def evaluate_check(item: CheckInput) -> BackendFinding:
    definition = REGISTRY[item.check_id]
    facts = definition.facts_type.model_validate(item.facts)

    def result(verdict, reason, missing=(), status="evaluated"):
        return BackendFinding(check_id=item.check_id, scope_id=item.scope_id,
            evaluation_status=status, verdict=verdict, rule_verdict=verdict, reason=reason,
            facts=item.facts, supporting_spans=item.fact_spans,
            missing_information=list(missing), verification_steps=list(definition.tests))

    if item.applicable is False:
        return result(None, "The caller explicitly marked this check inapplicable.", status="not_evaluated")
    if item.applicable is None:
        return result(Verdict.INSUFFICIENT, "Applicability is not established.", ["applicable"])
    if item.check_id in {"retry_safety", "ack_durability", "queue_overload"}:
        old = next(f for f in check_rules(ReviewRequest.model_validate(item.facts)) if f.check_id == item.check_id)
        return result(old.verdict, old.reason, old.missing_information)

    missing = [key for key in type(facts).model_fields if getattr(facts, key) is None]
    supported, contradiction = False, None
    key = item.check_id
    if key == "transaction_atomicity":
        if facts.atomic_visibility_required is False:
            return result(None, "Atomic visibility is explicitly not required; other consistency concerns are outside this check.",
                          status="not_evaluated")
        if facts.atomic_visibility_required:
            if facts.partial_state_visible is True and facts.multiple_writes is True:
                contradiction = "Partial writes are visible despite the atomic-visibility requirement."
            elif facts.concurrent_invariant_protected is False:
                contradiction = "Concurrent writes are explicitly allowed to violate the required invariant."
            supported = (facts.boundary in {"single_transaction", "distributed_transaction"}
                         and facts.covers_all_writes is True and facts.partial_state_visible is False
                         and facts.concurrent_invariant_protected is True)
            if not supported and not missing and not contradiction:
                missing = ["An atomic boundary covering the invariant; compensation alone is not atomic visibility."]
    elif key == "replica_consistency":
        if facts.required_consistency == "eventual":
            return result(None, "No stronger read consistency is requested; eventual convergence is not checked here.",
                          status="not_evaluated")
        if facts.required_consistency is not None:
            required = (facts.read_your_writes_enforced if facts.required_consistency == "read_your_writes"
                        else facts.linearizable_reads_enforced)
            if required is True and facts.stale_reads_possible is True:
                return result(Verdict.INSUFFICIENT,
                              "The facts both guarantee and permit violation of the requested read consistency.",
                              ["Resolve inconsistent read guarantee and stale-read declarations."])
            if facts.stale_reads_possible is True:
                contradiction = "A permitted read can violate the explicitly required consistency guarantee."
            supported = required is True and facts.route is not None
            missing = ["required read guarantee and route"] if not supported and not contradiction else []
    elif key == "cache_freshness":
        if facts.max_staleness_ms is not None:
            limits = [facts.worst_case_staleness_ms]
            if facts.stale_on_error is True:
                limits.append(facts.stale_on_error_bound_ms)
            if any(value is not None and value > facts.max_staleness_ms for value in limits):
                contradiction = "The declared cache-age bound exceeds the permitted staleness."
            supported = (all(value is not None for value in limits) and facts.bound_covers_failures is True
                         and facts.stale_on_error is not None)
            if facts.stale_on_error is False:
                missing = [v for v in missing if v != "stale_on_error_bound_ms"]
    elif key == "failover_safety":
        if facts.single_writer_required is False:
            return result(None, "Single-writer ownership is not requested; multi-writer conflict resolution needs another check.",
                          status="not_evaluated")
        if facts.single_writer_required is True:
            if facts.old_writer_can_commit_after_promotion is True:
                contradiction = "The old writer can still commit after a successor is promoted."
            supported = (facts.old_writer_can_commit_after_promotion is False
                         and facts.fence_enforced_at_storage is True and facts.ownership_epoch_monotonic is True
                         and facts.partition_in_scope is not None)
    elif key == "dependency_resilience":
        numeric = [facts.deadline_ms, facts.per_attempt_timeout_ms, facts.max_attempts,
                   facts.total_backoff_ms, facts.local_work_ms]
        if all(v is not None for v in numeric):
            worst = facts.per_attempt_timeout_ms * facts.max_attempts + facts.total_backoff_ms + facts.local_work_ms
            if worst > facts.deadline_ms and facts.deadline_propagated is False:
                contradiction = f"The serial retry path can take {worst:g} ms, exceeding the {facts.deadline_ms:g} ms deadline."
            supported = (worst <= facts.deadline_ms and facts.deadline_propagated is True
                         and facts.retry_budget_shared is True and (facts.max_attempts == 1 or facts.jitter is True))
            if facts.max_attempts == 1:
                missing = [v for v in missing if v != "jitter"]
        if not supported and not missing and not contradiction:
            missing = ["A bounded retry policy and deadline propagation covering the entire call path."]
    if contradiction:
        return result(Verdict.CONTRADICTED, contradiction)
    if supported:
        return result(Verdict.SUPPORTED, "Explicit facts satisfy this check under the stated scope; verify the mechanism in deployment.")
    return result(Verdict.INSUFFICIENT, "The declared facts do not establish the required guarantee.", missing)
