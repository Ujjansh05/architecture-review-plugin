"""Versioned, bounded backend decisions. Facts are declarations, not deployment proof."""
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from .schema import StrictModel


class MessagingFacts(StrictModel):
    # V1's exact schema and rules are reused for the three legacy decisions.
    requirements: dict = Field(default_factory=dict)
    design: dict = Field(default_factory=dict)


class TransactionFacts(StrictModel):
    atomic_visibility_required: bool | None = None
    multiple_writes: bool | None = None
    boundary: Literal["single_transaction", "distributed_transaction", "saga", "independent_writes"] | None = None
    covers_all_writes: bool | None = None
    partial_state_visible: bool | None = None
    concurrent_invariant_protected: bool | None = None


class ReplicaFacts(StrictModel):
    required_consistency: Literal["eventual", "read_your_writes", "linearizable"] | None = None
    route: Literal["primary", "replica", "either"] | None = None
    replication: Literal["synchronous", "asynchronous"] | None = None
    read_your_writes_enforced: bool | None = None
    linearizable_reads_enforced: bool | None = None
    stale_reads_possible: bool | None = Field(default=None,
        description="A permitted read can be stale in a way that violates the requested consistency guarantee.")


class CacheFacts(StrictModel):
    max_staleness_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    worst_case_staleness_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    bound_covers_failures: bool | None = None
    stale_on_error: bool | None = None
    stale_on_error_bound_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class FailoverFacts(StrictModel):
    single_writer_required: bool | None = None
    partition_in_scope: bool | None = None
    old_writer_can_commit_after_promotion: bool | None = None
    fence_enforced_at_storage: bool | None = None
    ownership_epoch_monotonic: bool | None = None


class DependencyFacts(StrictModel):
    deadline_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    per_attempt_timeout_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_attempts: int | None = Field(default=None, ge=1, le=1000, strict=True)
    total_backoff_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    local_work_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    deadline_propagated: bool | None = None
    retry_budget_shared: bool | None = None
    jitter: bool | None = None


@dataclass(frozen=True)
class Decision:
    title: str
    definition: str
    facts_type: type[StrictModel]
    tags: tuple[str, ...]
    tests: tuple[str, ...]


REGISTRY = {
    "retry_safety": Decision("Retry safety", "Retries must not repeat protected business effects.",
        MessagingFacts, ("idempotency", "retry", "deduplication"),
        ("Crash after effect and before acknowledgement, then redeliver concurrently and at retry expiry.",)),
    "ack_durability": Decision("Acknowledgement durability", "Acknowledged operations survive the required failures.",
        MessagingFacts, ("durability", "commit", "replication"),
        ("Inject every declared failure immediately after acknowledgement and reconcile accepted operations.",)),
    "queue_overload": Decision("Queue overload", "Overload preserves declared queue bounds and accepted messages.",
        MessagingFacts, ("queue", "backpressure", "admission"),
        ("Overload arrival rate, measure queue growth and reconcile accepted messages during recovery.",)),
    "transaction_atomicity": Decision("Transaction atomicity",
        "All writes covered by an atomic-visibility requirement preserve its invariant under partial failure and concurrency.",
        TransactionFacts, ("transaction", "atomicity", "isolation"),
        ("Crash between writes and read concurrently; observe whether a partial invariant becomes visible.",)),
    "replica_consistency": Decision("Replica read consistency",
        "Read routing enforces the required read guarantee, including after failover.",
        ReplicaFacts, ("replica", "consistency", "read"),
        ("Delay replication, write, then read through each permitted route and repeat after failover.",)),
    "cache_freshness": Decision("Cache freshness",
        "The stated worst-case cache age, including failure paths, stays within the required staleness bound.",
        CacheFacts, ("cache", "invalidation", "staleness"),
        ("Delay invalidation and fail the source while measuring the oldest value returned.",)),
    "failover_safety": Decision("Failover writer safety",
        "When single-writer ownership is required, an old writer cannot commit after promotion of its successor.",
        FailoverFacts, ("failover", "fencing", "partition"),
        ("Partition the leader, promote a successor, and attempt writes from both ownership epochs.",)),
    "dependency_resilience": Decision("Dependency resilience",
        "A serial dependency call's timeouts, total retry budget and backoff fit the end-to-end deadline.",
        DependencyFacts, ("timeout", "retry", "deadline"),
        ("Delay each attempt to its timeout; verify propagated cancellation, total attempts and retry spread.",)),
}

REGISTRY_VERSION = "backend-2.0"


def definitions():
    from .documents import fact_fields
    return {key: {"title": d.title, "definition": d.definition,
                  "facts_schema": d.facts_type.model_json_schema(), "extraction_paths": fact_fields(key),
                  "tests": list(d.tests)}
            for key, d in REGISTRY.items()}
