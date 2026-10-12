from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(max_length=8000)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Failure = Literal["process_crash", "host_loss", "zone_loss", "region_loss"]
CheckId = Literal["retry_safety", "ack_durability", "queue_overload"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Verdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient_information"


class Requirements(StrictModel):
    features: list[Text] = Field(default_factory=list, max_length=30)
    prevent_duplicate_effects: bool | None = None
    acknowledged_operation: Text | None = None
    survive_failures: list[Failure] | None = None
    preserve_accepted_messages: bool | None = None
    bounded_queue_required: bool | None = None
    latency_target_ms: Positive | None = None
    notes: Text = ""


class RetryDesign(StrictModel):
    redelivery_possible: bool | None = None
    all_effects_intrinsically_idempotent: bool | None = None
    stable_message_id: bool | None = None
    dedup_enforced: bool | None = None
    dedup_atomic_with_effect: bool | None = None
    dedup_retention_seconds: Nonnegative | None = None
    max_redelivery_age_seconds: Nonnegative | None = None
    external_effects: bool | None = None
    external_effects_idempotent: bool | None = None
    notes: Text = ""


class AckDesign(StrictModel):
    acknowledged_operation: Text | None = None
    after_durable_commit: bool | None = None
    survives_failures: list[Failure] | None = None
    notes: Text = ""


class QueueDesign(StrictModel):
    max_messages: Annotated[int, Field(gt=0)] | None = None
    policy: Literal["backpressure", "reject_before_accept", "drop", "unbounded"] | None = None
    admission_before_accept: bool | None = None
    preserves_accepted_messages: bool | None = None
    retry_backoff: bool | None = None
    notes: Text = ""


class Design(StrictModel):
    components: list[Text] = Field(default_factory=list, max_length=50)
    flows: list[Text] = Field(default_factory=list, max_length=50)
    retry: RetryDesign = Field(default_factory=RetryDesign)
    acknowledgement: AckDesign = Field(default_factory=AckDesign)
    queue: QueueDesign = Field(default_factory=QueueDesign)
    notes: Text = ""


class Workload(StrictModel):
    average_messages_per_second: Nonnegative | None = None
    peak_factor: Annotated[float, Field(ge=1, allow_inf_nan=False)] | None = None
    message_bytes: Annotated[int, Field(gt=0)] | None = None
    average_recipients: Annotated[float, Field(ge=1, allow_inf_nan=False)] | None = None
    retention_days: Nonnegative | None = None
    storage_copies_per_message: Positive | None = None
    replication_factor: Annotated[int, Field(ge=1)] | None = None
    storage_overhead_factor: Annotated[float, Field(ge=1, allow_inf_nan=False)] | None = None
    monthly_growth_fraction: Nonnegative | None = None
    growth_horizon_months: Annotated[int, Field(ge=1, le=120)] | None = None


class Benchmark(StrictModel):
    provenance: Literal["direct_measurement", "comparable_reference"]
    source: Annotated[str, Field(min_length=1, max_length=2000)]
    measured_messages_per_second: Positive
    p95_latency_ms: Positive
    error_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    duration_seconds: Positive
    workload_matches: bool
    configuration_matches: bool
    notes: Text = ""


class Deployment(StrictModel):
    hardware: Text | None = None
    topology: Text | None = None
    configuration: Text | None = None
    usable_storage_bytes: Annotated[int, Field(gt=0)] | None = None
    benchmark: Benchmark | None = None


class ReviewRequest(StrictModel):
    requirements: Requirements
    design: Design
    workload: Workload = Field(default_factory=Workload)
    deployment: Deployment = Field(default_factory=Deployment)

    @model_validator(mode="after")
    def limit_size(self):
        if len(self.model_dump_json().encode("utf-8")) > 100_000:
            raise ValueError("Review input exceeds 100 KB; narrow it to the relevant messaging flows.")
        return self


class ModelDecision(StrictModel):
    verdict: Verdict
    probabilities: dict[Verdict, float] = Field(default_factory=dict)
    calibration_status: Literal["unvalidated", "fitted_unvalidated", "validated"] = "unvalidated"
    checkpoint: str


class Finding(StrictModel):
    check_id: CheckId
    verdict: Verdict
    rule_verdict: Verdict
    reason: str
    facts: dict = Field(default_factory=dict)
    risk: dict
    missing_information: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    verification_steps: list[str] = Field(default_factory=list)
    model_decision: ModelDecision | None = None
    conflict: bool = False


class ReviewResponse(StrictModel):
    schema_version: str = "1.0"
    review_status: Literal["complete", "partial"]
    review_mode: Literal["rules_only", "laya_assisted"]
    checks: list[Finding]
    workload_estimates: dict
    capacity: dict
    evidence: list[dict]
    warnings: list[str]
    versions: dict
    elapsed_ms: float
