"""Additive V2 contracts; V1 remains unchanged."""
from typing import Literal

from pydantic import Field, model_validator

from .registry import REGISTRY
from .schema import Deployment, ReviewRequest, StrictModel, Verdict, Workload

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_DOCUMENT_TOKENS = 128_000
REQUIREMENTS_DOCUMENT_ID = "__requirements__"


class Document(StrictModel):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_.-]+$")
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_BYTES)
    format: Literal["text", "markdown", "json"] = "text"


class Span(StrictModel):
    document_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class CheckInput(StrictModel):
    check_id: str
    scope_id: str = Field(default="design", min_length=1, max_length=128)
    applicable: bool | None = True
    facts: dict = Field(default_factory=dict)
    fact_spans: dict[str, list[Span]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_facts(self):
        if self.check_id not in REGISTRY:
            raise ValueError(f"Unknown check: {self.check_id}")
        obj = REGISTRY[self.check_id].facts_type.model_validate(self.facts)
        if self.check_id in {"retry_safety", "ack_durability", "queue_overload"}:
            ReviewRequest.model_validate(obj.model_dump())
        self.facts = obj.model_dump(exclude_none=True)
        return self


class DesignReviewRequest(StrictModel):
    requirements: str = Field(default="", max_length=16_000)
    documents: list[Document] = Field(default_factory=list, max_length=64)
    checks: list[CheckInput] = Field(default_factory=list, max_length=128)
    requested_checks: list[str] = Field(default_factory=lambda: list(REGISTRY), max_length=8)
    workload: Workload = Field(default_factory=Workload)
    deployment: Deployment = Field(default_factory=Deployment)

    @model_validator(mode="after")
    def validate_request(self):
        if len(self.model_dump_json().encode("utf-8")) > MAX_DOCUMENT_BYTES:
            raise ValueError("V2 request exceeds 2 MiB")
        if not self.documents and not self.checks:
            raise ValueError("Provide documents or explicit structured checks")
        if not self.requested_checks or len(set(self.requested_checks)) != len(self.requested_checks):
            raise ValueError("requested_checks must be nonempty and unique")
        if set(self.requested_checks) - REGISTRY.keys():
            raise ValueError("requested_checks contains an unsupported check")
        ids = [d.id for d in self.documents]
        if REQUIREMENTS_DOCUMENT_ID in ids:
            raise ValueError("__requirements__ is reserved for the requirements text and its supporting spans")
        scopes = [(c.scope_id, c.check_id) for c in self.checks]
        if len(ids) != len(set(ids)) or len(scopes) != len(set(scopes)):
            raise ValueError("Document IDs and scope/check pairs must be unique")
        return self


class BackendJudgment(StrictModel):
    verdict: Verdict
    reason: str = Field(default="", max_length=4000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    missing_information: list[str] = Field(default_factory=list, max_length=20)


class BackendFinding(StrictModel):
    check_id: str
    scope_id: str
    evaluation_status: Literal["evaluated", "not_evaluated"] = "evaluated"
    verdict: Verdict | None
    rule_verdict: Verdict | None
    reason: str
    facts: dict = Field(default_factory=dict)
    supporting_spans: dict[str, list[Span]] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    verification_steps: list[str] = Field(default_factory=list)
    model_decision: dict | None = None
    conflict: bool = False
    production_failure_probability: None = None


class DesignReviewResponse(StrictModel):
    schema_version: str = "2.0"
    review_status: Literal["complete", "partial"]
    checks: list[BackendFinding]
    coverage: dict
    evidence: list[dict]
    workload_estimates: dict
    capacity: dict
    backend: dict
    resources: dict = Field(default_factory=dict)
    warnings: list[str]
    elapsed_ms: float
