"""Versioned experimental input semantics; does not change the released MCP contract."""
import json

from .decision_input import decision_input
from .documents import flatten
from .registry import REGISTRY
from .schema import AckDesign, QueueDesign, RetryDesign

VERSION = "research-facts-1"
PRESENTATIONS = ("legacy", "implicit", "explicit")
CACHE_NAMES = {
    "max_staleness_ms": "required_max_age_ms",
    "worst_case_staleness_ms": "normal_path_max_age_ms",
    "bound_covers_failures": "other_failure_paths_covered",
    "stale_on_error": "stale_fallback_enabled",
    "stale_on_error_bound_ms": "stale_fallback_max_age_ms",
}
CACHE_DEFINITIONS = {
    "required_max_age_ms": "Required maximum age of any returned value, measured from its source timestamp.",
    "normal_path_max_age_ms": "Declared age bound on the normal serving path; excludes stale-on-error fallback.",
    "other_failure_paths_covered": "Whether the declared bounds cover other scoped failures; this does not supply a missing fallback-age bound.",
    "stale_fallback_enabled": "Whether stale-on-error fallback is permitted; its age is bounded separately.",
    "stale_fallback_max_age_ms": "Maximum returned age on stale-on-error fallback, if enabled; not required when disabled.",
}


def field_inventory(check):
    if check == "retry_safety":
        return ["requirements.prevent_duplicate_effects", *["design.retry." + k for k in RetryDesign.model_fields if k != "notes"]]
    if check == "ack_durability":
        return ["requirements.acknowledged_operation", "requirements.survive_failures",
                *["design.acknowledgement." + k for k in AckDesign.model_fields if k != "notes"]]
    if check == "queue_overload":
        return ["requirements.preserve_accepted_messages", "requirements.bounded_queue_required",
                *["design.queue." + k for k in QueueDesign.model_fields if k != "notes"]]
    return list(REGISTRY[check].facts_type.model_fields)


def render(check, facts, evidence, context="", presentation="implicit"):
    if presentation not in PRESENTATIONS:
        raise ValueError("Unknown fact presentation")
    payload = decision_input(check, facts, evidence, context)
    if presentation == "legacy":
        return payload
    state = json.loads(payload["state"])
    visible = flatten(state["facts"])
    unknown = [p for p in field_inventory(check) if visible.get(p) is None]
    if check == "cache_freshness":
        state["facts"] = {CACHE_NAMES[k]: value for k, value in state["facts"].items()}
        unknown = [CACHE_NAMES[k] for k in unknown]
        state["field_definitions"] = CACHE_DEFINITIONS
    state["input_contract"] = VERSION
    if presentation == "explicit":
        # This lists every unsupplied field, not an oracle list of decisive facts.
        # It is derived identically at train and inference time from visible input.
        state["unspecified_fields"] = sorted(unknown)
    # Keep the policy in the validated state budget, outside the independently
    # truncated question head. Both research arms receive the identical policy.
    state["decision_policy"] = (
        "Unsupplied facts are unknown, not false. Missing optional or inapplicable facts need not prevent a decision."
        " An explicit violation can establish contradiction despite other missing facts."
        " Background evidence describes general mechanisms; it does not establish this design's configuration.")
    payload["state"] = json.dumps(state, ensure_ascii=False, sort_keys=True)
    return payload


def render_row(row, case, presentation):
    check = case["check"]
    return {**row, **render(check["check_id"], check["facts"], json.loads(row["state"])["evidence"],
                           check.get("context", ""), presentation)}
