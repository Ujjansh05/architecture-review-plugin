import math

from .schema import ReviewRequest


def calculate(req: ReviewRequest) -> tuple[dict, dict]:
    w = req.workload
    estimates, missing = {}, []
    def add(name, fields, formula, compute, unit):
        absent = [key for key in fields if getattr(w, key) is None]
        if absent:
            missing.extend(absent)
            return
        try:
            value = float(compute())
        except OverflowError:
            value = math.inf
        if not math.isfinite(value):
            missing.append(name + ": inputs exceed finite arithmetic range")
            return
        estimates[name] = {"value": value, "unit": unit, "formula": formula,
                           "assumptions": {key: getattr(w, key) for key in fields}}
    add("peak_messages", ["average_messages_per_second", "peak_factor"], "average_rate * peak_factor",
        lambda: w.average_messages_per_second * w.peak_factor, "messages/second")
    add("average_deliveries", ["average_messages_per_second", "average_recipients"], "average_rate * recipients",
        lambda: w.average_messages_per_second * w.average_recipients, "deliveries/second")
    add("peak_deliveries", ["average_messages_per_second", "average_recipients", "peak_factor"],
        "average_rate * recipients * peak_factor",
        lambda: w.average_messages_per_second * w.average_recipients * w.peak_factor, "deliveries/second")
    add("daily_payload", ["average_messages_per_second", "message_bytes"], "average_rate * 86400 * message_bytes",
        lambda: w.average_messages_per_second * 86400 * w.message_bytes, "bytes/day")
    storage_fields = ["average_messages_per_second", "message_bytes", "retention_days",
                      "storage_copies_per_message", "replication_factor", "storage_overhead_factor"]
    add("retained_storage", storage_fields,
        "average_rate * 86400 * message_bytes * retention_days * stored_copies * replication * overhead",
        lambda: w.average_messages_per_second * 86400 * w.message_bytes * w.retention_days
        * w.storage_copies_per_message * w.replication_factor * w.storage_overhead_factor, "bytes")
    add("projected_average_messages", ["average_messages_per_second", "monthly_growth_fraction", "growth_horizon_months"],
        "average_rate * (1 + monthly_growth_fraction) ** horizon_months",
        lambda: w.average_messages_per_second * (1 + w.monthly_growth_fraction) ** w.growth_horizon_months,
        "messages/second")
    capacity = {"evidence_status": "unknown", "supported_capacity": None, "headroom_ratio": None,
                "latency_slo_status": "unknown", "missing_evidence": [],
                "required_tests": ["Benchmark the same message sizes, fan-out, durability, configuration and error budget.",
                                   "Test sustained overload, failure recovery and p95/p99 latency at peak traffic."]}
    b = req.deployment.benchmark
    if b is None:
        capacity["missing_evidence"].append("A benchmark with workload, configuration, latency, errors and duration.")
    else:
        capacity["benchmark"] = b.model_dump()
        if b.workload_matches and b.configuration_matches:
            capacity["evidence_status"] = "measured" if b.provenance == "direct_measurement" else "comparable_estimate"
            capacity["supported_capacity"] = {"value": b.measured_messages_per_second, "unit": "messages/second",
                                               "scope": "Supplied benchmark conditions; provenance not independently verified."}
            peak = estimates.get("peak_messages", {}).get("value")
            if peak is not None and peak > 0:
                capacity["headroom_ratio"] = b.measured_messages_per_second / peak
            if req.requirements.latency_target_ms is not None:
                capacity["latency_slo_status"] = "met_in_supplied_benchmark" if b.p95_latency_ms <= req.requirements.latency_target_ms else "violated_in_supplied_benchmark"
        else:
            capacity["missing_evidence"].append("Benchmark workload/configuration does not match the proposed deployment.")
    if req.deployment.usable_storage_bytes is not None and "retained_storage" in estimates:
        required = estimates["retained_storage"]["value"]
        capacity["storage_requirement_exceeds_available"] = required > req.deployment.usable_storage_bytes
        capacity["storage_comparison_scope"] = "Steady workload over the stated retention window; excludes unprovided copies and growth."
    return {"estimates": estimates, "missing_inputs": sorted(set(missing)),
            "notes": ["Recipient fan-out does not multiply storage unless separate stored copies are specified.",
                      "Retained storage assumes steady arrivals across the retention window.",
                      "Growth projections are scenarios, not forecasts; no user-count capacity is inferred."]}, capacity
