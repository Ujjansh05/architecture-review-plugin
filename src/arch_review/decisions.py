from .schema import Verdict

DEFINITIONS = {
    "retry_safety": "Retries and redelivery preserve the required protection against duplicate business effects.",
    "ack_durability": "Success acknowledgements occur only after the required durability has been achieved.",
    "queue_overload": "Queue overload handling is consistent with accepted-message preservation and queue bounds.",
}
CRITERIA = {
    Verdict.SUPPORTED.value: "Explicit design facts satisfy the stated requirement under its assumptions.",
    Verdict.CONTRADICTED.value: "Explicit facts show a failure scenario that violates the requirement.",
    Verdict.INSUFFICIENT.value: "Necessary facts or requirements are missing, ambiguous, or conflicting.",
}


def question(check_id: str) -> dict:
    return {check_id: {"type": "choice", "instructions": DEFINITIONS[check_id], "criteria": CRITERIA.copy()}}


TEST_STEPS = {
    "retry_safety": [
        "Crash the worker immediately after the business effect and before queue acknowledgement; redeliver.",
        "Deliver the same ID concurrently and again at the maximum retry age; count durable business effects.",
        "Repeat the failure injection at every external API boundary, including its idempotency expiry.",
    ],
    "ack_durability": [
        "Record successful acknowledgements, inject each required failure, and reconcile recovered messages.",
        "Verify commit, replication and failover configuration for the exact acknowledged operation.",
    ],
    "queue_overload": [
        "Sustain arrivals above measured service rate; record queue size, memory, rejection rate and latency.",
        "Verify that accepted messages survive overload and clients back off after rejected submissions.",
        "Recover from overload and verify bounded drain time and absence of retry amplification.",
    ],
}
