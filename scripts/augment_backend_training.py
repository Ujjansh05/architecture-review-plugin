"""Controlled train-only representation augmentation of frozen diagnostic cases.

This creates no new independently labeled scenarios. Original questions, evidence,
labels and scenario families are inherited; all held-out split bytes are preserved.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from arch_review.backend_dataset import audit_cases
from arch_review.documents import flatten
from arch_review.evaluation_v2 import validate_backend_export

SPLITS = ("train", "development", "calibration", "test")
HELD_OUT = SPLITS[1:]
VERDICTS = ("supported", "contradicted", "insufficient_information")
PINNED_SOURCE_SPLITS = {
    "train": "209a2a955a85396346ba3f785c4e5102fa3546827c1d4a325a74fb563edb9775",
    "development": "650fe3b5658046bad6325b3d255cf5714f5f7f7f451dca2bf056e412b07bc953",
    "calibration": "0f18591c64b455f3332fad5e019f6ad179c1dd51e48b8a3891b15131bc01d3ef",
    "test": "073f13110f04d328880b5226e395eba3ae55437fdaa2f6b216ab381ab2c92b6e",
}
PINNED_SOURCE_CASES = "38599572b811c23500f5947fba0323f45278b331273d7f55f1b387d00ebaa255"

# Every pair is (explicit true declaration, explicit false declaration).
BOOLEAN_PROSE = {
    "requirements.prevent_duplicate_effects": (
        "Retries must not repeat protected business effects.",
        "The requirements do not demand prevention of repeated business effects."),
    "design.retry.redelivery_possible": ("A message can be delivered again.", "Messages cannot be redelivered."),
    "design.retry.all_effects_intrinsically_idempotent": (
        "Every business effect is inherently idempotent.", "At least one business effect is not inherently idempotent."),
    "design.retry.stable_message_id": ("Retries reuse a stable message identifier.",
                                      "Retries do not reuse a stable message identifier."),
    "design.retry.dedup_enforced": ("Duplicate detection is enforced.", "Duplicate detection is not enforced."),
    "design.retry.dedup_atomic_with_effect": (
        "Duplicate detection and the protected effect share one atomic commit.",
        "Duplicate detection and the protected effect do not share one atomic commit."),
    "design.retry.external_effects": ("The operation produces business effects in external systems.",
                                      "The operation has no business effects in external systems."),
    "design.acknowledgement.after_durable_commit": (
        "Success is acknowledged only after durable commit.", "Success can be acknowledged before durable commit."),
    "requirements.bounded_queue_required": ("The queue must have a finite size bound.",
                                            "The requirements do not demand a finite queue size bound."),
    "requirements.preserve_accepted_messages": (
        "Messages already accepted must survive queue overload.",
        "The requirements do not demand preservation of accepted messages during overload."),
    "design.queue.admission_before_accept": ("Admission limits are checked before an arrival is accepted.",
                                            "An arrival can be accepted before admission limits are checked."),
    "design.queue.preserves_accepted_messages": ("Queue overload does not discard messages already accepted.",
                                                "Queue overload can discard messages already accepted."),
    "design.queue.retry_backoff": ("Clients delay retries using backoff.", "Clients retry without backoff."),
    "atomic_visibility_required": ("Readers must observe the operation's writes atomically.",
                                   "Atomic visibility is not a stated requirement."),
    "multiple_writes": ("The operation performs multiple writes.", "The operation performs at most one write."),
    "covers_all_writes": ("The declared transaction boundary covers every write in the operation.",
                          "At least one operation write lies outside the declared transaction boundary."),
    "partial_state_visible": ("Readers can observe a partly applied operation.",
                              "Readers cannot observe a partly applied operation."),
    "concurrent_invariant_protected": ("Concurrent access preserves the declared invariant.",
                                       "The declared invariant is not protected against concurrent access."),
    "read_your_writes_enforced": ("Read routing enforces read-your-writes.",
                                  "Read routing does not enforce read-your-writes."),
    "linearizable_reads_enforced": ("Reads have an enforced linearizability guarantee.",
                                     "Reads do not have an enforced linearizability guarantee."),
    "stale_reads_possible": ("A permitted read can be stale enough to violate the required read guarantee.",
                             "No permitted read can be stale in a way that violates the required read guarantee."),
    "bound_covers_failures": ("The declared cache-age bound also covers failure paths.",
                              "The declared cache-age bound does not cover failure paths."),
    "stale_on_error": ("The cache can serve stale values when its source fails.",
                       "The cache does not serve stale values when its source fails."),
    "single_writer_required": ("Only one writer may own the right to commit.",
                               "The requirements do not demand single-writer ownership."),
    "partition_in_scope": ("Network partitions are included in the failure scope.",
                           "Network partitions are outside the declared failure scope."),
    "old_writer_can_commit_after_promotion": ("The former writer can still commit after its successor is promoted.",
                                              "The former writer cannot commit after its successor is promoted."),
    "fence_enforced_at_storage": ("Storage rejects writes that fail the ownership fence.",
                                  "Storage does not enforce the ownership fence."),
    "ownership_epoch_monotonic": ("Ownership epochs increase monotonically.",
                                  "Ownership epochs are not guaranteed to increase monotonically."),
    "deadline_propagated": ("The end-to-end deadline is propagated to the dependency call.",
                            "The end-to-end deadline is not propagated to the dependency call."),
    "retry_budget_shared": ("The call chain shares the declared retry budget.",
                             "The call chain does not share one declared retry budget."),
    "jitter": ("Retry delays include random jitter.", "Retry delays do not include random jitter."),
}
NUMERIC_PROSE = {
    "design.retry.dedup_retention_seconds": "The duplicate record is retained for {value} seconds.",
    "design.retry.max_redelivery_age_seconds": "The maximum redelivery age is {value} seconds.",
    "design.queue.max_messages": "The queue holds at most {value} messages.",
    "max_staleness_ms": "The allowed maximum staleness is {value} milliseconds.",
    "worst_case_staleness_ms": "The declared worst-case staleness is {value} milliseconds.",
    "deadline_ms": "The end-to-end deadline is {value} milliseconds.",
    "per_attempt_timeout_ms": "Each dependency attempt times out after {value} milliseconds.",
    "max_attempts": "The maximum number of dependency attempts is {value}.",
    "total_backoff_ms": "The total retry backoff budget is {value} milliseconds.",
    "local_work_ms": "Local work takes {value} milliseconds within the deadline budget.",
}
ENUM_PROSE = {
    "design.queue.policy": {"reject_before_accept": "When the queue is full, ingress rejects arrivals before accepting them."},
    "boundary": {"single_transaction": "The operation uses one transaction boundary."},
    "required_consistency": {"read_your_writes": "Reads must reflect this client's completed writes."},
    "route": {"replica": "Reads are routed to a replica."},
    "replication": {"asynchronous": "Replication is asynchronous."},
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def canonical_digest(value):
    return digest(json.dumps(value, sort_keys=True).encode())


def sentence(path, value):
    if path in BOOLEAN_PROSE:
        if type(value) is not bool:
            raise ValueError(f"Expected an explicit boolean for {path}")
        return BOOLEAN_PROSE[path][0 if value else 1]
    if path in NUMERIC_PROSE:
        if type(value) not in (int, float):
            raise ValueError(f"Expected an explicit number for {path}")
        return NUMERIC_PROSE[path].format(value=json.dumps(value, allow_nan=False))
    if path in ENUM_PROSE and value in ENUM_PROSE[path]:
        return ENUM_PROSE[path][value]
    if path in ("requirements.acknowledged_operation", "design.acknowledgement.acknowledged_operation"):
        if not isinstance(value, str) or not value:
            raise ValueError("Acknowledged operation must remain explicit")
        prefix = "The required acknowledgement boundary is" if path.startswith("requirements.") else (
            "The design acknowledges the operation named")
        return f"{prefix} {json.dumps(value, ensure_ascii=False)}."
    if path in ("requirements.survive_failures", "design.acknowledgement.survives_failures"):
        names = {"process_crash": "process crash", "host_loss": "host loss"}
        if not isinstance(value, list) or not value or any(item not in names for item in value):
            raise ValueError("Unexpected failure scope; add a reviewed field-specific rendering")
        failures = " and ".join(names[item] for item in value)
        prefix = "Acknowledged operations must survive" if path.startswith("requirements.") else (
            "The design declares that acknowledged operations survive")
        return f"{prefix} {failures}."
    raise ValueError(f"No explicit rendering for {path}; do not guess or invent a guarantee")


def prose_description(case):
    if case["split"] != "train":
        raise ValueError("Only training cases may be augmented")
    marker = "\nExplicit design declarations:\n"
    if marker not in case["description"]:
        raise ValueError("Unexpected source description format")
    context = case["description"].split(marker, 1)[0]
    sentences = [sentence(path, value) for path, value in flatten(case["check"]["facts"]).items()]
    return context + "\n" + " ".join(sentences)


def make_variants(case, row):
    if case["split"] != "train" or case["id"] != row["id"]:
        raise ValueError("Only aligned training cases may be augmented")
    variants = []
    for kind, description in (
        ("prose_v1", prose_description(case)),
        ("json_facts_v1", json.dumps(case["check"]["facts"], ensure_ascii=False, sort_keys=True)),
    ):
        derived_case, derived_row = copy.deepcopy(case), copy.deepcopy(row)
        identifier = case["id"] + "--" + kind
        derived_case.update(id=identifier, description=description, augmentation={
            "kind": "representation_only", "representation": kind, "parent_case_id": case["id"],
            "parent_split": "train", "new_independent_case": False, "label_recomputed": False})
        state = json.loads(row["state"])
        state["design"] = description
        rendered = json.dumps(state, ensure_ascii=False)
        if any(label in rendered for label in VERDICTS):
            raise ValueError("Verdict text leaked into a rendered state")
        derived_row.update(id=identifier, state=rendered)
        variants.append((derived_case, derived_row))
    return variants


README = """# Backend training representation augmentation

This controlled export keeps the 240 frozen diagnostic source cases and adds two
training-only representations for each of the original 120 training cases:

* Field-specific natural English, preserving explicit values and units.
* A JSON serialization of exactly the original facts, stored as the `design`
  string inside the existing `{"design": ..., "evidence": ...}` state wrapper.

There are 360 training rows and 480 total rows. **There are zero new independent
scenarios or labels.** The original 80 scenario families remain unchanged. This
is representation augmentation, not an enlarged independent benchmark.

Development (36), calibration (36), and final test (48) JSONL files are copied
byte-for-byte. Original training rows remain unchanged at the beginning of the
training file. Every added row inherits its original label, question, evidence,
facts, family, rationale, and label provenance. No rule or model relabels cases.
Only known fact fields are rendered; missing fields remain missing, and unknown
fields cause an error instead of an invented guarantee. Verdict names remain in
the unchanged question choices; answers and verdict labels are absent from states.

The second representation is not the runtime's exact top-level `facts` wrapper:
it preserves the validator's `state.design == case.description` contract. An
exact-runtime-format training experiment is separate and deferred. Repeated facts
can produce identical structured views within training; these are not counted as
new independent evidence. Existing template overlap limitations remain in force.

Reproduce into a new immutable directory:

    python scripts/augment_backend_training.py --output .runtime/backend-augmented-next

The manifest records exact source split and manifest hashes, a semantic source-manifest hash,
source-case hashes, and the generator hash. A semantic manifest hash intentionally
ignores JSON formatting/newline differences; held-out file checksums use raw bytes.
This dataset is synthetic and not independently reviewed. Do not interpret higher
agreement with these labels as established production accuracy or auto-promote a
checkpoint trained on it.
"""


def write_export(source, output):
    if output.exists():
        raise ValueError("Augmented exports are immutable; choose a new output directory")
    manifest, exported, indexed, _ = validate_backend_export(source)
    if (manifest["source_cases_sha256"] != PINNED_SOURCE_CASES or len(indexed) != 240
            or any(manifest["files"][split + ".jsonl"]["sha256"] != expected
                   for split, expected in PINNED_SOURCE_SPLITS.items())):
        raise ValueError("Use the original frozen 240-case backend-diagnostics-v1 source")
    source_contents = {split: (source / (split + ".jsonl")).read_bytes() for split in SPLITS}
    if any(digest(source_contents[split]) != expected for split, expected in PINNED_SOURCE_SPLITS.items()):
        raise ValueError("Source split changed after validation")
    cases = [json.loads(line) for line in (source / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if canonical_digest(cases) != PINNED_SOURCE_CASES:
        raise ValueError("Source cases changed after validation")
    derived_rows = []
    for row in exported["train"]:
        for case, derived in make_variants(indexed[row["id"]], row):
            cases.append(case)
            derived_rows.append(derived)
    audit = audit_cases(cases)
    audit.update(augmentation="representation_only", new_independent_cases=0,
                 original_source_case_count=240, added_training_representations=len(derived_rows))
    contents = dict(source_contents)
    if not contents["train"].endswith(b"\n"):
        raise ValueError("Frozen training file must end in a newline before appending representations")
    contents["train"] += "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in derived_rows).encode()
    all_training = exported["train"] + derived_rows
    unique_inputs = len({json.dumps([row["state"], row["questions"]], sort_keys=True) for row in all_training})
    result = copy.deepcopy(manifest)
    result.update(source_cases_sha256=canonical_digest(cases), quality_release_approved=False,
                  dataset_name="backend-augmented-v1", augmentation={
                      "kind": "representation_only", "source_dataset": "backend-diagnostics-v1",
                      "source_manifest_sha256": digest((source / "manifest.json").read_bytes()),
                      "source_manifest_semantic_sha256": canonical_digest(manifest),
                      "source_split_sha256": dict(PINNED_SOURCE_SPLITS), "source_cases_sha256": PINNED_SOURCE_CASES,
                      "generator_sha256": digest(Path(__file__).read_bytes()), "original_source_cases": 240,
                      "original_training_cases": 120, "added_training_representations": len(derived_rows),
                      "new_independent_cases": 0, "unique_training_inputs": unique_inputs,
                      "held_out_bytes_unchanged": True, "exact_runtime_wrapper_used": False,
                      "notice": "Synthetic training representation variants; no new independent scenarios or labels"})
    output.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        name = split + ".jsonl"
        (output / name).write_bytes(contents[split])
        result["files"][name] = {"sha256": digest(contents[split]),
                                 "count": len(all_training) if split == "train" else len(exported[split])}
    (output / "cases.jsonl").write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
                                        encoding="utf-8", newline="\n")
    for name, value in (("manifest.json", result), ("audit.json", audit)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "README.md").write_text(README, encoding="utf-8", newline="\n")
    validate_backend_export(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/backend-diagnostics-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_export(args.source, args.output)
    print(json.dumps({"output": str(args.output), "split_counts": {
        split: result["files"][split + ".jsonl"]["count"] for split in SPLITS},
        "augmentation": result["augmentation"]}, indent=2))


if __name__ == "__main__":
    main()
