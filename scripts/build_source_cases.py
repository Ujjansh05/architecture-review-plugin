"""Source-linked, AI-authored pilot cases. These are NOT human-reviewed gold labels."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from arch_review.dataset import validate_cases

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "datasets/open-reference-v1"
DESTINATION = ROOT / "datasets/source-linked-v1"
# Every source document has one split; all variants of a scenario stay together.
# These are hypothetical adaptations, never claims of flaws in named products.
SCENARIOS = [
    ("aosa-hdfs", "train", "ack_durability", "journal-sync", "journal file is flushed", "A metadata service records a directory mutation and returns success to its client.",
     "The historical HDFS chapter distinguishes journal synchronization from acknowledgement.", "Storage synchronization can increase write latency; batching amortizes that work."),
    ("aosa-zeromq", "train", "queue_overload", "queue-bound", "size of the queue is not", "A message relay receives bursts faster than its network connection can drain them.",
     "Queued messages contribute latency; an unbounded queue can make latency unbounded.", "Batching can raise throughput while adding delay for individual messages."),
    ("primer-mint", "train", "retry_safety", "atomic-effect", "Service extracts transactions", "A financial-import worker increments an account's monthly spending total, then crashes before acknowledging delivery.",
     "The Mint exercise describes periodic extraction and categorization of transactions.", "The exercise's workload assumptions are illustrative and do not establish a production guarantee."),
    ("primer-twitter", "train", "retry_safety", "external-effect", "fanout", "A fan-out worker writes a timeline entry and invokes a push-notification provider; queue delivery can repeat.",
     "The Twitter exercise separates tweet ingestion, fan-out, and timeline retrieval.", "Fan-out work depends on follower distribution, not just the average recipient count."),
    ("primer-scaling_aws", "train", "ack_durability", "failure-domain", "replica", "A messaging API writes accepted messages to a leader before replying; the requirement includes loss of that host.",
     "The scaling exercise introduces additional database instances and replication as traffic grows.", "Replica count alone does not specify synchronous acknowledgement or failover safety."),
    ("aosa-graphite", "train", "queue_overload", "accepted-drop", "buffer", "A metric collector accepts points into a pending-write buffer while its disk writer is temporarily slow.",
     "The Graphite chapter discusses buffering writes to work around storage throughput limits.", "Buffered acceptance and durable persistence must be distinguished for a loss-sensitive workload."),
    ("primer-sales_rank", "development", "retry_safety", "dedup-window", "sales", "A replayable sales event increments a product's ranking counter; retries can arrive up to two days later.",
     "The sales-rank exercise uses sales data to compute product rankings.", "Aggregation and retry policy need explicit semantics when records can be replayed."),
    ("aosa-riak", "development", "ack_durability", "failure-domain", "replication", "A replicated message store acknowledges an accepted message; users require survival of a full host loss.",
     "The Riak account describes replication and fault tolerance in a distributed key-value store.", "A system family name does not establish a particular deployment's durability settings."),
    ("aosa-distsys", "development", "queue_overload", "queue-bound", "queues", "An asynchronous image-message pipeline accumulates work during a sustained traffic spike.",
     "The distributed web architecture chapter explains asynchronous queues and scaling trade-offs.", "A queue can absorb a temporary burst; it cannot resolve a permanent arrival-rate deficit."),
    ("primer-pastebin", "calibration", "ack_durability", "journal-sync", "write", "An API reports that a text message has been accepted and must remain available after a process crash.",
     "The pastebin exercise separates writes, content storage, and reads.", "Storage sizing assumptions are distinct from the acknowledgement contract."),
    ("primer-web_crawler", "calibration", "retry_safety", "stable-id", "crawl", "A crawler-result worker charges a per-page processing counter; a timed-out task may be redelivered.",
     "The crawler exercise considers distributing URL fetching and processing work.", "Revisiting a URL can be intentional; a retry identity must distinguish it from a new crawl task."),
    ("aosa-nginx", "calibration", "queue_overload", "queue-bound", "connections", "A gateway fronts a slow message-processing service while concurrent requests rise.",
     "The nginx account explains event-driven handling of concurrent connections.", "Efficient connection handling does not itself establish downstream queue capacity."),
    ("aosa-mediawiki", "test", "retry_safety", "external-effect", "job", "A wiki edit triggers a queued notification and an external mail-provider call that may be replayed.",
     "The MediaWiki account discusses decomposing web requests and deferred work.", "An internal database guard does not automatically cover an external notification provider."),
    ("aosa-nosql", "test", "ack_durability", "journal-sync", "durability", "A message service persists an accepted operation using a storage system with configurable durability.",
     "The NoSQL overview discusses differing storage and consistency design choices.", "Product categories alone do not specify persistence or acknowledgement guarantees."),
    ("primer-social_graph", "test", "queue_overload", "accepted-drop", "graph", "A social-graph mutation queue buffers accepted relationship updates while the graph store is unavailable.",
     "The social-graph exercise describes storage and access patterns for relationships.", "Graph connectivity and traffic skew change resource demand; acceptance must still have a clear contract."),
]


def build():
    source_map = {item["id"]: item for item in json.loads((CORPUS / "sources.json").read_text(encoding="utf-8"))}
    chunks = [json.loads(line) for line in (CORPUS / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    rows, lessons = [], []
    for index, (source_id, split, check, mutation, needle, situation, lesson, tradeoff) in enumerate(SCENARIOS):
        source = source_map[source_id]
        candidates = [item for item in chunks if item["source_id"] == source_id]
        matches = [item for item in candidates if needle.casefold() in " ".join(item["text"].split()).casefold()]
        if not matches:
            raise ValueError(f"Missing supporting passage: {source_id}: {needle}")
        passage = matches[0]
        record = {"id": "source-lesson-" + source_id, "source_id": source_id, "chunk_id": passage["id"],
                  "source_url": source["source_url"], "license": source["license"], "authors": source["authors"],
                  "problem": situation, "conditions": ["Use the historical text as background; evaluate only explicit facts of the new design."],
                  "lesson": lesson, "tradeoffs": [tradeoff], "passage_kind": "exact_licensed_chunk_reference",
                  "label_origin": "ai_authored_interpretation", "review_status": "pending_expert",
                  "reviewer": None, "reviewed_at": None, "source_is_gold_verdict": False}
        lessons.append(record)
        family = "source-" + source_id
        for variant, label in [("complete", "supported"), ("flawed", "contradicted"), ("missing", "insufficient_information")]:
            request = copy.deepcopy(json.loads((ROOT / "examples/complete.json").read_text()))
            request["requirements"]["features"] = [situation]
            request["design"]["components"] = ["Proposed API", "Queue", "Worker", "Persistent store"]
            request["design"]["flows"] = [situation]
            request["design"]["notes"] = "Hypothetical design inspired by a published reference; not a reconstruction of the named product."
            request["workload"] = {"average_messages_per_second": 75 + index * 25, "peak_factor": 4,
                                   "message_bytes": 512, "average_recipients": 2, "retention_days": 14,
                                   "replication_factor": 3, "storage_copies_per_message": 1, "storage_overhead_factor": 1.3}
            retry = request["design"]["retry"]
            ack = request["design"]["acknowledgement"]
            queue = request["design"]["queue"]
            for section in [retry, ack, queue]:
                section["notes"] = "Only explicit fields establish guarantees; no benchmark has been supplied."
            if mutation == "atomic-effect":
                field, section = "dedup_atomic_with_effect", retry
                good, bad, unknown = True, False, None
            elif mutation == "external-effect":
                retry["external_effects"] = True
                field, section = "external_effects_idempotent", retry
                good, bad, unknown = True, False, None
            elif mutation == "stable-id":
                field, section = "stable_message_id", retry
                good, bad, unknown = True, False, None
            elif mutation == "dedup-window":
                retry["max_redelivery_age_seconds"] = 172800
                field, section = "dedup_retention_seconds", retry
                good, bad, unknown = 259200, 86400, None
            elif mutation == "failure-domain":
                field, section = "survives_failures", ack
                good, bad, unknown = ["process_crash", "host_loss"], ["process_crash"], None
            elif mutation == "journal-sync":
                field, section = "after_durable_commit", ack
                good, bad, unknown = True, False, None
            elif mutation == "queue-bound":
                field, section = "policy", queue
                good, bad, unknown = "reject_before_accept", "unbounded", None
                # The configured limit is constant; the deliberately unbounded
                # variant does not enforce it. Review this policy/cap interaction
                # during independent human labeling before calling it gold data.
                queue["max_messages"] = 20000
            else:
                field, section = "preserves_accepted_messages", queue
                good, bad, unknown = True, False, None
            section[field] = {"complete": good, "flawed": bad, "missing": unknown}[variant]
            rows.append({"id": family + "-" + variant, "scenario_family": family, "source_group": source_id,
                         "split": split, "origin": "ai_authored_source_linked_hypothetical", "review_status": "pending_expert",
                         "reviewer": None, "reviewed_at": None, "request": request, "expected": {check: label},
                         "rationale": f"The {field} field is {section[field]!r}; apply the documented {check} definition to this explicit scenario.",
                         "changed_field": field, "variant": variant, "critical": label == "contradicted",
                         "source_ids": [source_id], "source_chunk_ids": [passage["id"]],
                         "background_evidence": [{"id": record["id"], "conditions": record["conditions"], "lesson": lesson}],
                         "license": source["license"], "attribution": {"authors": source["authors"], "url": source["source_url"],
                         "changes": "New hypothetical scenarios, assumptions, and labels authored by AI; not endorsed by source authors."}})
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, items in [("cases.jsonl", rows), ("lessons.jsonl", lessons)]:
        (DESTINATION / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items), encoding="utf-8", newline="\n")
    manifest = {"version": 1, **validate_cases(rows), "source_groups": {row["source_group"]: row["split"] for row in rows},
                "labels": dict(Counter(next(iter(row["expected"].values())) for row in rows)),
                "origin": "AI-authored draft labels, grounded in human-origin references; not human gold labels",
                "test_status": "frozen_not_evaluated", "source_corpus_manifest_sha256": hashlib.sha256((CORPUS / "manifest.json").read_bytes()).hexdigest(),
                "files": {name: hashlib.sha256((DESTINATION / name).read_bytes()).hexdigest() for name in ["cases.jsonl", "lessons.jsonl"]}}
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
