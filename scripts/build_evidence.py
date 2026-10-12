"""Materialize the manually curated, primary-source seed collection. No expert review is implied."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "arch_review" / "data"
CHECKED = "2026-10-13"
LESSON_CHECKS = {
    "rabbit-confirms": [["retry_safety"], ["ack_durability"], ["queue_overload"]],
    "aws-retries": [["queue_overload"], ["queue_overload"], ["retry_safety"]],
    "aws-outbox": [["ack_durability"], ["retry_safety"], ["ack_durability"]],
}

# Each lesson is (problem, applicability, decision/lesson, trade-off, source passage paraphrase).
# Small summaries are deliberately separate from synthetic training labels.
SOURCES = [
    ("rabbit-confirms", "RabbitMQ", "Consumer acknowledgements and publisher confirms", "https://www.rabbitmq.com/docs/confirms", None, "messaging", ["retry_safety", "ack_durability", "queue_overload"], [
        ("Redelivery after a worker crash", "Manual consumer acknowledgement", "Make repeated processing safe before acknowledging completion.", "Duplicate protection adds application state.", "Unacknowledged deliveries are requeued after connection loss."),
        ("Confusing acknowledgement boundaries", "Producer, broker and consumer are separate participants", "Name the precise operation each acknowledgement confirms.", "End-to-end success needs additional coordination.", "Publisher confirms and consumer acknowledgements cover different links."),
        ("Excessive in-flight deliveries", "Consumers receive work faster than they finish it", "Bound outstanding work with an appropriate prefetch setting.", "Small prefetch can reduce throughput.", "Prefetch limits unacknowledged deliveries.")]),
    ("rabbit-queues", "RabbitMQ", "Queues", "https://www.rabbitmq.com/docs/queues", None, "messaging", ["ack_durability"], [
        ("Queue metadata mistaken for message persistence", "Restart survival is required", "Specify durable queues and persistent messages separately.", "Transient workloads may not need persistence.", "Transient messages can be lost from durable queues on recovery."),
        ("Temporary queue used for durable work", "Queue lifetime is tied to a connection", "Check queue lifetime against the required recovery behaviour.", "Temporary queues simplify ephemeral sessions.", "Exclusive queues cannot outlive their declaring connection."),
        ("Restart survival assumed from a name", "Queue configuration is unknown", "Request the concrete queue durability configuration.", "More configuration must be supplied.", "Durable and transient queues recover differently.")]),
    ("rabbit-quorum", "RabbitMQ", "Quorum queues", "https://www.rabbitmq.com/docs/quorum-queues", None, "messaging", ["ack_durability"], [
        ("Replica count mistaken for unlimited fault tolerance", "A replicated queue requires a majority", "State the number and placement of failures that remain tolerable.", "More replicas increase resource use.", "A three-node quorum tolerates one node failure."),
        ("Durability without publisher confirmation", "Producer needs confirmation of accepted work", "Include publisher confirms in the durability flow.", "Acknowledgement latency includes replication.", "Confirmed-message safety depends on a surviving majority."),
        ("Assumed availability during every partition", "Quorum members are separated by a partition", "Explain which side can continue serving writes.", "Consistency may reduce availability.", "Quorum queues favour consistency under partition.")]),
    ("sqs-redelivery", "AWS", "Amazon SQS at-least-once delivery", "https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues-at-least-once-delivery.html", None, "messaging", ["retry_safety"], [
        ("Duplicate delivery", "SQS standard queue consumption", "Design the consumer to handle the same message again.", "Consumers need duplicate-safe effects.", "A stored copy can be received more than once."),
        ("Delete treated as proof every copy disappeared", "A storage server is unavailable during deletion", "Keep duplicate protection after successful handling.", "Deduplication requires a defined lifetime.", "An unavailable copy may remain after deletion."),
        ("Broker reliability mistaken for business uniqueness", "A worker produces non-idempotent effects", "Check application effect semantics separately from delivery reliability.", "Application guarantees require additional mechanisms.", "AWS recommends idempotent processing for standard queues.")]),
    ("stripe-idempotency", "Stripe", "Idempotent requests", "https://docs.stripe.com/api/idempotent_requests", None, "payments", ["retry_safety"], [
        ("Retried API operation repeats an effect", "Stripe API v1 POST with an idempotency key", "Reuse the same key for retries of the same operation.", "Key management becomes part of the client contract.", "API v1 replays the saved result for a reused key."),
        ("Retries outlive deduplication", "API v1 keys can be pruned after at least 24 hours", "Compare the retry horizon with key retention.", "Long retries need additional safeguards.", "Reusing a pruned key starts a new request."),
        ("Key reused for a different operation", "API v1 requests have changed parameters", "Keep request identity and parameters consistent across retries.", "New operations need distinct keys.", "The idempotency layer checks parameters for mismatch.")]),
    ("aws-retries", "AWS", "Timeouts, retries, and backoff with jitter", "https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/", None, "messaging", ["retry_safety", "queue_overload"], [
        ("Retry storm", "Dependency is overloaded", "Limit retries and add backoff.", "Recovery attempts take longer.", "Retries can increase load on an overloaded dependency."),
        ("Synchronized retry bursts", "Many clients retry together", "Use jitter to spread retry timing.", "Individual retry delays vary.", "Jitter disperses correlated retries."),
        ("Timeout mistaken for absence of an effect", "A side-effecting request times out", "Require duplicate-safe retry semantics.", "A timeout alone cannot settle the operation outcome.", "A timeout does not mean side effects did not occur.")]),
    ("aws-outbox", "AWS", "Transactional outbox pattern", "https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html", None, "commerce", ["retry_safety", "ack_durability"], [
        ("Database and event diverge", "A transaction also needs to publish an event", "Commit the business change and outbox record together.", "A relay and outbox lifecycle must be operated.", "A single transaction stores the business row and event."),
        ("Outbox relay repeats an event", "Relay publication can be retried", "Keep consumers idempotent even with an outbox.", "Outbox atomicity does not eliminate downstream duplicates.", "Consumers should track processed messages."),
        ("Uncommitted work is published", "Relay reads an outbox table", "Publish only committed event records.", "Delivery is asynchronous.", "The relay reads committed rows before publishing.")]),
    ("postgres-constraints", "PostgreSQL", "Constraints", "https://www.postgresql.org/docs/current/ddl-constraints.html", None, "messaging", ["retry_safety"], [
        ("Identifier without database enforcement", "Repeated message IDs can be inserted", "Use an enforced unique or primary-key constraint where appropriate.", "Conflicts require an explicit handling path.", "Unique constraints enforce uniqueness among rows."),
        ("Nullable deduplication key", "A unique column permits null", "Make message identity non-null or define null semantics explicitly.", "Malformed messages must be rejected or repaired.", "Default unique constraints treat null values as distinct."),
        ("Wrong uniqueness scope", "Message IDs are unique only within a tenant", "Match the constraint columns to the intended identity scope.", "Scope mistakes can suppress legitimate work.", "A multicolumn constraint enforces uniqueness of the combination.")]),
    ("google-overload", "Google", "SRE: Handling overload", "https://sre.google/sre-book/handling-overload/", "2016", "messaging", ["queue_overload"], [
        ("Request rate treated as capacity", "Requests consume different resources", "Measure resource cost for the actual workload.", "Capacity measurement needs representative traffic.", "Queries per second can be a poor capacity proxy."),
        ("One tenant consumes shared capacity", "Many clients share a backend", "Apply per-customer limits.", "Quotas require ownership and tuning.", "Per-customer limits isolate excessive resource use."),
        ("Rejecting traffic still overloads the server", "Rejection processing consumes significant resources", "Add client-side throttling.", "Client behaviour becomes part of protection.", "Clients can reduce attempted traffic when rejections rise.")]),
    ("discord-storage", "Discord", "How Discord stores trillions of messages", "https://discord.com/blog/how-discord-stores-trillions-of-messages", "2023-03-06", "messaging", ["queue_overload"], [
        ("Hot partition cascades latency", "Message traffic is concentrated in a channel and time bucket", "Evaluate skew as well as total message rate.", "Partition balancing cannot assume uniform activity.", "Hot partitions degraded latency across the cluster."),
        ("Database replacement leaves overload unchecked", "Concurrent requests target hot keys", "Limit upstream concurrency in a data-service layer.", "An additional service must be operated.", "Data services shielded storage from unbounded concurrency."),
        ("Migration ignores operational evidence", "Storage maintenance and latency are costly", "Use workload tests and operational experience to justify migration.", "Large migrations consume engineering effort.", "Discord tested and gained experience before migrating messages.")]),
    ("whatsapp-devices", "WhatsApp", "How WhatsApp enables multi-device capability", "https://engineering.fb.com/2021/07/14/security/whatsapp-multi-device/", "2021-07-14", "messaging", [], [
        ("Phone availability constrains companion devices", "A companion depends on the phone", "Consider independent device connections when that dependency is unacceptable.", "Synchronization and key management become more complex.", "The updated design removed the phone as a required source of truth."),
        ("Device fan-out omitted from workload", "Messages are encrypted separately for multiple devices", "Count delivery destinations at device level.", "Encryption and network work increase with devices.", "The client encrypts and sends to recipient device lists."),
        ("History transfer confused with ongoing state sync", "A new device joins an existing account", "Specify initial history transfer and subsequent synchronization separately.", "Different data classes need different lifecycles.", "New devices receive encrypted history and later application-state updates.")]),
    ("slack-messaging", "Slack", "Real-time messaging", "https://slack.engineering/real-time-messaging/", "2023-04-11", "messaging", [], [
        ("Persistent connections and channel routing conflated", "Many online clients subscribe to channels", "Describe connection gateways separately from channel routing.", "Routing and membership must remain coordinated.", "Gateway servers connect clients to channel servers."),
        ("Fan-out workload hidden by inbound rate", "A channel message reaches multiple online clients", "Measure deliveries as well as incoming messages.", "Large channels can dominate outbound traffic.", "Channel servers broadcast through subscribed gateways."),
        ("All events assigned identical durability", "Typing and durable chat messages coexist", "Give transient and persistent events distinct requirements.", "Users observe different recovery behaviour.", "Typing events follow a transient flow without database persistence.")]),
    ("amazon-dynamo", "Amazon", "Dynamo: Amazon's highly available key-value store", "https://www.amazon.science/publications/dynamo-amazons-highly-available-key-value-store", "2007", "commerce", [], [
        ("Availability requires accepting divergent versions", "A workload favours accepting writes during failures", "Document reconciliation and consistency trade-offs.", "Applications may handle conflicts.", "Dynamo combines versioning with application-assisted conflict resolution."),
        ("Failure treated as exceptional", "A distributed storage fleet has routine faults", "Include repair and failure handling in the normal design.", "Recovery mechanisms add operational complexity.", "The design treats failures as a normal operating condition."),
        ("Historical system conflated with a current product", "Using the 2007 Dynamo paper", "Keep the evidence scoped to the system and date described.", "Current products need their own references.", "The paper describes Amazon's historical Dynamo implementation.")]),
    ("google-spanner", "Google", "Spanner: Google's globally-distributed database", "https://research.google/pubs/spanner-googles-globally-distributed-database-2/", "2012", "commerce", [], [
        ("Global consistency assumed free", "Transactions span replicated geographic storage", "Account for coordination when selecting consistent transactions.", "Coordination affects latency and failure behaviour.", "Spanner combines synchronous replication with distributed transactions."),
        ("Clock timestamps treated as exact", "Ordering depends on bounded time uncertainty", "State the clock assumptions behind timestamp ordering.", "Uncertainty can require waiting.", "TrueTime exposes uncertainty instead of assuming perfect clocks."),
        ("Paper-level guarantees copied without mechanisms", "Adopting a globally consistent database idea", "Identify which replication and transaction mechanisms provide the requirement.", "Similar terminology does not establish equivalent behaviour.", "The paper ties external consistency to its implementation.")]),
    ("meta-tao", "Meta", "TAO: The power of the graph", "https://engineering.fb.com/2013/06/25/core-infra/tao-the-power-of-the-graph/", "2013-06-25", "social_feeds", [], [
        ("Cache behaviour mismatches graph reads", "Read-heavy graph access has unusual locality", "Choose caching around actual query patterns.", "Specialized caching adds system complexity.", "The social graph workload differed from block-cache assumptions."),
        ("Every product team manages cache consistency", "Applications directly coordinate database and cache", "Centralize repeated data-access responsibilities when justified.", "The abstraction becomes shared infrastructure.", "Separate memcache and MySQL logic caused application complexity."),
        ("Generic data interface poorly fits relationships", "Products repeatedly access objects and associations", "Consider an interface matching common relationship queries.", "Specialization narrows supported access patterns.", "TAO introduced objects and associations for graph access.")]),
    ("netflix-openconnect", "Netflix", "Open Connect", "https://openconnect.netflix.com/en/", None, "media_delivery", [], [
        ("Video delivery repeatedly crosses transit networks", "Large regional content demand", "Consider placing content nearer viewers.", "Placement introduces distribution and operations work.", "Embedded appliances localize Netflix traffic within ISP networks."),
        ("One deployment layout assumed everywhere", "ISP network conditions differ", "Choose placement and interconnection around local network conditions.", "Deployment becomes environment-specific.", "Appliances can be distributed or clustered geographically."),
        ("Delivery architecture confused with the full application", "An Open Connect case study is used as evidence", "Limit conclusions to content delivery.", "Other application subsystems need separate evidence.", "Open Connect describes delivery partnerships and appliances.")]),
    ("shopify-modules", "Shopify", "Deconstructing the monolith", "https://shopify.engineering/deconstructing-monolith-designing-software-maximizes-developer-productivity", "2019-02-21", "commerce", [], [
        ("High coupling reduces developer productivity", "A large monolith contains multiple business domains", "Consider explicit module boundaries.", "Boundaries need enforcement and maintenance.", "Shopify chose to evolve toward a modular monolith."),
        ("Services chosen solely because they are popular", "A product is choosing its first architecture", "Compare operational complexity with the current team's needs.", "A single deployment retains some coupling.", "The article describes advantages and costs of architecture choices."),
        ("Architecture assumed permanent", "Codebase and team size grow", "Reassess architecture as constraints change.", "Evolution requires migration effort.", "The original monolith worked before its downsides grew.")]),
    ("salesforce-tenants", "Salesforce", "Platform multitenant architecture", "https://architect.salesforce.com/docs/architect/fundamentals/guide/platform-multitenant-architecture.html", None, "multi_tenant", [], [
        ("Shared data lacks explicit tenant separation", "Tenants share infrastructure", "Specify tenant identity at data-access boundaries.", "Isolation depends on consistent enforcement.", "OrgID separates organization-specific records in shared tables."),
        ("One tenant monopolizes runtime resources", "Tenant code executes on a shared platform", "Enforce resource budgets per tenant execution.", "Limits constrain application implementations.", "Runtime governors limit CPU, memory and database operations."),
        ("Customization requires dedicated deployments", "Many tenants customize one application platform", "Evaluate metadata-driven configuration.", "The runtime must interpret and cache metadata.", "A shared kernel uses tenant-specific metadata.")]),
    ("uber-tiered", "Uber", "Introduction to Kafka tiered storage at Uber", "https://www.uber.com/us/en/blog/kafka-tiered-storage/", "2024-07-01", "analytics", [], [
        ("Long retention forces more broker compute", "Storage and processing scale together", "Consider separating older log storage from broker resources.", "Remote storage adds a dependency.", "Tiered storage decouples retention from broker-local capacity."),
        ("Historical reads interfere with fresh traffic", "Tail reads and backfills have different latency needs", "Define separate paths for recent and historical log reads.", "Remote reads have different performance characteristics.", "Recent reads use local storage; older reads use the remote tier."),
        ("Local retention confused with total retention", "Logs span local and remote tiers", "Track each tier's retention and recovery behaviour.", "Capacity calculations need both tiers.", "Local and remote tiers have independent retention settings.")]),
    ("elastic-refresh", "Elastic", "Near real-time search", "https://www.elastic.co/docs/manage-data/data-store/near-real-time-search", None, "search", [], [
        ("Write success mistaken for immediate search visibility", "Documents are indexed into Elasticsearch", "State the search-freshness requirement separately.", "Refresh frequency affects resource use.", "A refresh makes recent operations visible to search."),
        ("Search visibility mistaken for a full disk commit", "Index refresh and persistence are conflated", "Distinguish searchable state from durability mechanisms.", "Recovery guarantees require additional configuration evidence.", "Lucene can open segments without a full commit."),
        ("Refresh timing assumed without configuration", "Freshness has an application deadline", "Request the relevant refresh policy and measure visibility delay.", "Explicit refreshes have operational cost.", "Refresh can be periodic or explicitly requested.")]),
]


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    records, catalog, cases = [], [], []
    for key, company, title, url, published, family, checks, lessons in SOURCES:
        catalog.append({"id": key, "company": company, "title": title, "url": url,
                        "publication_date": published, "accessed_at": CHECKED,
                        "date_precision": "unknown" if published is None else "year" if len(published) == 4 else "day",
                        "source_type": "primary", "source_checked_by": "AI-assisted source inspection",
                        "review_status": "pending_expert"})
        for index, (problem, condition, lesson, tradeoff, passage) in enumerate(lessons, 1):
            lesson_checks = LESSON_CHECKS.get(key, [checks] * len(lessons))[index - 1]
            records.append({"id": f"{key}-{index}", "source_id": key, "problem": problem,
                "conditions": [condition], "lesson": lesson, "tradeoffs": [tradeoff],
                "supporting_passage": {"kind": "paraphrase", "text": passage},
                "source_url": url, "publication_date": published, "accessed_at": CHECKED,
                "tags": [family, *lesson_checks, *key.split("-")], "checks": lesson_checks,
                "review_status": "pending_expert", "reviewer": None,
                "lesson_kind": "conditional_engineering_inference",
                "limitation": "Public subsystem snapshot; not a complete or current private company architecture."})
        if key in {"discord-storage", "whatsapp-devices", "slack-messaging", "amazon-dynamo", "google-spanner",
                   "meta-tao", "netflix-openconnect", "shopify-modules", "salesforce-tenants", "uber-tiered",
                   "stripe-idempotency", "elastic-refresh"}:
            cases.append({"id": key, "company": company, "subsystem": title, "system_family": family,
                          "source_url": url, "publication_date": published, "workload_scale": None,
                          "evidence_ids": [f"{key}-{i}" for i in range(1, 4)],
                          "documented_facts": [entry[4] for entry in lessons],
                          "reviewer_inferences": [entry[2] for entry in lessons],
                          "review_status": "pending_expert", "diagram": None,
                          "unknowns": ["Complete company architecture", "Current deployment configuration", "Comparable benchmark"],
                          "small_product_alternative": {"kind": "synthetic_design_prompt",
                              "text": "Test whether one application, one transactional database, and a bounded worker queue meet the declared requirements before adding distributed components."}})
    for name, rows in [("evidence.jsonl", records), ("case_studies.jsonl", cases)]:
        (DATA / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (DATA / "sources.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    families = ["messaging", "social_feeds", "media_delivery", "commerce", "payments", "search", "analytics", "multi_tenant"]
    coverage = {family: {"case_studies": [c["id"] for c in cases if c["system_family"] == family],
                         "lessons": sum(family in r["tags"] for r in records),
                         "expert_reviewed": False,
                         "automated_review_scope": "three messaging checks" if family == "messaging" else "reference_only"}
                for family in families}
    (DATA / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} draft lessons, {len(catalog)} sources, and {len(cases)} case studies.")


if __name__ == "__main__":
    main()
