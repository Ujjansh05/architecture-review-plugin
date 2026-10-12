# Dataset and company architecture coverage

The [open reference collection](../datasets/README.md) now adds 18 licensed documents, 372 attributed chunks, 15 draft lessons, 45 source-linked cases, and a blank human-review packet. The [screening audit](../datasets/source-screening.json) records why downloaded Kaggle/Hugging Face samples were kept out of gold evaluation data. See the [experiment report](research-validation-2026-10-13.md) for actual baseline and training results.

The dataset describes dated, publicly documented subsystems and their trade-offs. It does not claim to reproduce a company's entire private architecture or a universally perfect design.

| Family | Included references | Example learning goal |
|---|---|---|
| Messaging | WhatsApp multi-device, Slack real-time messaging, Discord storage, RabbitMQ, SQS | Device delivery, persistent messages, retries, skew and overload |
| Social feeds/graphs | Meta TAO | Graph access and caching trade-offs |
| Media delivery | Netflix Open Connect | Delivery placement and origin load |
| Commerce/storage | Amazon Dynamo, Google Spanner, Shopify modular monolith, transactional outbox | Consistency, availability, transaction boundaries and operational complexity |
| Payments | Stripe idempotent requests | Request identity and retry lifetime |
| Search | Elasticsearch refresh | Visibility delay and indexing trade-offs |
| Analytics/event pipelines | Uber Kafka tiered storage | Retention and storage separation |
| Multi-tenancy | Salesforce platform | Isolation and shared infrastructure |

Exact URLs and available publication dates live in `src/arch_review/data/sources.json`. Living documentation with no verified publication date uses null plus an access date. `case_studies.jsonl` separates documented facts, engineering inferences and unknowns; absent diagrams and comparable benchmarks remain null. The source-checked summaries still require independent expert review.

The 60 lesson records contain a problem, applicability conditions, lesson, trade-offs, a clearly marked passage paraphrase, source provenance and decision tags. Retrieval only considers lessons eligible for the specific check. Brand names do not establish a guarantee. Final-test labels and case rationales are never added to the retrieval index.

The 100 pilot examples are deliberately smaller, synthetic decision exercises, not company architecture reconstructions. Their 20 families cover repeated business effects, external side effects, deduplication expiry, concurrent workers, acknowledgement boundaries, host/zone failure, accepted-message loss, tenant bursts and retry amplification. Good, flawed and incomplete variants expose single-detail changes in the verdict. The ten starter regressions provide quick checks; they are not extra independent evaluation examples.

To add a family, collect primary sources and state the workload, constraints, evolution, alternatives and applicable failure model. Extract conditional lessons first. Define a new bounded decision only after reviewers agree on supported/contradicted/insufficient-information examples. Group every variant or reconstruction of a subsystem before assigning a split. Freeze an independent final set before creating training variants; track source and family overlap explicitly.

Review the existing three messaging decisions before expanding automated coverage. References for payments, search or media delivery do not mean the current plugin can validate an entire product in those domains.
