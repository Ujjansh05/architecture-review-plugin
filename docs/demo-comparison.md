# Codex with and without ArchRivet

This is one recorded example, not a general AI accuracy benchmark. The same original cache-design prompt was used; the plugin run additionally requests tool use and retention of the tool summary. The Codex CLI did not report its underlying model ID.

**Codex already found the missing detail without ArchRivet.** The useful addition is a structured review, a named fact to supply, and a before/after comparison that preserves remaining unknowns. The demonstration uses rules mode; Laya accuracy is measured [separately](benchmarks/2026-10-preview.md).

The GIF is a designed, shortened walkthrough based on these actual outputs and the packaged MCP protocol test. It is not a live screen recording or a latency measurement.

## Without the plugin

The design does **not establish that it meets the freshness requirement**. Normal responses are at most three seconds old, within the five-second limit. But the missing detail is the **maximum age of data served during a database outage**.

For example, if a price is already three seconds old when an outage starts and is served three seconds later, it is six seconds old and violates the requirement. The unspecified outage bound leaves this possible; it does not prove it happens.

The other failure paths meet the stated bounds.

**100 reads per second does not prove capacity.** It describes demand, not demonstrated ability to handle it while meeting the freshness requirement.

## With the plugin

The design does not establish the five-second guarantee. The normal path’s three-second bound meets it, but the missing detail is the maximum age permitted during a database outage.

For example, if a price is already three seconds old when the database fails, serving that same value three seconds later would return a six-second-old price. Specify and enforce an outage age limit of five seconds or less, then test an outage while measuring returned prices’ ages.

100 reads per second describes demand; it does not prove capacity. That needs a representative benchmark measuring throughput, latency, and errors.

ArchRivet’s completed, partial review found insufficient information. It ran in rules-only mode; no model benchmark ran.

ArchRivet review summary
- Only the listed requirements were reviewed; the whole architecture is not certified.
- Findings depend on supplied facts. Source interpretation and deployment behavior are not independently verified.
- Suggested changes are proposals, not evidence of an implemented fix.
- Scope/check ["product-price-cache", "cache_freshness"]: insufficient_information. "The declared facts do not establish the required guarantee."
- Missing or unresolved for ["product-price-cache", "cache_freshness"]: "stale_on_error_bound_ms"
- Background evidence for ["product-price-cache", "cache_freshness"]: "backend-cache-1"
- Application capacity is unknown; workload arithmetic is not a capacity benchmark.
- Missing capacity evidence: "A benchmark with workload, configuration, latency, errors and duration."
- Missing workload input: "average_messages_per_second"
- Missing workload input: "average_recipients"
- Missing workload input: "growth_horizon_months"
- Missing workload input: "message_bytes"
- Missing workload input: "monthly_growth_fraction"
- Missing workload input: "peak_factor"
- Missing workload input: "replication_factor"
- Missing workload input: "retention_days"
- Missing workload input: "storage_copies_per_message"
- Missing workload input: "storage_overhead_factor"
- Limitation: "Experimental review of supplied facts; no production certification or outage probability."
- Limitation: "Source lessons are background evidence; extracted facts and labels await independent review."
- Limitation: "Complete means the requested checks finished, not that the entire architecture is safe."

## After one clarification

Originally, compliance is unproven: the normal three-second age meets the five-second limit, but the maximum age served during a database outage is missing. ArchRivet returned `insufficient_information`, with a partial review.

With only the 60-second outage clarification added, the verdict becomes `contradicted`. For example, serving a 30-second-old price during an outage is permitted by that clarification but violates the five-second requirement. ArchRivet’s comparison confirmed that only the outage-age fact changed.

100 reads per second describes demand; it does not prove capacity. That needs measurements under representative load and failures. Verify freshness by failing the database and measuring the oldest returned price.

Both reviews used experimental rules-only mode; no model benchmark ran.

First review’s verbatim summary:

```text
ArchRivet review summary
- Only the listed requirements were reviewed; the whole architecture is not certified.
- Findings depend on supplied facts. Source interpretation and deployment behavior are not independently verified.
- Suggested changes are proposals, not evidence of an implemented fix.
- Scope/check ["product-price-cache", "cache_freshness"]: insufficient_information. "The declared facts do not establish the required guarantee."
- Missing or unresolved for ["product-price-cache", "cache_freshness"]: "stale_on_error_bound_ms"
- Background evidence for ["product-price-cache", "cache_freshness"]: "backend-cache-1"
- Application capacity is unknown; workload arithmetic is not a capacity benchmark.
- Missing capacity evidence: "A benchmark with workload, configuration, latency, errors and duration."
- Missing workload input: "average_messages_per_second"
- Missing workload input: "average_recipients"
- Missing workload input: "growth_horizon_months"
- Missing workload input: "message_bytes"
- Missing workload input: "monthly_growth_fraction"
- Missing workload input: "peak_factor"
- Missing workload input: "replication_factor"
- Missing workload input: "retention_days"
- Missing workload input: "storage_copies_per_message"
- Missing workload input: "storage_overhead_factor"
- Limitation: "Experimental review of supplied facts; no production certification or outage probability."
- Limitation: "Source lessons are background evidence; extracted facts and labels await independent review."
- Limitation: "Complete means the requested checks finished, not that the entire architecture is safe."
```

Second review’s verbatim summary:

```text
ArchRivet review summary
- Only the listed requirements were reviewed; the whole architecture is not certified.
- Findings depend on supplied facts. Source interpretation and deployment behavior are not independently verified.
- Suggested changes are proposals, not evidence of an implemented fix.
- Scope/check ["product-price-cache", "cache_freshness"]: contradicted. "The declared cache-age bound exceeds the permitted staleness."
- Background evidence for ["product-price-cache", "cache_freshness"]: "backend-cache-1"
- Application capacity is unknown; workload arithmetic is not a capacity benchmark.
- Missing capacity evidence: "A benchmark with workload, configuration, latency, errors and duration."
- Missing workload input: "average_messages_per_second"
- Missing workload input: "average_recipients"
- Missing workload input: "growth_horizon_months"
- Missing workload input: "message_bytes"
- Missing workload input: "monthly_growth_fraction"
- Missing workload input: "peak_factor"
- Missing workload input: "replication_factor"
- Missing workload input: "retention_days"
- Missing workload input: "storage_copies_per_message"
- Missing workload input: "storage_overhead_factor"
- Limitation: "Experimental review of supplied facts; no production certification or outage probability."
- Limitation: "Source lessons are background evidence; extracted facts and labels await independent review."
- Limitation: "Complete means the requested checks finished, not that the entire architecture is safe."
```

The last run called `compare_reviews` on two completed jobs in the same session. Only `stale_on_error_bound_ms` changed from unknown to 60,000. Both plugin answers retained the corresponding completed review summary.

[Recorded prompts, answers, tool names and completed reviews](assets/2026-10-codex-recording.json) · [Setup instructions](quick-start-preview.md)
