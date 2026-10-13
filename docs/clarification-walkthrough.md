# Review an incomplete design

ArchRivet checks the design facts supplied to it. A missing guarantee stays unresolved until the user supplies the relevant facts. It does not inspect application code or discover undocumented retry paths.

## Run the example

After [installing ArchRivet](../README.md#try-it-locally), run this command in your activated Python environment:

```bash
python -m arch_review review examples/needs-clarification.json --rules-only
```

The [input](../examples/needs-clarification.json) says messages can be redelivered, writes are not intrinsically idempotent, and stable message IDs exist. It leaves two facts unknown: whether duplicate protection is enforced and whether that protection is atomic with the business write.

The retry finding includes:

```json
{
  "check_id": "retry_safety",
  "verdict": "insufficient_information",
  "reason": "An identifier alone does not establish protection for every business effect.",
  "missing_information": [
    "design.retry.dedup_enforced",
    "design.retry.dedup_atomic_with_effect"
  ]
}
```

The acknowledgement and queue checks are supported under their supplied facts. An unresolved retry check does not erase those separate findings. The review stays `rules_only` / `partial`, and production capacity remains unknown without benchmark evidence.

## Ask specific follow-up questions

The host assistant can turn the returned fields into these questions:

1. What enforces duplicate protection when two workers process the same message ID concurrently?
2. Is recording that protection atomic with applying the business write, including a crash between the two?

Record only what the user establishes. If they do not know, keep the corresponding value `null`. Adding a message ID, naming a database, or mentioning transactions does not establish these two guarantees by itself.

With all other facts in this example unchanged, the retry rule behaves as follows:

| Supplied duplicate-protection facts | Retry verdict | What remains unresolved |
| --- | --- | --- |
| Enforcement and atomicity unknown | `insufficient_information` | Both facts |
| Atomicity explicitly false; enforcement unknown | `contradicted` | A definite violation is already established |
| Atomicity true; enforcement unknown | `insufficient_information` | Enforcement |
| Enforcement and atomicity both true | `supported` | Implementation still requires verification |

The last row matches the retry facts in [complete.json](../examples/complete.json). If retry/redelivery behavior itself is unknown, the rule first asks for `design.retry.redelivery_possible`.

A clarification describes an existing guarantee. A design revision changes a guarantee. Keep that distinction visible; do not fill fields with desired answers just to obtain a supported verdict.

## Suggested assistant instruction

> Review my design with ArchRivet using only supplied facts. Show definite violations and unresolved checks separately. Translate missing facts into short follow-up questions, and keep unanswered facts unknown. After I answer, update only the facts my answers establish, run the review again, and explain what changed. Retain remaining gaps and verification steps in the final explanation.

This is a host-guided workflow. The plugin returns findings and missing facts; it does not independently interview the user. A supported verdict checks declarations, so the suggested implementation and failure tests still matter.

## Show it in a demo

Start with the incomplete example and display its two missing facts. Show an explicit answer and rerun the tool. Include both an unsafe answer that produces a contradiction and a fully specified answer that supports the check. Keep the unknown-capacity result visible. This demonstrates how the decision responds to the supplied facts.
