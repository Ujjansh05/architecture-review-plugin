# Fact verification in 0.3.0a2

The update prevents a quoted passage from being mistaken for proof of the assistant's interpretation. It keeps Python and the existing optional Laya weights.

## What changes for a user

Suppose the design says, “During a database outage, we serve cached prices for up to sixty seconds.” An assistant might mistakenly extract `stale_on_error: false`. The old span check accepted the genuine quote even though it contradicted that interpretation.

The new review retains the proposed value and its quote, withholds it from both rules and Laya, and asks for confirmation. The resulting clarification concerns the intended design; it does not verify running software.

- Explicit user declarations remain usable as declarations.
- Machine interpretations of prose remain provisional.
- Matching typed JSON source declarations can establish what the source explicitly declares.
- Conflicting declarations remain unresolved, including conflicts in other supplied structured documents.
- Missing information, model disagreements and unknown production capacity stay visible.

The existing MCP tool names, input shapes and three verdicts remain available. New metadata uses the existing `coverage` and clarification fields. Hosts must retain confirmation questions and must not relabel their own guesses as user declarations.

## Structured source example

For automatic source matching, provide the complete declaration as JSON in a source document and cite its whole declaration:

```json
{
  "check_id": "cache_freshness",
  "scope_id": "product_prices",
  "facts": {
    "max_staleness_ms": 5000,
    "worst_case_staleness_ms": 3000,
    "bound_covers_failures": true,
    "stale_on_error": true,
    "stale_on_error_bound_ms": 60000
  }
}
```

The machine-supplied check, scope, field and typed value must match. An array of declarations or an object containing `checks` is also accepted. Duplicate JSON keys, mismatched scope, conflicting values, and boolean/number substitutions are rejected. Ordinary prose and a quoted value fragment are not a mechanically verified declaration. JSON source matching does not prove deployment behavior.

## What the benchmark does and does not show

Rules score 780/780 on the published finite synthetic cases, compared with 769/780 for the optional Laya deployment. These inputs closely match the rules and are now public regressions. They do not prove that Laya improves real user reviews or that either mode is generally accurate. [Results and reproduction](benchmarks/2026-10-preview.md).

The next evaluation compares the host alone, host plus rules, and host plus rules plus Laya on separately grouped prose scenarios. Extraction mistakes and final-answer errors must be scored separately. Unreviewed source articles and generated cases do not automatically become verified training labels.

## Installation and measurements

The Rules bundle omits Torch and Laya. The separate Laya CPU bundle has explicit model opt-in, asynchronous status, cancellation and retry. Use one variant per host registration. No new shared daemon is introduced; separate desktop sessions may still load separate workers.

Review stage and queue timing are separate from worker import/load/inference metrics. CUDA metrics come from the model worker's allocator; CPU paths report null and do not probe GPU drivers. RAM sampling covers the process tree during review. [Reproduce complete-review measurements](benchmarks/2026-10-runtime.md).

No model was trained or promoted for this runtime update. The second authenticated AI-host and desktop GUI tests remain pending.
