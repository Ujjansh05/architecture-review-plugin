# Review a design with your assistant

The 0.3 alpha lets your existing AI ask ArchRivet what is missing, collect your answers, and show what changed. The default rules workflow needs no local model or GPU.

After installation, stop an existing service before configuring it:

```powershell
arch-review stop
arch-review configure --backend rules
arch-review doctor
arch-review start
arch-review mcp-config
```

Copy the generated MCP configuration into your assistant's host settings. On Windows, start the service in a separate terminal before connecting. See the [full host setup guide](backend-v2.md#shared-jobs-cli-and-mcp).

Ask:

> Review my design with ArchRivet. Ask up to three questions at a time. Keep anything I cannot answer unknown. After I answer, review it again and show what changed.

Hosts supporting MCP prompts can select `review_my_design`, provide `design`, and choose `audience=developer` or `audience=beginner`. Otherwise use the request above; the same instructions are also in the server's tool guidance.

For example, a worker saves a message and crashes before acknowledging it. The queue redelivers it. A stable message ID alone does not establish whether a second write is prevented. ArchRivet asks whether duplicate protection is enforced and whether it commits with the business write. An unanswered question keeps the verdict `insufficient_information`. An explicit separate commit can produce `contradicted`; a fully specified protective flow can produce `supported` for this operation.

## Tool sequence

1. `get_review_schema` returns eight decision definitions, exact fact paths and types.
2. The assistant maps explicit facts to distinct operation scopes. It calls `review_design` with `input_mode="supplied_facts"`, then polls `get_review`.
3. The assistant asks the returned `next_questions`, at most three per round. It uses the schema to translate an answer into facts. “I don't know” leaves the fact absent.
4. After answers, it submits the complete updated request. `compare_reviews(before_job_id, after_job_id)` compares the completed, unexpired jobs.

The [example input](../examples/guided-review.json) is also runnable without an AI host:

```powershell
arch-review review-design examples/guided-review.json --local --output review.json
```

Each known decision fact needs an origin: `user_declared` or `host_extracted`. Host interpretations require exact `fact_quotes` or valid character `fact_spans` in a supplied source document. **Omit origins and quotes for missing/null facts.** A quote must occur exactly once; when text repeats, use character spans instead. Quotes validate the reference, not the truth of the interpretation.

`context` can describe the operation for optional model inference; it is not parsed by the rules. In supplied-fact mode, documents are reference material and are marked `document_content_reviewed=false`. Facts from omitted operations are not automatically discovered. The host must surface these scope limits.

## What the result means

| Verdict | Meaning |
|---|---|
| supported | Explicit facts satisfy the particular checked requirement. |
| contradicted | Explicit facts establish a violation. |
| insufficient_information | Necessary facts are missing, ambiguous or conflicting. |

A proposed fix is not an implemented fix. Removing a check from a revision does not resolve it. Comparison retains remaining questions and unknown capacity. Workload arithmetic estimates demand; capacity needs applicable benchmarks. The plugin does not certify an entire design or predict an outage probability.

The old three-check tool and document-extraction mode remain available. Natural-language conversations still depend on the host extracting facts correctly. One real host has been exercised; the second-host test is pending.
