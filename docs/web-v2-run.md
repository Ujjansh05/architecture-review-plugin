# Expanded web workflow execution — 2026-10-13

The collection, integrity checks, CPU fine-tuning and two-client MCP review completed locally. The corrected export and measurements are published; the trained head remains an experimental local artifact. Three agents independently checked dataset integrity, pipeline tests and MCP/host behavior while model-heavy stages ran sequentially.

## Data and training

The scraper reused its permitted-source cache: 57 references, 598 chunks and 60 attempted pages. Full candidate selection considered 36 relevant documents. The first generated export had cross-split copies hidden by changed evidence IDs and quotes containing only resource links. It was preserved for diagnosis and excluded from training.

The corrected export has 16 families and 48 cases, with source/check/split assignments preserved. Three source passages were repaired before any model test predictions; one weakly supported family was quarantined. New validation rejects link-only quotes and citation-normalized duplicate inputs across splits. The independent agent audit found no remaining exact input or normalized quote duplicates across splits. This does not exclude semantic overlap or establish human-verified labels.

| Measurement | Result |
|---|---:|
| Train / development / calibration / test cases | 24 / 9 / 6 / 9 |
| Model / trainable parameters | Approximately 322M / 15M |
| Device / epochs | CPU / 1 |
| Development accuracy, base → trained | 4/9 → 4/9 |
| False supported predictions, base → trained | 4 → 4 |
| Missing-information recall, base → trained | 1/3 → 1/3 |
| Answer-order changes after training | 1/9 |
| Training command wall time, including startup | 34.594 seconds |
| Training loop and evaluation time | 20.985 seconds |
| Peak sampled summed process RSS | 2,660,814,848 bytes, approximately 2.48 GiB |
| Saved head size | 59,880,768 bytes |
| Paid GPU spend | $0 |

Memory was sampled every 0.5 seconds across the command's process tree; shared pages can be counted more than once. The encoder and original base checkpoint were verified unchanged, and the saved head passed a load round trip. The final test was not scored. Development lacks durability cases; calibration and test lack retry cases. Six-case calibration is not validated. These results establish that the experiment runs, not that training produced a better reviewer.

The [exact dataset](../datasets/web-solo-v2/README.md), [integrity audit](../datasets/web-solo-v2/audit.json), [repair provenance](../datasets/web-solo-v2/repair-provenance.json) and [machine-readable results](web-v2-results.json) retain the details. Source text and adaptations retain per-item CC BY attribution terms. The earlier snapshots remain unchanged.

## MCP integration

Two cold MCP trials exposed a Windows lifecycle bug: the SDK's stdio client creates a Windows job with `KILL_ON_JOB_CLOSE`. A daemon spawned by that client is stopped when the client closes, interrupting a second client. Historical memory errors in an appended log initially confused diagnosis; they were not evidence of a new out-of-memory failure.

Starting the service independently from the CLI resolved the two-client trial. Both clients used one service and one real base-Laya worker. The complete example returned three supported verdicts, and the flawed example returned three contradicted verdicts. These are the combined plugin verdicts, including deterministic rule handling, not proof of model-only accuracy. The first successful run took 10.454 seconds for the initial CLI review and 4.281 seconds for the two-client protocol test; peak sampled service-tree RSS was 1,798,709,248 bytes. This sample excludes the full AI-host memory cost.

The implementation now provides `arch-review start` and requires independent startup for Windows MCP when no service is running. It does not change job restrictions or global assistant settings. The model loads on first review, and all sessions reuse it. See [the README](../README.md#local-service) for startup commands.

The final sequential trial also passed: two MCP clients returned all six expected fixture verdicts through one worker in 4.422 seconds. The initial CLI review took 10.703 seconds. Sampled peak service-tree RSS was 2,128,416,768 bytes (approximately 1.98 GiB); this excludes the host's full process tree.

## Actual assistant trial and checks

One actual AI host, Codex, called the MCP tool for complete, flawed and incomplete designs. In the final trial, all three requests matched the fixtures and all nine combined plugin verdicts survived into the final answer. The answer also preserved unknown capacity, missing evidence, and the distinction between model confidence and production failure probability. Host elapsed time was 118.062 seconds. Base Laya alone selected supported for all nine decisions in these fixtures; the successful combined results depend on the plugin's explicit rules. A second AI product and independent usefulness comparison were not tested in this run.

Earlier attempts are retained in the [sanitized integration report](web-v2-mcp-results.json): one provider-capacity failure occurred before tool use; another trial returned nine correct plugin verdicts but only six were preserved by the host after batched output was truncated. The final trial required sequential tool calls and compact inspection of each complete response. Host instructions now warn against batching large reviews into a truncated combined result. The final success does not erase those observed limitations.

**66 local tests passed** after the fixes, including source/export integrity for both published web snapshots and a real service-survives-client-exit regression. Ruff and `pip check` passed. These automated and AI-agent checks are not independent human validation. No trained checkpoint was deployed, no global assistant configuration was changed, and no paid GPU job was started.
