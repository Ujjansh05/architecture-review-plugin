# ArchRivet demo

The [GIF](archrivet-demo.gif) is an animated summary of **actual rules-only CLI results**. It is a designed walkthrough, not a screen recording, a model-inference demonstration, or a performance benchmark. Wording and timing are adapted for readability. The [static preview](archrivet-demo.png) shows the first scene without motion.

## Text walkthrough

1. **Review the draft.** `arch-review review examples/flawed.json --rules-only` returns three `contradicted` verdicts: duplicate protection is not enforced atomically, success precedes durable commit, and queue growth is explicitly unbounded.
2. **Revise the design.** The supplied `examples/complete.json` adds enforced atomic deduplication covering the retry window, acknowledgement after durable commit with stated failure-survival guarantees, and a bounded queue with rejection before acceptance and retry backoff.
3. **Review the revision.** `arch-review review examples/complete.json --rules-only` returns three `supported` rule verdicts under those assumptions. The result remains `partial` because model inference was explicitly skipped. These declarations still require implementation and failure tests.
4. **Calculate the workload.** The example yields 500 peak messages/second, 1,500 peak deliveries/second and approximately 0.956 TB retained storage. Inputs are 100 average messages/second, a 5× peak, three recipients, 1,024 bytes per message, 30 days' retention, one stored copy per message, three replicas and 20% overhead.
5. **Keep unknowns visible.** Capacity and latency remain unknown without a matching benchmark. A supported rule verdict does not certify a production deployment.

The [capture metadata](archrivet-demo.json) records input hashes, commands, exact reasons, workload formulas, capacity gaps and generated asset hashes. No account details, local paths, private reviews or model weights are included.

## Rebuild

Use the installed ArchRivet development environment. Pillow is an optional documentation-authoring dependency; it is not required by the plugin.

```bash
python -m pip install Pillow==12.3.0
python scripts/render_readme_demo.py
```

The renderer runs both CLI commands, validates the expected modes/verdicts, and regenerates the GIF, PNG and metadata in this directory. Use `--output PATH` for a separate preview. It uses available local UI/monospace fonts with a fallback; font files are not distributed, and asset bytes can vary by platform. The renderer supports Windows and Linux font locations.

The artwork and renderer are original ArchRivet project material under the repository's MIT license. The animation does not establish that Laya improves review accuracy; see [the measured experiment](../web-v2-run.md).
