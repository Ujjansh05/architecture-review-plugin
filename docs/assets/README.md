# ArchRivet demo

## Project icon

The original ArchRivet mark is a connected **A** with an amber rivet. A soft light travels along the connections in the animation; it is branding, not review progress or a success indicator.

- [Animated GIF](archrivet-icon.gif) for the GitHub README and sharing.
- [Animated SVG](archrivet-icon-animated.svg) for compatible websites; respects `prefers-reduced-motion`.
- [Static SVG](archrivet-icon.svg), [512 px PNG](archrivet-icon-512.png), and [128 px PNG](archrivet-icon-128.png).
- [Preview at different sizes](archrivet-icon-preview.png).

The README provides a static PNG for browsers that request reduced motion. The desktop MCP manifest uses static PNGs because that is the supported icon format. The server also supplies an embedded PNG during MCP initialization; whether it appears is controlled by the assistant host. No remote image request is needed.

To rebuild the artwork, install Pillow 11.3.0 in an authoring environment and run `python scripts/render_brand_icon.py`. Pillow is not added to the plugin's runtime dependencies. The original artwork and renderer use the repository's MIT license.

## Product walkthrough

The [animated flowchart](archrivet-flow.gif) follows the recorded price-cache example through the assistant, local checks, missing-detail question, clarification and revised finding. The [still diagram](archrivet-flow.png) shows the whole cycle; the [frame preview](archrivet-flow-preview.png) shows four stages. It illustrates rules-mode behavior, not a live run, model accuracy or response timing. Optional Laya is identified separately and measured in the published benchmark. Rebuild with `python scripts/render_flowchart.py` using the same Pillow authoring environment as the icon.

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
