<div align="center">

# ArchRivet

### A second opinion for your system design.

Your AI drafts the architecture. **ArchRivet checks the messaging risks, brings evidence, and does the workload math.**

[![CI](https://github.com/Ujjansh05/archrivet/actions/workflows/ci.yml/badge.svg)](https://github.com/Ujjansh05/archrivet/actions/workflows/ci.yml) [![Status: experimental alpha](https://img.shields.io/badge/status-experimental_alpha-f5a623)](#what-is-proven-today) [![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776ab)](#try-it-locally) [![Code license: MIT](https://img.shields.io/badge/code_license-MIT-14b8a6)](LICENSE)

[Watch the demo](#see-it-in-action) · [Try it locally](#try-it-locally) · [Connect your AI](#connect-your-ai) · [Results & limits](#what-is-proven-today)

</div>

> [!NOTE]
> **Experimental alpha.** ArchRivet currently reviews three messaging decisions. Findings depend on the facts you supply; they do not certify a deployment or establish production capacity.

## See it in action

**One flawed design. Three concrete findings. A revision you can check again.**

![Animated ArchRivet walkthrough: review a flawed design, add duplicate protection, fix acknowledgement ordering and bound the queue, then review the revision. Capacity remains unknown without a benchmark.](docs/assets/archrivet-demo.gif)

*Animated summary of actual CLI results, with shortened wording and presentation timing. This is the rules-only path: no model download required. [Static preview](docs/assets/archrivet-demo.png) · [Text walkthrough and reproduction](docs/assets/README.md)*

Suppose your requirements say retries must not repeat business writes, acknowledged messages must survive process and host failures, and the queue must stay bounded. Your first draft has message IDs, early acknowledgements and an unbounded queue:

| Decision | What ArchRivet finds | Change to review next |
|---|---|---|
| **Retry safety** | `contradicted` — IDs exist, but duplicate protection is not enforced atomically with the write. | Enforce duplicate protection with the business effect and cover the full retry window. |
| **Acknowledgement durability** | `contradicted` — success is returned before durable commit. | Acknowledge after a commit that covers the required failures. |
| **Queue overload** | `contradicted` — sustained overload can grow the queue without a bound. | Bound the queue and reject excess work before accepting it, with a retry policy. |

The [revised example](examples/complete.json) supplies those mechanisms, so all three rule verdicts become `supported`. The review still asks for failure and load tests. It returns **capacity unknown** because neither example supplies a matching benchmark.

For the same example workload, the code calculates **500 peak messages/second**, **1,500 peak deliveries/second** and **0.956 TB retained storage**. Those estimates use 100 average messages/second, a 5× peak, three recipients, 1,024-byte messages, 30-day retention, one stored copy, replication factor three and 20% storage overhead. They describe demand, not measured server capacity.

## Where it helps

| You are working on… | Use ArchRivet to… |
|---|---|
| A chat, notification or messaging backend | Check retry, acknowledgement and queue policies before implementation. |
| An architecture drafted by an AI assistant | Request explicit facts and surface contradictions or missing information. |
| A workload or growth estimate | Calculate rates, fan-out and storage with visible formulas and assumptions. |
| A system-design exercise | Connect a proposed mechanism to a failure scenario, source lesson and verification test. |

Each review includes **decision labels, reasons, risk conditions, source references, missing facts and suggested tests**. Model confidence is kept separate from production failure risk. A product name such as Kafka or PostgreSQL does not establish a guarantee.

## How it works

```mermaid
flowchart LR
    A[Your idea and requirements] --> B[Your AI drafts a design]
    B --> C[ArchRivet via MCP]
    E[Source-backed lessons] --> C
    F[Rules and workload calculations] --> C
    G[Local Laya decisions] --> C
    C --> D[Findings and missing information]
    D --> H[Your AI explains and revises]
```

The host assistant calls one tool:

```text
review_architecture(requirements, design, workload, deployment)
```

| Verdict | Meaning |
|---|---|
| `supported` | Explicit facts satisfy the checked requirement under its assumptions. |
| `contradicted` | Explicit facts reveal a violating failure scenario. |
| `insufficient_information` | Required facts are missing, ambiguous or unresolved. |

Rules and model judgments are returned separately. An explicit rule contradiction survives a model approval, and disagreements remain visible. Unknown facts should be omitted or null. Oversized model inputs and unavailable inference produce a partial review.

## Try it locally

**Start with the rules-only demo.** You need Git and Python **3.11 or 3.12**. It requires no model weights, GPU or AI account. Installation downloads Python dependencies.

Use the `main` checkout below. The packaged **v0.1.1** release predates the latest web-training workflow and Windows startup fix; the project is not published on PyPI.

```bash
git clone https://github.com/Ujjansh05/archrivet.git
cd archrivet
```

<details open>
<summary><strong>Windows · PowerShell</strong></summary>

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m arch_review review examples\flawed.json --rules-only
.\.venv\Scripts\python.exe -m arch_review review examples\complete.json --rules-only
```

</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
arch-review review examples/flawed.json --rules-only
arch-review review examples/complete.json --rules-only
```

</details>

The CLI returns JSON. `review_mode: "rules_only"` and `review_status: "partial"` are expected: this path intentionally skips model inference. See the [flawed](examples/flawed.json), [revised](examples/complete.json) and [incomplete](examples/incomplete.json) inputs, or the [full output schema](examples/schema.json).

## Connect your AI

Add the CPU model runtime, then copy the generated MCP entry into your assistant's MCP configuration. Allow several GB of disk space for dependencies, model weights and installation files. Reuse an existing checkpoint if you have one.

<details open>
<summary><strong>Windows · add Laya and start the shared service</strong></summary>

From the cloned repository in PowerShell:

```powershell
.\scripts\setup.ps1 -WithModel
$env:ARCH_REVIEW_HOME = "$PWD\.runtime"
.\.venv\Scripts\python.exe -m arch_review start
.\.venv\Scripts\python.exe -m arch_review mcp-config
```

On Windows, run `start` in an independent terminal before connecting the assistant, and again after a reboot or `stop`. Keep the same `ARCH_REVIEW_HOME` as the generated configuration. The model loads on the first review.

If PowerShell blocks the helper script, the manual commands below work without changing its execution policy.

</details>

<details>
<summary><strong>Windows · manual model installation</strong></summary>

With the core environment already installed:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[model]'
$env:ARCH_REVIEW_HOME = "$PWD\.runtime"
.\.venv\Scripts\python.exe -m arch_review download-model --destination models\laya-multilingual
.\.venv\Scripts\python.exe -m arch_review configure --checkpoint models\laya-multilingual
.\.venv\Scripts\python.exe -m arch_review start
.\.venv\Scripts\python.exe -m arch_review mcp-config
```

</details>

<details>
<summary><strong>Linux · add the CPU model runtime</strong></summary>

With the virtual environment from the quick start activated:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[model]'
export ARCH_REVIEW_HOME="$PWD/.runtime"
arch-review download-model --destination models/laya-multilingual
arch-review configure --checkpoint models/laya-multilingual
arch-review start
arch-review mcp-config
```

</details>

Add the generated entry to your host and apply the [host instructions](docs/host-instructions.txt). Each person generates configuration for their own machine; the [template](examples/mcp-config.template.json) contains placeholders. MCP support is required. One actual Codex host has been tested; other hosts need verification.

**Try this prompt after connecting:**

> Help me design a messaging backend. Ask about requirements that are missing. Draft the design, then use ArchRivet's `review_architecture` tool to check retries, acknowledgement durability and queue overload. Explain the findings, cite the returned evidence, and keep unknown capacity and unresolved risks visible.

Review one architecture per call. Large combined tool outputs can be truncated. The host decides whether to invoke the tool; if it cannot, it should say the review is unavailable.

## What is proven today

The [recorded validation run](docs/web-v2-run.md) separates software behavior from model quality:

| Measurement | Observed result |
|---|---|
| Local regression tests | **66 passed**, with lint and dependency checks passing. |
| Actual Codex integration | **9/9 combined plugin verdicts** preserved across three known examples. |
| Laya head fine-tuning | Development accuracy stayed at **4/9 before and after**. |
| Warm plugin responses in that trial | **0.51–0.87 seconds**; initial model load plus one review took **10.7 seconds**. |
| Peak sampled plugin service-tree memory | Approximately **1.98 GiB**, excluding the host AI. |

These are small local measurements, not general performance guarantees. Summed process memory can count shared pages more than once; the full three-case host trial took about 118 seconds.

**The default checkpoint is base Laya Multilingual, approximately 322M parameters.** The trained head is experimental and is not bundled or activated. Laya alone approved all nine integration checks; explicit rules preserved the combined plugin's contradictions and missing-information results. The current evidence does not establish an improvement over a general AI assistant.

**Current limits:** three messaging decisions; AI-generated labels awaiting independent review; uneven evaluation coverage; one tested AI host; no validated production-capacity prediction. Published company architectures are dated subsystem references or design exercises. macOS model inference has not been validated. Local plugin inference runs offline after setup; the host AI follows its own provider and data-handling settings.

## Explore and contribute

| Goal | Start here |
|---|---|
| Understand the exact three decisions | [Product specification](docs/product-specification.md) |
| Use reference material and company examples | [Dataset guide](docs/dataset-guide.md) |
| Inspect the latest web-only dataset | [57 references, 598 chunks, 48 generated cases](datasets/web-solo-v2/README.md) |
| Scrape permitted sources and train locally | [Solo web-training guide](docs/solo-web-training.md) |
| Run baselines and evaluate a checkpoint | [Training and evaluation](docs/training.md) |
| Inspect what was measured | [Execution report](docs/web-v2-run.md) · [MCP/host results](docs/web-v2-mcp-results.json) |
| Rebuild the animation | [Demo sources and instructions](docs/assets/README.md) |

The runtime includes 100 synthetic pilot cases and 60 draft evidence summaries. Research datasets are separate and are not automatically added to runtime retrieval. The solo training workflow uses an authenticated AI host to generate source-linked hypothetical cases; those labels are explicitly marked as machine-generated. Ordinary reviews are not automatically saved or used for training.

**Useful feedback:** share a reproducible example, the requirement you expected to be checked, the actual verdict, and the missing fact or failure scenario. Read [CONTRIBUTING.md](CONTRIBUTING.md), then [open an issue](https://github.com/Ujjansh05/archrivet/issues).

<details>
<summary><strong>Developer checks and service details</strong></summary>

With the virtual environment activated:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests scripts
```

The authenticated loopback service exposes `POST /v1/reviews`, `GET /health` and `GET /v1/schema`. Its token lives in `ARCH_REVIEW_HOME` and is not part of shared MCP configuration. One service handles one active review plus four waiting requests; one CPU model worker is shared across sessions and unloads after five idle minutes.

Use `arch-review doctor` to inspect setup and `arch-review stop` before changing settings or code. Windows requires independent startup because a stdio client's cleanup job can terminate services launched beneath that client. Windows checkpoint loading uses a validated NumPy mapping adapter; Linux uses the upstream loader. These implementation details and their validation history are in the linked reports.

ArchRivet was previously named `architecture-review-plugin`. The distribution retains that name; the CLI remains `arch-review` and the Python module remains `arch_review` for compatibility.

</details>

## License and acknowledgements

Original code and original synthetic examples use the [MIT License](LICENSE). Source text and source-derived adaptations retain their per-item **CC BY 3.0 or CC BY 4.0** terms. Model weights and dependencies retain upstream licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).

Built with [Laya](https://nandhakishorm.github.io/laya/reference/agent/) and [MCP](https://modelcontextprotocol.io/docs/develop/build-server). Engineering references are attributed in the evidence records and datasets.
