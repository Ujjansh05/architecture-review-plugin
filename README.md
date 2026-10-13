# ArchRivet

**A second opinion for your system design.**

ArchRivet is a local MCP tool that reviews messaging designs drafted by an existing AI assistant. It checks retry safety, acknowledgement durability and queue overload, retrieves source-backed lessons, and calculates workload/storage in code.

**Status: experimental implementation.** The runtime includes 100 synthetic cases and 60 draft evidence summaries. A separate [research dataset](datasets/README.md) adds licensed references and hypothetical cases. The latest [web-only dataset](datasets/web-solo-v2/README.md) contains 57 references, 598 chunks and 48 source-linked cases. All decision labels await independent expert review. The downloadable checkpoint remains base Laya: the latest CPU head-training experiment left development accuracy unchanged at 4/9. No trained weights are bundled. See the [measured results](docs/web-v2-results.json).

The project's original code, documentation and synthetic examples use the [MIT License](LICENSE). Dependencies, downloaded model weights and cited source materials retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

The separate reference corpus and source-linked adaptations retain CC BY attribution licenses, as recorded per item. They are not part of the MIT software grant.

## Quick start

Clone the project with Python 3.11 or 3.12 installed. On Windows PowerShell:

```powershell
git clone https://github.com/Ujjansh05/archrivet.git
cd archrivet
.\scripts\setup.ps1 -WithModel
$env:ARCH_REVIEW_HOME = "$PWD\.runtime"
.\.venv\Scripts\python.exe -m arch_review doctor
.\.venv\Scripts\python.exe -m arch_review review examples\flawed.json --rules-only
.\.venv\Scripts\python.exe -m arch_review review examples\complete.json
.\.venv\Scripts\python.exe -m arch_review start
.\.venv\Scripts\python.exe -m arch_review mcp-config
```

The setup script installs the CPU runtime, downloads the base Laya checkpoint, and prints your machine's MCP configuration. On Windows, run `arch-review start` from your terminal before connecting an assistant; it starts the shared service without loading the model yet. Add the generated configuration to an MCP-capable assistant and apply [the host instructions](docs/host-instructions.txt). The [configuration template](examples/mcp-config.template.json) shows the format; generate actual paths with `arch-review mcp-config`.

For a quick rules-only trial, omit `-WithModel` and use the `--rules-only` example. Model weights, virtual environments, tokens and private reviews are not included in this repository.

## Install elsewhere

Requires Python 3.11 or 3.12. Create a fresh environment. Install CPU PyTorch before the model extra to avoid an unnecessary GPU runtime:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[model,dev]'
.\.venv\Scripts\python.exe -m arch_review download-model --destination models\laya-multilingual
.\.venv\Scripts\python.exe -m arch_review configure --checkpoint models\laya-multilingual
.\.venv\Scripts\python.exe -m arch_review start
.\.venv\Scripts\python.exe -m arch_review mcp-config
```

The checkpoint download pins the remote revision and records artifact hashes. Installation and model download use the network; review-time model loading is offline. The core install works without torch, returning an explicitly partial rules-only review until a checkpoint is configured.

For the Windows helper, use `scripts/setup.ps1`; `-WithModel` adds the model dependencies and download. Choose a model directory with adequate free space. Allow several GB for the environment, model and temporary installation files.

On Linux, after cloning:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[model]'
export ARCH_REVIEW_HOME="$PWD/.runtime"
arch-review download-model --destination models/laya-multilingual
arch-review configure --checkpoint models/laya-multilingual
arch-review review examples/complete.json
arch-review mcp-config
```

[GitHub Releases](https://github.com/Ujjansh05/archrivet/releases) provide an installable Python wheel and source archive. The current v0.1.1 release predates the web-training workflow and Windows `start` command; use the `main` checkout for the commands in this README. The package is not published on PyPI. macOS model inference has not been validated. [CI](https://github.com/Ujjansh05/archrivet/actions) tests supported Python versions on Windows and Linux.

ArchRivet was previously named `architecture-review-plugin`. The Python distribution retains that name, and the CLI remains `arch-review`, so existing environments and MCP configurations continue to work.

## What the assistant calls

```text
review_architecture(requirements, design, workload, deployment)
```

Use `examples/complete.json`, `examples/flawed.json`, and `examples/incomplete.json` as input examples. `examples/complete.review.json` is explicitly a rules-only output example; `examples/complete.laya.review.json` is a real base-Laya-assisted response. Run `arch-review schema` for the exact machine-readable input/output schema, also saved in `examples/schema.json`.

- `supported`: explicit facts satisfy the specified requirement under its assumptions.
- `contradicted`: explicit facts reveal a violating failure scenario.
- `insufficient_information`: required facts are missing, ambiguous or unresolved.

Unknown facts must be omitted or null. A component name such as Kafka, WhatsApp, or PostgreSQL never establishes a guarantee. `acknowledged_operation` identifies the exact same boundary in requirements and design. `survives_failures` is the exhaustive stated guarantee. Duplicate-protection fields cover every relevant effect and the entire declared retry window.

The service returns rule and model judgments separately. An explicit counterexample cannot be cleared by a model. Unresolved disagreements remain visible. Model unavailability or context overflow produces a partial review; no truncated model verdict is accepted. Model probabilities describe classification uncertainty, never outage probability.

Arithmetic includes rates, fan-out deliveries, retained storage and supplied growth scenarios. Unknown assumptions remain missing. Capacity requires a matching supplied benchmark; its provenance is displayed as user-supplied. Fan-out changes storage only when separate stored copies are specified. Latency targets and future capacity need measurements.

## Local service

MCP and the normal CLI connect to one loopback daemon. It exposes authenticated `POST /v1/reviews`, `GET /health`, and `GET /v1/schema`. Authentication is a random local token in `ARCH_REVIEW_HOME`; do not put it in shared configuration files.

On Windows, start the service from an independent terminal with `python -m arch_review start`, using the same environment and `ARCH_REVIEW_HOME` as your MCP configuration. Repeat after a reboot or `arch-review stop`. Some Windows hosts put each stdio client and its children in a job that terminates them when that client closes. Starting the daemon there would interrupt other assistant sessions, so Windows MCP requires the separately started service. CLI reviews can still start it automatically; POSIX MCP retains automatic startup. `start` reuses an already running service and does not load another model.

The daemon allows one active review plus four waiting requests. A separate CPU model worker loads one checkpoint and exits after five idle minutes. Model failure leaves the daemon available. Run `arch-review stop` before changing settings or code. The MCP process never imports torch and writes only protocol messages to stdout.

Windows loading uses validated, read-only NumPy mappings to avoid a native Torch/Safetensors mapping failure observed during validation. Laya copies these tensors into its model. The adapter preserves checkpoint precision; it supports the referenced F16/F32 Laya checkpoint. Linux retains the upstream loader.

Host instructions are in `docs/host-instructions.txt`. They request a review after an initial draft and one follow-up after a relevant revision. The host decides whether to invoke tools.

## Data and training

For a solo workflow that collects permitted web references and automatically prepares new decision examples, use [the web scraping and fine-tuning guide](docs/solo-web-training.md). It includes three runnable scripts, uses Laya Multilingual (**0.322 billion parameters**), and keeps this web-only experiment separate from the older pilot dataset. No human labeler is required to run it; generated labels remain explicitly unreviewed.

The package contains 20 primary-source references, 60 draft lessons, 12 subsystem case studies, an eight-family coverage matrix, 100 synthetic pilot cases, and ten development regressions. Broad architecture references do not add new automated verdict types.

See `docs/dataset-guide.md` for company coverage, source review, and how to add future architecture families.

```powershell
.\.venv\Scripts\python.exe -m arch_review dataset audit
.\.venv\Scripts\python.exe -m arch_review evaluate --split development --backend rules
.\.venv\Scripts\python.exe -m arch_review dataset export --allow-unreviewed --output training\pilot-export
.\.venv\Scripts\python.exe -m arch_review train --data training\pilot-export --base models\laya-multilingual --output models\candidate --allow-unreviewed --dry-run
```

`--allow-unreviewed` creates explicitly experimental artifacts. Normal training/export requires every case to have an expert reviewer and review date. Exported states and questions use the same rendering and retrieval as inference. Labels/rationales are excluded from general-AI comparison prompts.

`docs/training.md` describes baseline comparison, fixed splits, calibration, budget enforcement and the paid-GPU handoff. Never use synthetic-label agreement as evidence of production safety. Ordinary user reviews are not saved or incorporated into training.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m arch_review benchmark examples\complete.json --runs 5 --output reports\benchmark.json
```

The tests include real stdio MCP subprocesses and two concurrent clients sharing a daemon, including a Windows client-exit regression. These protocol tests are separate from actual AI-host trials and independent backend-engineer comparisons. See [the latest execution report](docs/web-v2-run.md) for which checks completed.

Upstream references: [Laya training](https://nandhakishorm.github.io/laya/finetune/), [Laya API](https://nandhakishorm.github.io/laya/reference/agent/), [MCP server SDK guide](https://modelcontextprotocol.io/docs/develop/build-server). Source-specific architecture references are stored with each evidence record.
