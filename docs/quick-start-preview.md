# Try ArchRivet with your assistant

This is an **experimental Windows x64 preview**. Start with the lightweight Rules bundle, or choose the separate Laya CPU bundle. No training is needed.

## Download and connect

**Desktop extension:** download `archrivet-windows-x64-rules-0.3.0-alpha.1.mcpb` from the [0.3.0a2 release](https://github.com/Ujjansh05/archrivet/releases/tag/v0.3.0a2). Install it in a host that supports **MCPB 0.4 with the UV runtime**. Such a host manages Python and dependencies; you do not run terminal commands. This is a direct download, not an approved extension-directory listing. Actual Claude Desktop GUI testing remains pending.

**Codex on Windows:** download `archrivet-windows-x64-rules-0.3.0-alpha.1.zip` from the same release and extract it to `C:\Apps\ArchRivet`. The folder should contain `server`, `vendor`, `pyproject.toml` and `uv.lock`. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available. With Codex installed and signed in, run this once:

```powershell
codex mcp add archrivet -- uv run --frozen --managed-python --directory "C:\Apps\ArchRivet" server/main.py
```

Open a new Codex conversation. Initial setup needs internet and disk space for Python and dependencies. The Rules bundle does not install Torch or Laya. The Laya variant additionally needs several GB for its dependencies, download cache and model. If `uv` is not found, use the full path to `uv.exe` in the command. [Official Codex MCP documentation](https://developers.openai.com/codex/mcp/)

## Ask one clear question

> Use ArchRivet to review my product-price cache. Prices must never be more than five seconds old. Normal responses are at most three seconds old. During a database outage we serve old data, but I have not specified how old it may be. Other failure paths obey the declared age bounds. Name the missing fact and keep capacity unknown without a matching load test.

The assistant gets the schema, submits your facts with their origins, and retrieves the review. It may first ask you to confirm its interpretation of the prose. ArchRivet then asks how old prices can become during an outage. A matching quote alone does not validate the assistant's interpretation.

Then say:

> During an outage we may serve prices up to 60 seconds old. Rerun the review and compare it with the previous result.

The five-second requirement now has an explicit violation. A proposed fix is not treated as an implemented guarantee. Ask the assistant to retain the tool's review summary so unresolved facts and limitations stay visible.

## Optional local Laya model

Replace the Rules bundle with `archrivet-windows-x64-laya-cpu-0.3.0-alpha.1.mcpb` (or its ZIP for Codex). Both variants identify as `archrivet`: replace the installation/registration rather than registering both. Existing model files can be reused; switching to Rules does not delete them.

For a compatible desktop host, enable **Enable experimental local Laya model** in the Laya bundle's extension settings. For Codex, point the existing registration at the extracted Laya bundle and explicitly opt in:

```powershell
codex mcp add archrivet --env ARCHRIVET_ENABLE_LAYA=true -- uv run --frozen --managed-python --directory "C:\Apps\ArchRivet" server/main.py
```

This downloads the complete merged checkpoint (about 1 GB; roughly 1.3 GB extracted) and uses CPU inference. Ask the assistant to call `get_setup_status`; `files_ready` means verified files, `loading_model` means the worker is starting, and `ready` means it has loaded. Idle unloading releases model memory. Use `cancel_model_setup` to cancel setup and `retry_model_setup` to retry. SHA-256 checks protect the model download. No separate base model or adapter is required. Interrupted downloads can resume when the server supports byte ranges.

The new model and runtime must be used together: this checkpoint requires the new fact formatter and a 2,048-token context. Do not load it into the old 0.2 runtime. Confidence is not the chance that your production system will fail. See the [benchmark and remaining errors](benchmarks/2026-10-preview.md) before enabling it.

## What stays local

ArchRivet runs its checks locally, does not train on your design, and does not send it to a remote inference service. Your assistant's own provider and storage settings still apply. Installation contacts package registries and GitHub. Review jobs are kept in session memory; model files and the evidence index remain on disk. Separate host sessions can each load a model process.

The bundle opens no HTTP listener and requires no separately started service. It supports eight bounded decisions, not every architecture question. Workload calculations estimate demand; tested capacity needs a relevant load test.
