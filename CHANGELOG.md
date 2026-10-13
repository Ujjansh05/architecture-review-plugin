# Changelog

## 0.2.0a1 — experimental backend review workflow

- Add eight scoped backend checks with explicit supported, contradicted and insufficient-information results; preserve the original three-check API.
- Add background review, polling and cancellation over MCP and HTTP, with a shared bounded queue, result expiry and sampled RAM reports.
- Add optional local generative document extraction with exact source spans, cross-section conflict tracking and explicit unprocessed coverage. A 128,000-token document limit is separate from each model call's context limit.
- Add 240 labeled diagnostic fixtures, family/source leakage checks, bound final-test selection manifests, per-check metrics and answer-order stress tests.
- Add reproducible CPU head training, bounded CUDA partial-encoder training and rank-eight encoder adapters; record development selection, calibration, option-order sensitivity and actual RAM/VRAM use.
- Add 360 training representations with unchanged held-out splits and a separate 12-case executable regression packet; reject transformed inputs that overlap across splits.
- Add an opt-in Qwen 4B QLoRA/Google Drive notebook. Cloud GPU training has not been executed; no experimental checkpoint is promoted automatically.
- Reduce cumulative compute reservations to USD 10 inside the owner's USD 15 experiment cap; external provider billing still requires its own hard limits.

## Unreleased — research dataset and validation

- Add a visual README walkthrough, a reproducible GIF of actual rules-only results, and clearer first-run and MCP setup instructions.
- Rename the project and GitHub repository to ArchRivet; preserve the Python distribution, CLI and MCP identifiers for existing installations.
- Publish the corrected 48-case web-only dataset, source/repair provenance and measured CPU training resources; development accuracy remains 4/9.
- Add explicit service startup for Windows MCP so closing one assistant session cannot terminate a daemon launched by that session and interrupt other clients.
- Reject cross-split copies of web decision inputs that differ only in evidence citation IDs.
- Reject headings and resource-link lists as supporting quotes, and instruct the data-generation teacher to use explanatory prose.
- Disable the unused TensorFlow backend inside the PyTorch model worker, including when MCP hosts filter environment variables.
- Add a bounded, cached website scraper, a source-only automatic decision-data generator, and a solo fine-tuning command for 322M-parameter Laya.
- Preserve source-document splits, exact supporting quotes, machine-generated label provenance, and automatic rejection reports without mixing in the older pilot dataset.
- Add head-only CPU and full-model training modes, context checks, and a reproducible web-only smoke experiment.
- Restore training mode after full-trainer development evaluation between epochs.
- Add 18 pinned, licensed human-origin architecture references, 372 attributed chunks, 15 draft lessons, 45 hypothetical decision cases, and a blind reviewer packet.
- Record screening of 11 source candidates, including actual Kaggle/Hugging Face sample audits; exclude samples with unverified label provenance.
- Enforce source-document split grouping and validate optional background evidence in dataset exports.
- Add reproducible published-wheel/MCP/AI-host checks, blind general-AI baselines, and a bounded CPU head-training diagnostic.
- Report no accuracy improvement from the small head-training experiment; retain base Laya as the default and keep trained weights undistributed.

## 0.1.1 — MIT-licensed experimental preview

- Add the owner-selected MIT license to the repository, wheel and source distribution.
- Declare license metadata and preserve third-party notices.
- Supersede the initial 0.1.0 preview with licensed downloads; runtime behavior is unchanged.

## 0.1.0 — experimental

- Local MCP tool, authenticated loopback review API and CLI.
- Three messaging decisions: retry safety, acknowledgement durability and queue overload.
- Evidence retrieval, workload arithmetic, capacity assumptions and verification steps.
- Shared CPU Laya worker, limited concurrency and idle unloading.
- Sixty draft lessons, twelve subsystem studies, one hundred synthetic pilot cases and ten starter regressions.
- Fixed dataset exports, baseline evaluation, calibration/training runner and budget controls.
- Windows checkpoint-loading compatibility and protocol/concurrency tests.

The v2 interface defaults to rules; configured base Laya remains unmodified. GPU training results are exploratory and do not establish an independently validated specialist. Independent review, trials with a second real AI host and human usefulness evaluation remain pending. See [current model-improvement results](docs/model-improvement-status.md) before interpreting model output.
