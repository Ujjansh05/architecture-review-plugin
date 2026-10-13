# Changelog

## Unreleased — research dataset and validation

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

The default checkpoint is unmodified base Laya. Specialist training, independent expert review, two actual AI-host trials and human usefulness evaluation remain pending. See [validation](docs/validation.md) before interpreting model output.
