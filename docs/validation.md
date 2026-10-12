# Local validation — 2026-10-13

**Historical first-run report.** Later package downloads, dataset screening, general-AI baselines, and head-only training are recorded in [the research validation update](research-validation-2026-10-13.md). The measurements below describe the earlier 100-case pilot and predate that experiment.

The local MCP service works with the real, unmodified Laya multilingual checkpoint. This is an experimental software implementation. Specialist training, independent label review, and product-quality validation remain unfinished.

## Software and protocol

- 36 automated tests passed; Ruff passed; `pip check` reported no broken requirements.
- Tests exercise arithmetic, missing facts, duplicate-effect failure scenarios, durability boundaries, queue handling, malformed input, authentication, concurrency, export integrity and split leakage.
- A Windows checkpoint regression verifies dtype/value preservation, scalar/empty tensors, malformed-file rejection, and independent model weights after copying.
- Two concurrent stdio MCP clients successfully reviewed complete/flawed examples through one HTTP daemon and one real model worker. All six model decisions were returned. See `reports/mcp-model-smoke.json` and `scripts/smoke_mcp.py`.
- These clients are protocol clients, not two AI products. No actual AI host was configured or human usefulness study performed.

## Local performance

Five consecutive reviews on Windows 11, Intel processor with 14 physical/20 logical cores, approximately 16 GB installed RAM, four inference threads, CPU only:

| Measurement | Observed |
|---|---:|
| First review, including cold model load | 7.80 seconds |
| Median of four subsequent reviews | 1.27 seconds |
| Maximum of four subsequent reviews, nearest-rank sample p95 | 1.36 seconds |
| Peak aggregate resident memory | 2,392,297,472 bytes / 2.23 GiB |
| Requests receiving all three real model decisions | 5 / 5 |

Memory sampling includes the daemon, model descendants and benchmark CLI. Summed resident memory can count shared pages twice. This small sample does not estimate a reliable long-run p95, and the system was not tested on an actual 8 GB machine. An AI host would add its own memory use. No quantization was applied.

Raw measurements are in `reports/benchmark.json`; the durable summary is in `docs/validation-results.json`.

## Baseline quality

Evidence revision: `ee7279815f09f325`. Checkpoint revision: `1720e3e3357cfe1e281542e223f8273b0890ca34`. Fifteen development examples, one labeled decision each, using synthetic labels that await expert review:

| Baseline | Label agreement | False supported verdicts | Critical false supported verdicts |
|---|---:|---:|---:|
| Deterministic rules | 15 / 15 | 0 | 0 |
| Unmodified Laya | 6 / 15 | 9 | 6 |

The base model chose supported for all original-order development questions. Reversing answer order changed five answers. Its multiclass Brier score was 0.890 and ten-bin ECE was 0.462. These are exploratory measurements on a tiny draft dataset, not validated calibration or production-risk estimates.

Rules and pilot cases implement the same narrow definitions; perfect agreement is not independent evidence of generalization. The final service retains explicit rule contradictions even when Laya approves. A general-AI baseline has an implemented import path, but no independent general-AI comparison was run.

## Data and training

The collection has 60 draft lessons from 20 primary references, 12 subsystem case studies, and 100 pilot cases across 20 scenario families. The split is 60 train / 15 development / 10 calibration / 15 final test. All variants remain with their family. Zero cases or source summaries have been independently expert-reviewed.

Training export integrity and the tokenizer preflight passed. Maximum state lengths were 337 / 329 / 336 / 311 tokens across those four splits, below the 760-token state budget. The actual optimizer/calibration training run has not been executed. No specialized checkpoint or independently validated calibration is claimed. Final-test model evaluation was not run.

No paid GPU job was launched; GPU spend was zero. The installed CPU runtime cannot run CUDA training. At the end of setup, neither local volume had enough free space for another full checkpoint plus headroom (approximately 148 MB free on C: and 323 MB on D:). The trainer now checks output space before starting. Existing user files and the global CUDA environment were not removed or replaced.

## Remaining release work

1. Have backend engineers review the three definitions, source applicability and pilot labels; author independent final scenarios.
2. Use a training environment with adequate output storage, run the baselines with reviewed labels, then fine-tune and calibrate the specialist. Enforce both the local compute budget and the provider's whole-job timeout.
3. Compare the trained candidate, base model, rules and a general AI on frozen scenarios. Investigate false approvals and answer-order instability before release.
4. Configure two chosen MCP-capable AI hosts and conduct a blind engineer comparison. Measure complete host-plus-plugin RAM on the intended hardware.

The next executable commands and reviewer procedure are in `docs/training.md`; host instructions and the local MCP entry are already provided.
