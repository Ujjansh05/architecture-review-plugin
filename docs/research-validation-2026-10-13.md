# Dataset and local training experiment — 2026-10-13

The reference collection and experimental workflow are implemented. The evidence does **not** justify shipping a trained specialist: the small head-only training run did not improve decision accuracy. Independent human label review and a second authenticated AI-host trial remain outstanding.

## What was collected

- Screened 11 candidate collections/sites across Hugging Face, Kaggle, GitHub, and publishers.
- Collected 18 pinned, licensed documents from the System Design Primer and The Architecture of Open Source Applications; produced 372 attributed, hashed text chunks.
- Added 15 AI-written source interpretations and 45 source-linked hypothetical decision cases, grouped by source and scenario before generating variants.
- Created a blind 45-row human-review CSV. Reviewer names, verdicts, rationales, and dates are blank.
- Combined the new cases with the original pilot: 145 draft cases, split 78 train / 24 development / 19 calibration / 24 test. The test split was not evaluated.

The references describe real implementations and teaching exercises across distributed storage, key-value databases, metrics, web servers, asynchronous networking, messaging, wiki applications, feeds, transaction imports, caching, rankings, graph workloads, and crawlers. They are not a dataset of universally perfect architectures or current private company internals.

The downloaded Kaggle archive contained **200** rows rather than the description's 250, including 24 System Design questions. It lacked source citations and independent review metadata. The downloaded Hugging Face instruction collection contained **60** records with no documented human authorship/review procedure. Both remain quarantined outside the published training/evaluation corpus. Another large Hugging Face software-architecture collection is explicitly synthetic according to its uploader. See [the screening audit](../datasets/source-screening.json).

GeeksforGeeks was not bulk-scraped: its [terms](https://www.geeksforgeeks.org/legal/terms-of-use/) prohibit systematic database collection and automated extraction without written permission. The accepted alternatives have explicit [CC BY 4.0](https://github.com/donnemartin/system-design-primer/blob/ae9bbd7b02d90b9866215de185217d33f39ab733/LICENSE.txt) and [CC BY 3.0](https://aosabook.org/en/) content licenses. A license found on a repository is checked for scope: Mixu's CC notices apply to icons/logo, while its book republication requires permission, so that book was excluded.

## Measured decision results

All methods received the same 24 development scenarios. Laya and the general-AI baseline received the same exported states, evidence, and decision criteria. Rules operate on the explicit structured facts. The general-AI run received no expected labels or rationales and made no tool calls.

| Method | Draft-label agreement | False supported verdicts | Critical false supported verdicts |
|---|---:|---:|---:|
| Deterministic rules | 24 / 24 | 0 | 0 |
| Base Laya | 10 / 24 | 14 | 8 |
| Codex general-AI baseline | 23 / 24 | 1 | 0 |
| Laya after head-only training | 10 / 24 | 14 | 8 |

The general-AI disagreement was `expiry-2`: it returned supported where the draft label requires insufficient information. Base Laya changed 7 of 24 answers when answer order was reversed; the selected trained head changed 8. Unfitted trained-head accuracy and fitted-temperature accuracy were both 10/24. Temperature fitting reduced development Brier score from about 0.883 to 0.669, but this tiny result is not validated calibration and did not fix false approvals.

These examples are deliberately related to the implemented rules. Their labels are AI-authored and none have independent expert review. Rule agreement is therefore an implementation consistency check, not independent proof of correctness. The reference documents do not constitute expert labels for the new hypothetical cases. Treat all comparative numbers as exploratory.

The general-AI host was Codex CLI 0.162.1 using its default model. The underlying model identifier was not emitted in the recorded result, which limits model-level reproducibility. The blind run took 23.5 seconds for one batch of 24 cases. This is not directly comparable with single-request local inference latency; batching, network services, context, and model size differ.

## Actual training run

The experiment used the pinned Laya 0.4.6 API and multilingual checkpoint revision `1720e3e3357cfe1e281542e223f8273b0890ca34`, CPU inference/training, four threads, seed 0, two epochs, soft cross-entropy, shuffled answer options, micro-batch 2 and gradient accumulation 4. The encoder was frozen. Epoch 1 was selected by development accuracy; the tie retained that first epoch. Calibration used the separate 19-case calibration split. A 30-minute process timeout bounded the experiment; measured work took about **156 seconds**.

The saved non-encoder artifact is **59,880,768 bytes**. Save/load equality, unchanged frozen encoder, and unchanged original checkpoint were verified. It requires the exact base checkpoint and is not a standalone MCP model or a release-approved specialist. It stays local and is not distributed as production weights. Paid GPU spend was zero.

Limited disk space prevented a second full checkpoint, but the head diagnostic fit without deleting user files or replacing the existing global CUDA environment. A 6 GB NVIDIA GPU is present; this run used the installed CPU runtime. This experiment cannot establish whether a larger reviewed dataset and full fine-tuning would help.

## Package and host validation

The public v0.1.1 wheel was downloaded from GitHub, matched against both published SHA256SUMS and GitHub's asset digest, and installed into a separate environment. The import path was checked to ensure the wheel, rather than the workspace source, was used. Complete, flawed, and incomplete examples returned real Laya decisions through the CLI and three concurrent stdio MCP protocol clients.

Dependencies and the model checkpoint were shared with the existing local installation to fit disk space; this was a fresh **package** test, not a fresh dependency installation. The existing release CI separately checks clean dependency installation. The space-saving harness initially omitted pywin32's `.pth` initialization; processing the trusted dependency paths corrected that test-environment issue.

Codex successfully connected to the downloaded package and made three actual MCP review calls. An initial host pass exposed tool-output truncation; a per-tool output limit of 24,000 tokens corrected it in the final test. The [retained host answer](codex-host-trial.txt) correctly reported all nine verdicts, rule/model disagreements, and missing benchmark evidence. Claude was installed but `claude auth status` reported `loggedIn: false`; no Claude model call or second-host success is claimed. Global host configuration was not modified.

All **39 automated tests** passed after training released memory; Ruff passed. An earlier concurrent test run hit a memory allocation failure while importing Torch during training, so the verification was rerun sequentially. This is a practical resource limitation, not evidence that the plugin and every AI host fit in a fixed RAM budget.

## Reproduce and continue

```powershell
python scripts/prepare_research_experiment.py
arch-review evaluate --source .runtime/research-experiment/cases.jsonl --backend rules --output reports/rules.json
arch-review evaluate --source .runtime/research-experiment/cases.jsonl --backend laya --checkpoint PATH_TO_BASE --perturb --output reports/laya.json
python scripts/run_codex_baseline.py --inputs .runtime/research-experiment/export/general_ai_inputs.jsonl --codex PATH_TO_CODEX --output .runtime/general-ai
arch-review evaluate --source .runtime/research-experiment/cases.jsonl --predictions .runtime/general-ai/predictions.jsonl --output reports/general-ai.json
python scripts/train_head_diagnostic.py --data .runtime/research-experiment/export --base PATH_TO_BASE --output EMPTY_OUTPUT_DIRECTORY --epochs 2 --timeout-seconds 1800
```

The next useful work is independent human review and harder, independently authored scenarios. Review ambiguous policy/cap combinations, inferred failure guarantees, source applicability, retry scope, and external effects. Add measured failure-injection examples and realistic incomplete prose designs. Preserve source/scenario groups and an untouched test set. Full fine-tuning is only worth promoting after it improves held-out defect detection without increasing false approvals.

Machine-readable measurements are in [research-results.json](research-results.json); source licensing, reproduction, and labeling instructions are in [the dataset guide](../datasets/README.md).
