# Model improvement status

All-layer LoRA reached **25/36 (69.44%) on synthetic development cases**, up from the base model's 11/36, but **stayed at 4/12 (33.33%) on separate executable regression cases**, equal to the base model. It recognized none of the four regression unknowns. The development gain did not transfer to accuracy on these narrow untrained local traces. The candidate remains **research-only and was not promoted**. Software behavior, model-only scores and host integration results remain separate.

## Dataset and baseline evidence

The new [backend diagnostic dataset](../datasets/backend-diagnostics-v1/) contains **240 template-generated examples**, with 120 training, 36 development, 36 calibration and 48 test cases. It covers all eight checks and all three verdicts. Scenario families are assigned to one split, but sibling templates can still share wording and assumptions; this is not independent evidence of generalization.

The separate [augmented training export](../datasets/backend-augmented-v1/README.md) contains **360 training, 36 development, 36 calibration and 48 test rows**. It keeps the original 120 training rows and adds 240 representations of those same cases: field-specific English and structured facts serialized inside the existing training wrapper. There are **315 unique training inputs and zero new independent cases or labels**. All held-out split bytes are unchanged. This tests input representation, not an expanded independent benchmark; the JSON view is not the runtime's exact top-level facts format. Reproduce the export with [augment_backend_training.py](../scripts/augment_backend_training.py).

The recorded development baselines currently show:

| Backend | Development label agreement | Interpretation |
|---|---|---|
| Rules | 36/36 | A consistency check against template-generated labels, not independent accuracy. |
| Unmodified Laya | 11/36; macro F1 about 0.248; 10 false-supported results | The current base model does not reliably solve the expanded task. |
| Unmodified Laya, neutral-label ablation | 12/36; macro F1 about 0.296; 13 false-supported results | Answer-order changes altered 32/36 verdicts; the alternate prompt is not a reliable improvement. |
| General AI through Codex CLI | 36/36; no false-supported results | Blind, shuffled cases without tools or gold labels; all 36 cases shared one batch. |

Sources: [rules results](backend-rules-results.json), [base Laya results](backend-laya-base-results.json), [neutral-label ablation](backend-laya-neutral-results.json), and [general-AI results](backend-general-ai-results.json). Training results follow below. These measurements reuse 36 small development examples for candidate selection; they are not a fresh held-out estimate after each experiment. The 48-case final test split was not evaluated. The reports with independent quality gates retain `independent_quality_release_approved: false`, and all trained artifacts have `release_approved: false`. Numeric gates alone cannot turn synthetic labels into independent validation.

Neutral labels correctly identified only 1/12 missing-information cases. Its 32/36 answer-order changes (88.89%) and increased false support outweigh the one additional matching label. This ablation was not promoted and does not establish a better model.

The [general-AI run record](backend-general-ai-run.json) reports 31.828 seconds through Codex CLI 0.162.1. Case identifiers were anonymized, order was shuffled with seed 17, and neither tools nor gold labels were supplied. Its default underlying model identifier was not emitted. This is not a controlled comparison of identical context: the general AI saw all 36 cases together, while Laya saw one case per call. Template/rule-label agreement and that batch-context advantage prevent a generalization claim. This prediction-only run also did not exercise the new MCP job tools.

The earlier web-only three-check experiment remained at 4/9 development labels before and after head training. That is a separate historical result, not the new eight-check experiment.

## What the completed CPU training runs established

The standard-label CPU run used **120 training, 36 development and 36 calibration cases**, three epochs and seed 17. Candidate selection ranked development macro F1, then fewer false-supported results, and kept the earliest epoch on an exact tie; epoch 1 was selected. Inputs and evidence were identical for before/after evaluation. The saved head round-trip, unchanged encoder and unchanged base checkpoint were verified in the [experiment report](backend-head-seed17-results.json), which includes dataset and weight hashes.

| Measurement | Before | After training and small-sample calibration |
|---|---|---|
| Correct development labels | 11/36 | 11/36 |
| Macro F1 | 0.2484 | 0.2484 |
| False-supported predictions | 10 | 10 |
| Correct missing-information labels | 0/12 | 0/12 |
| Cases changing verdict under answer-option permutations | 5/36 (13.89%) | 5/36 (13.89%) |
| Development ECE, ten bins | 0.40268 | 0.08002 |

All six answer-option orders were checked before and after. Temperature calibration used the separate 36-case calibration split. The lower development ECE describes this small diagnostic sample, not validated confidence or production-failure risk; the report marks it `fitted_unvalidated_small_sample`. It did not improve classification or missing-information handling.

The supervised run took **818.172 seconds**. Sampled peak training-worker process-tree RSS was **2,222,624,768 bytes (about 2.07 GiB)**, excluding the supervisor; sampled RSS can miss peaks and count shared pages more than once. The saved non-encoder head is **59,880,768 bytes (about 57.1 MiB)** and requires its exact base checkpoint. It is not a standalone deployable model or the active plugin checkpoint. The run spent no paid GPU budget.

The second [neutral-label head run](backend-head-neutral-seed17-results.json) also used three CPU epochs, seed 17 and the original 120/36/36 splits, selecting epoch 1. Accuracy remained 12/36, macro F1 remained 0.2960, false support remained 13, and missing-information recall remained 1/12. Answer-order changes still affected 32/36 cases (88.89%). Temperature-only calibration on the calibration split lowered measured development ECE from 0.20301 to 0.03994 without improving verdicts or stability. Its calibration remains fitted but unvalidated, and the head was not promoted.

## Hardware and training work

The recorded local preflight found approximately **2.8 GiB of free workspace disk** and an RTX 3050 Laptop GPU with **6 GiB VRAM**. The larger-model workflow requests at least **30 GiB free workspace** before downloading and training; the local storage check failed. A GPU being listed does not establish that the installed Python runtime can use CUDA or that a particular fine-tuning configuration fits. See [preflight evidence](model-improvement-preflight.json).

A separate environment built from a cached PyTorch 2.6.0 CUDA 12.4 wheel passed an actual CUDA tensor probe and completed the following runs on `cuda:0`. The existing CPU environment is unchanged. The partial-encoder experiments update encoder layers 20 and 21 plus the head: **24,999,427 trainable parameters** out of approximately 322 million. The all-layer LoRA experiments train **16,658,947 parameters**, including the head. Runs used three epochs unless marked otherwise, FP32 training/evaluation and an 80% VRAM allocator limit. Reports verify unchanged frozen/base weights, changed selected weights or adapters, and saved-delta round-trips.

| GPU experiment | Correct labels | Macro F1 | False support | Correct unknowns | Order-sensitive cases | Wall seconds |
|---|---|---|---|---|---|---|
| Base model before training | 11/36 | 0.2484 | 10 | 0/12 | 5/36 | — |
| [Original 120 rows, seed 17](backend-partial-cuda-narrative-seed17-results.json) | 11/36 | 0.2484 | 10 | 0/12 | 8/36 | 57.875 |
| [Augmented 360 rows, seed 17](backend-partial-cuda-augmented-seed17-results.json) | 11/36 | 0.2484 | 10 | 0/12 | 10/36 | 71.344 |
| [1,080 steps, seed 17](backend-partial-cuda-steps-seed17-results.json) | 12/36 | 0.2976 | 7 | 1/12 | 9/36 | 83.188 |
| [1,080 steps, seed 29](backend-partial-cuda-steps-seed29-results.json) | 13/36 | 0.3418 | 6 | 2/12 | 12/36 | 81.547 |
| Neutral-label base before training | 12/36 | 0.2960 | 13 | 1/12 | 32/36 | — |
| [Neutral labels, 1,080 steps, seed 17](backend-partial-cuda-neutral-steps-seed17-results.json) | 11/36 | 0.2963 | 4 | 5/12 | 30/36 | 81.297 |
| [All-layer LoRA, three epochs, seed 17](backend-lora-cuda-seed17-results.json) | 18/36 | 0.5038 | 6 | 6/12 | 14/36 | 188.750 |
| [All-layer LoRA, six epochs, seed 17](backend-lora-cuda-6ep-seed17-results.json) | 23/36 | 0.6519 | 6 | 9/12 | 17/36 | 377.109 |
| [All-layer LoRA, six epochs, seed 29](backend-lora-cuda-6ep-seed29-results.json) | 25/36 | 0.6919 | 4 | 11/12 | 15/36 | 360.359 |

The first two runs used gradient accumulation of eight and selected epoch 1. The standard-label higher-step runs used the augmented data with accumulation of one, giving 1,080 optimizer steps rather than 135, and selected epoch 3. All six answer-option orders were evaluated. The modest partial-encoder gains across two seeds came with worse order sensitivity and persistent missed unknowns. The neutral-label higher-step run selected epoch 2 and regressed against its own 12/36 baseline despite fewer false-supported results; 30/36 cases remained order-sensitive. None approaches rules/general-AI agreement on this synthetic set. Temperature-only calibration used 36 separate cases and remains fitted but unvalidated. Final-test inference and document extraction were not evaluated.

For the partial-encoder runs, peak PyTorch-allocated GPU memory ranged from **1,689,126,912 to 1,696,068,096 bytes (about 1.57–1.58 GiB)**; reserved memory peaked at **1,769,996,288 bytes (about 1.65 GiB)**. These allocator measurements are not total device usage. Sampled peak worker process-tree RSS was about **2.49–2.51 GiB**; per-run bytes and sampling limitations are in the reports. Each **100,002,244-byte** parameter delta requires its exact base checkpoint and calibration. The artifacts are not standalone models or installed plugin checkpoints, and this configuration fitting locally does not establish that a larger model will fit.

The rank-eight LoRA runs use alpha 16 across all 22 encoder layers plus the head, 360 training rows and accumulation of one. The three-epoch seed-17 run selected epoch 3 at 18/36. Separate six-epoch runs selected epoch 5 for seed 17 at **23/36**, and epoch 6 for seed 29 at **25/36**. The latter recognized 11/12 development unknowns, but four false-supported findings remained and answer-order sensitivity worsened to **15/36**, compared with 5/36 before training. Training is finished for this round: **eight GPU runs and two CPU head runs, totaling 36 epochs across all trials**, with no cloud GPU spend. The seed-29 candidate was selected on development results before the separate regression evaluation below; no further training used those regression results.

The LoRA runs peaked at **1,589,266,432 bytes (about 1.48 GiB)** allocated GPU memory; peak reserved memory was **1,669,332,992–1,671,430,144 bytes**. Sampled peak process-tree RSS was about **2.38–2.49 GiB**. Their **66,656,660-byte** adapter/head deltas require the original base and adapter configuration; none was promoted. See the [dry-run and training commands](backend-v2.md#reproduce-an-experimental-gpu-run) for reproduction.

A separate facts-only transformation was rejected because different split cases collapsed to the same model input; training on that transformation would compromise the comparison.

## Separate executable regression result

The [one-shot regression comparison](backend-adapter-regression-results.json) evaluated base Laya and the selected seed-29 adapter on the same 12 inputs and evidence. They cover four narrow local SQLite/process-crash/queue mechanisms; eight cases contain recorded measurements and four are missing-information controls. These cases were excluded from training and the final test set. The evaluator reads recorded traces; it does not rerun the underlying systems, select a checkpoint, fit calibration or supply hidden outcomes to the model.

| Regression measurement | Base Laya | Selected adapter |
|---|---|---|
| Correct labels | 4/12 | 4/12 |
| Missed defects | 4/4 | 2/4 |
| False-supported findings | 5 | 3 |
| Correct missing-information labels | 0/4 | 0/4 |

Fewer missed defects and false approvals did not improve overall accuracy: the adapter also rejected valid cases and still failed every unknown control. This is evidence against treating the development score as general architectural competence. The packet is itself small, local and not independently reviewed; it does not estimate production accuracy. The model remains research-only, the base checkpoint is unchanged, and the final 48-case test split remains untouched.

## Remaining work

A 5 TB cloud Drive allocation can store datasets, checkpoints and run artifacts. It does not add local laptop disk, RAM or GPU memory. The [Colab training notebook](../notebooks/archrivet_cloud_training.ipynb) prepares the cloud workflow with a user-selected Drive location. The attempted browser connection was unavailable, so **Colab was not started and Drive connection remains pending**. Run the notebook's disk/VRAM preflight and retain dataset hashes, split manifests and results alongside checkpoints. Repository examples use a folder placeholder rather than a private folder ID. Notebook preparation is not evidence that a cloud GPU job ran or training improved the model.

The CPU/rules workflow remains usable. Larger-model experiments, including an optional Qwen/local-generative comparison, have not run. They should follow a stronger evaluation set rather than assume model size fixes the observed errors. No experimental checkpoint replaces the default model.

## Stronger data for a solo project

The new [executable regression packet](../datasets/executable-backend-v1/README.md) contains **12 cases**: eight measured local SQLite/queue runs and four missing-information controls. These cases are kept outside training and the final test set. They exercise four narrow local mechanisms; only the durability scenario uses an actual abrupt process exit. They do not reproduce power loss, host/zone failure, distributed transactions or concurrency at production scale. This supplies measured regression evidence without turning it into an independent production-architecture benchmark.

Expand from failure mechanisms and requirement boundaries rather than trying to collect every company's complete architecture. Public engineering articles describe dated subsystem snapshots, and tutorials are often hypothetical designs.

1. Collect permitted primary engineering references with source URL, publication/access date, license and precise applicability conditions. Keep original summaries separate from quoted passages and synthetic cases.
2. Build cases around distinct mechanisms: concurrent duplicate delivery, deduplication expiry, lost commit acknowledgements, replica lag during failover, invalidation failures, writer fencing and nested retry budgets. Change one decisive fact at a time and include genuinely incomplete cases.
3. Add executable checks where feasible: crash after commit and count effects; delay replication and inspect reads; interrupt a multi-write operation; resume an old writer after promotion. Record what the experiment actually establishes and what remains an assumption.
4. Split by source and scenario mechanism before generating variants, so training and evaluation cover different mechanisms instead of new wording of the same templates. Detect duplicate and near-duplicate content across splits, not just repeated IDs. Keep evaluation questions and labels out of evidence retrieval.
5. Compare rules, original Laya, a specialized candidate and a general AI using the same allowed evidence. Report per-check errors, critical false support, abstention, prompt/option-order stability and calibration. Select candidates on development data and reserve the final test set.
6. Keep automatic labels explicitly automatic. Executable outcomes and independently sourced counterexamples can strengthen verification without hiring an engineer, but agreement with the generating model or rules is not independent accuracy. Seek independent review before claiming expert-validated labels or general architecture accuracy.

## Host validation boundary

The earlier v1 Codex trial preserved **9/9 composite plugin verdicts** across three known fixtures after separate compact tool outputs were used. Base Laya itself approved all nine checks; rules preserved contradictions and missing information. Those results do not validate new v2 job tools or two real hosts.

An additional [actual v2 Codex host trial](backend-v2-host-validation.json) now validates the new job path on the supplied eight-check fixture. Codex CLI 0.162.1 called `review_design` once and `get_review` once, preserved the input facts and all eight scope IDs, and retained **8/8 returned verdicts** in its final answer. It kept the missing cache facts, unknown capacity and deployment-verification limitations visible, and correctly reported that no local model inference occurred. The host trial took **56.062 seconds** against an isolated rules-only daemon; that daemon exited afterward. No shared runtime, global host configuration or model checkpoint was changed. This tests one real host and one structured fixture, not Laya accuracy, generative extraction quality or production safety.

The new focused v2 checks passed **2 subprocess integration tests in 9.44 seconds** on the development machine. Two MCP protocol clients shared an independently started rules-only service, submitted and polled jobs, and invoked cancellation on completed work. The structured fixture returned the eight expected verdicts; a prose-only request preserved unprocessed documents and unknowns. The tests also checked authenticated schema access, an unknown cancellation ID and service survival after client shutdown. A separate local CLI run of [the public example](../examples/backend-v2-input.json) returned three supported, four contradicted and one insufficient-information verdict, with capacity unknown. No model was loaded by these checks.

The completed local regression suite passed **200 tests in 41.19 seconds**, and Ruff passed. Passing software tests does not validate the trained adapter's judgments.

The protocol test also calls `get_review_schema`, verifies all eight definitions and the typed `design.retry.dedup_atomic_with_effect` fact path. Schema discovery and cancellation were not exercised by the actual v2 AI-host trial, whose supplied exact request needed only submission and polling.

The current local availability check found Claude CLI installed but unauthenticated; Gemini and OpenCode were not installed. No second-host login or installation was performed. Two MCP protocol clients are useful transport tests but are not two AI assistants. Both actual AI-host trials used Codex; no second real AI host has been validated.

Document extraction through a local generative backend, maximum-size document handling, plugin-serving performance on GPU and end-to-end architecture improvement remain experimental until measured on their actual workloads.
