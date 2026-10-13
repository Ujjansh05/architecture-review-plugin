# Training and evaluation handoff

The [October 13 research experiment](research-validation-2026-10-13.md) adds 45 source-linked cases and a real CPU head-only training run. It did not improve development accuracy. The production checkpoint remains the unmodified base model. The original full-checkpoint training workflow below remains available; the new diagnostic is a separate, explicitly experimental path.

To reproduce the combined preparation, run `python scripts/prepare_research_experiment.py`. Its output has 78 train, 24 development, 19 calibration, and 24 untouched test cases, all with draft labels. Use `scripts/run_codex_baseline.py` for blind development predictions and `scripts/train_head_diagnostic.py` for the bounded head-only experiment. The latter saves approximately 60 MB of non-encoder weights and verifies that the original checkpoint and frozen encoder are unchanged. Its artifact is not a standalone plugin checkpoint and is not release-approved.

## Current artifacts

The shipped dataset is a synthetic pilot, not expert-reviewed training truth. It has 20 scenario families and 100 cases, with 60/15/10/15 cases assigned to train/development/calibration/test. Variants stay with their families. Ten known development regressions are separate and never count as independent final-test evidence.

Company case studies are source references; they are not gold architecture labels. Independently authored company/subsystem holdouts and the proposed expansion toward 1,000 reviewed examples remain outstanding. The two proposed company holdouts in the manifest are a curation task, not an implemented evaluation claim.

## Reviewer workflow

Review `docs/product-specification.md` first. For each case in a copy of `src/arch_review/data/pilot.jsonl`, check the requirement, explicit design facts, expected verdict, rationale, applicability and variant pairing. Resolve ambiguity by editing the case. Set `review_status` to `expert_reviewed` only after a real reviewer records their name and `reviewed_at`. Review source summaries independently; their status is recorded separately.

Group paraphrases, incident variants and reconstructions of the same company subsystem together before freezing a release dataset. Keep evaluation labels out of retrieval. Do not simply mark the existing pilot as an independently commissioned final test.

Run `arch-review dataset export --source REVIEWED.jsonl --output training/reviewed`. The export contains explicit split files, fixed model inputs, evidence version, family manifest and SHA-256 hashes. Re-export after intentional data changes; training rejects modified files with stale hashes.

## Baselines

Run rules and base Laya on the same development cases. For a general-AI comparison, use `general_ai_inputs.jsonl` and collect one JSONL record per selected case:

```json
{"id":"CASE-ID","answers":{"retry_safety":"insufficient_information"}}
```

Optional probabilities are keyed by check ID and then by all three labels. Import with `arch-review evaluate --source REVIEWED.jsonl --split development --predictions predictions.jsonl --output reports/general-ai.json`. Missing, duplicate or unexpected case IDs are rejected.

Use `--backend laya --checkpoint PATH --perturb` to compare original and reversed answer orders. The dataset contains paraphrase and naming variants. Add independently reviewed rewordings that preserve meaning when extending the corpus. Check errors by failure mode, rather than relying solely on overall accuracy.

## Fine-tuning

Use the pinned Laya 0.4.6 training API with a 1024-token sequence, 256-token head, soft cross-entropy and shuffled choices. The runner preserves the explicit splits; it does not use Laya's automatic row-level calibration partition.

```powershell
arch-review train --data training/reviewed --base models/laya-multilingual --output models/specialist-seed0 --device cuda --epochs 4 --seed 0 --timeout-seconds 3600 --hourly-rate-usd CURRENT_RATE
```

Development accuracy selects the epoch. Only the calibration split fits temperatures after selection. Calibration is recorded as fitted_unvalidated; it is not automatically promoted to validated. The final-test split never selects an epoch or temperature. The runner rejects inputs exceeding its conservative 760-token state budget.

For an explicit implementation experiment on draft labels, add `--allow-unreviewed`. The resulting checkpoint is marked experimental_specialized. `--freeze-encoder` is available for a small diagnostic training run and must not be reported as equivalent to a full fine-tune.

## GPU budget and external provider

No paid job has been launched. This project does not silently upload local files, infer billing credentials, or publish model weights. A funded GPU environment must be supplied to execute the paid experiment.

Use one 16 GB T4-class worker initially. Check the provider's live rate before launching. Stage the package, exported dataset and base model in that environment; run the command above there. Set the provider's own hard job timeout to cover installation, download, training and evaluation.

The local runner reserves the ceiling cost of its training timeout at the supplied hourly rate, rounded to billed minutes and cents. Reservations are cumulative in `training-budget.json`. They may total at most USD 10; USD 5 remains reserved within the owner's current USD 15 cap, including for older ledgers created under a higher limit. Reservations are not refunded automatically. Actual provider charges remain unknown until reconciled with billing.

Stopping a training subprocess does not terminate a provider VM. The provider job timeout is therefore required for the overall cap. Include setup time, evaluation time, storage, transfer and taxes in the reservation before launching; shorten the training allocation if necessary. Never start a job if total maximum charges cannot fit the cap.

## Quality report

`arch-review evaluate` reports per-check precision, recall, macro-F1, false approvals, critical false approvals, multiclass Brier score and ten-bin ECE where probabilities exist. Inference errors count as failed predictions, not missing-information successes. Missing labels in a split prevent a numeric gate from passing.

Pilot targets are macro-F1 >= 0.85 per check, contradiction recall >= 0.90, insufficient-information recall >= 0.90, and no false supported judgments on critical regressions. Compare with the base model and general AI. State sample counts and uncertainty; small pilot scores are not reliable production guarantees.

The evaluator does not automatically approve a quality release. A release also needs independent labels, source applicability review, improvement over the base model without increased false approvals, two actual AI-host trials, and a blind engineer comparison of usefulness.
