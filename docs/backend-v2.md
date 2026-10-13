# Backend reviews in 0.2.0a1

The additive v2 interface checks eight specific backend decisions. Start with structured facts and the rules backend; add local model inference when you need an experimental second judgment or document extraction. The existing `review_architecture` MCP tool, `review` CLI command and `/v1/reviews` API keep their three-check contracts.

The example is hypothetical. Neither its company-like workload nor a familiar database or queue name establishes a production guarantee.

## Install and run one review

From a checkout of `main`, use Python 3.11 or 3.12. PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
$env:ARCH_REVIEW_HOME = "$PWD\.runtime"
.\.venv\Scripts\python.exe -m arch_review configure --backend rules
.\.venv\Scripts\python.exe -m arch_review review-design examples\backend-v2-input.json --local
```

If a service already uses this runtime directory, stop it before `configure`, then restart it afterward. `--local` runs synchronously in the CLI process; it does not create a shared review job. Rules mode needs no model download, GPU or AI account.

The example produces three supported checks, four contradictions and one missing-information result with rules alone. Capacity remains unknown without a matching benchmark. Its cache bound is deliberately unspecified, so the overall review is partial.

Print the exact current schema and field definitions:

```powershell
.\.venv\Scripts\python.exe -m arch_review schema-v2 --output reports\schema-v2.json
```

## The eight decisions

| ID | What is checked | Representative counterexample |
|---|---|---|
| `retry_safety` | Retries do not repeat protected business effects throughout the retry window. | A worker commits an effect, crashes before acknowledgement, then repeats it on redelivery. |
| `ack_durability` | The exact acknowledged operation survives every required failure domain. | Success precedes durable commit, or a host-loss requirement exceeds the declared guarantee. |
| `queue_overload` | Queue and admission policies satisfy declared bounds and preserve accepted work. | An unbounded queue is proposed despite a bounded-queue requirement. |
| `transaction_atomicity` | A declared atomic-visibility invariant spans all writes and concurrent operations. | Another reader can observe a partial multi-write update. A saga alone does not establish atomic visibility. |
| `replica_consistency` | Permitted read routes enforce required read-your-writes or linearizable reads. | A client reads a lagging replica without the required read guarantee. Eventual convergence is outside this check. |
| `cache_freshness` | Declared worst-case staleness, including failure paths, stays within the allowed age. | Serving stale data on source failure can exceed the requirement. A TTL alone may leave relevant paths unspecified. |
| `failover_safety` | An old writer cannot commit after a successor is promoted when single-writer ownership is required. | A partitioned leader continues writing because storage does not enforce ownership fencing. |
| `dependency_resilience` | One serial dependency path's attempts, timeouts, backoff and local work fit the deadline. | Attempts × timeout + total backoff + local work exceeds the deadline without propagated cancellation. |

`supported` means the explicit facts satisfy that narrow check; `contradicted` means explicit facts expose a violation; `insufficient_information` means the guarantee is unresolved. An explicitly inapplicable check has `evaluation_status: "not_evaluated"` and a null verdict. It is never counted as supported.

Use separate `scope_id` values for distinct flows. The first three check schemas reuse the v1 `requirements` and `design` fields inside `facts`. The other five use their own small fact schemas. Missing facts should be omitted or null. In rules/Laya mode, declare requirements inside these structured facts and omit the top-level prose `requirements` field. Nonempty prose requirements are also treated as a document: without a generative extractor they remain unprocessed and otherwise supported conclusions are withheld.

## Choose the local backend

| Backend | Structured checks | Document extraction | Model confidence |
|---|---|---|---|
| `rules` | Deterministic checks and calculations | Unavailable; unprocessed documents stay visible | None |
| `laya` | Rules plus local decision-model judgments | Unavailable; Laya does not generate extraction JSON | Returned probabilities are unvalidated unless the checkpoint has valid calibration metadata |
| `llama_cpp` | Rules plus local generative judgments | Experimental text/Markdown/JSON extraction with source spans | No numerical confidence is fabricated |

Configure Laya after installing the optional runtime and obtaining a checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[model]'
.\.venv\Scripts\python.exe -m arch_review download-model --destination models\laya-multilingual
.\.venv\Scripts\python.exe -m arch_review configure --backend laya --checkpoint models\laya-multilingual --device cpu --context-tokens 1024
```

Reuse existing weights instead of downloading them again. The current Laya Multilingual checkpoint is approximately 322 million parameters. Setting a larger context limit does not demonstrate that a checkpoint supports it or make its decisions reliable.

For an already running local llama.cpp server:

```powershell
.\.venv\Scripts\python.exe -m arch_review configure --backend llama_cpp --llama-endpoint http://127.0.0.1:8766 --context-tokens 4096
```

Alternatively, supply existing local executable and GGUF paths so ArchRivet can manage that server:

```powershell
.\.venv\Scripts\python.exe -m arch_review configure --backend llama_cpp --llama-server C:\models\llama-server.exe --gguf C:\models\compatible-instruct.gguf --llama-endpoint http://127.0.0.1:8766 --device auto --context-tokens 4096
```

These are placeholders, not downloads or a tested model recommendation. Use a model compatible with the adapter's ChatML prompt and llama.cpp JSON-schema output. Only a loopback HTTP endpoint is accepted; there is no cloud fallback. Managed `auto` startup attempts GPU loading, then CPU if startup fails. An externally started server controls its own device and memory use. Managed servers unload after the configured idle interval; ArchRivet does not stop an externally owned server. Starting the managed generative backend unloads the shared Laya worker. With an external llama.cpp server configured, a v1 request is rejected with `409` instead of loading a second model beside that external process. GPU inference and extraction quality still require measurement for the selected model.

## Shared jobs, CLI and MCP

Start the daemon independently, particularly on Windows, before connecting MCP clients:

```powershell
.\.venv\Scripts\python.exe -m arch_review start
.\.venv\Scripts\python.exe -m arch_review review-design examples\backend-v2-input.json --wait --output reports\backend-v2-job.json
.\.venv\Scripts\python.exe -m arch_review mcp-config
```

Omit `--wait` to return the job ID immediately. Poll or cancel that ID:

```powershell
.\.venv\Scripts\python.exe -m arch_review get-review YOUR_JOB_ID
.\.venv\Scripts\python.exe -m arch_review cancel-review YOUR_JOB_ID
```

`get-review` returns a job envelope; completed findings are in `result`. Statuses are `queued`, `running`, `cancelling`, `completed`, `failed` and `cancelled`. Cancellation and the default 30-minute overall review deadline take effect cooperatively between bounded calls; an in-flight model call can finish before they take effect. Completed or cancelled results expire after about an hour through background cleanup and can be evicted earlier when the bounded result store fills. Restarting the daemon loses jobs. Save useful results with CLI `--output`.

Call `get_review_schema` before constructing a new request to discover the eight check IDs, typed fact paths and input schema without reading local files or handling the HTTP token. The new MCP tools are:

```text
get_review_schema() -> decision definitions, exact fact paths and input schema
review_design(request) -> job ID and current status
get_review(job_id) -> progress, errors or result
cancel_review(job_id) -> cancellation status
```

The service has one shared admission limit of five active-or-waiting requests across v1 and v2, executes reviews serially and shares inference resources. Review resource measurements sample the owned process tree's RSS; they can count shared pages more than once and exclude an externally managed inference server. The authenticated loopback API mirrors those tools:

| Method | Route | Result |
|---|---|---|
| `GET` | `/v2/schema` | Input/output schemas and decision definitions |
| `POST` | `/v2/reviews` | `202` plus job envelope |
| `GET` | `/v2/reviews/{job_id}` | Current job envelope; `404` if unknown or expired |
| `DELETE` | `/v2/reviews/{job_id}` | Cancellation request result |

Use the local service token as a bearer token; never put it into shared examples or commits. Browser-origin requests are rejected. Ordinary architecture submissions are not written into the training dataset. The persistent SQLite index stores reference evidence, not review jobs or user designs.

## Review text, Markdown or JSON documents

Send document contents inside the v2 request. The tool does not open paths, fetch URLs, parse PDFs or follow instructions embedded in a design. `format` describes the supplied text; it does not turn a raw JSON architecture into verified facts.

```powershell
$request = @{
  requirements = 'A returned acknowledgement must survive host loss.'
  documents = @(@{
    id = 'design-notes'
    format = 'markdown'
    text = Get-Content .\my-design.md -Raw
  })
  requested_checks = @('ack_durability', 'failover_safety')
}
$request | ConvertTo-Json -Depth 20 | Set-Content .\design-request.json -Encoding utf8
.\.venv\Scripts\python.exe -m arch_review review-design .\design-request.json --wait
```

This extraction path needs `llama_cpp`. Rules/Laya mode reports those documents as unprocessed and preserves unknowns; it does not quietly pretend to read them.

**A 128,000-token document limit is not a 128,000-token model context.** The total serialized request is capped at **2 MiB**; generative extraction also rejects aggregate document-plus-requirement text above **128,000 tokenizer-counted tokens**. Documents are processed in overlapping chunks, one check at a time. The configured model call budget is normally **4,096 tokens** for llama.cpp, including prompt and reserved output; long schemas or requirements can still make a call too large. Such failures remain visible. There is no demonstrated guarantee of complete reasoning across a document of the maximum size.

Extracted facts must match the schema and cite an exact unique quote in the current section. Prose requirements are processed as a reserved `__requirements__` document and can be cited too. Distinct flows may inherit global requirement flags; they do not inherit another flow's implementation guarantees. Character spans preserve provenance, but a matching quote does not prove the model interpreted it correctly. Contradictory extracted values are withheld, missing facts remain unknown, and unprocessed sections prevent supported conclusions. Inspect `coverage`, extraction/inference errors and model disagreements before using findings.

The checked-in example and protocol tests exercise the software with supplied facts. They do not validate arbitrary prose extraction, a large GGUF model, production capacity or two real AI hosts. See [model improvement status](model-improvement-status.md) for measured evidence and remaining work.

## Training inputs and experiment status

The two completed CPU head experiments did not improve verdict accuracy: standard labels stayed at 11/36 development cases, and neutral labels stayed at 12/36. The neutral-label head retained 32/36 answer-order changes after three epochs. Temperature fitting on the separate calibration split did not resolve those errors, and neither head became the active checkpoint.

The [representation-augmented export](../datasets/backend-augmented-v1/README.md) has 360 training rows: 120 originals plus 240 alternate renderings, with 315 unique training inputs and zero new independent cases. Development/calibration/test remain 36/36/48, copied unchanged. It can be reproduced into a new directory:

```powershell
.\.venv\Scripts\python.exe scripts\augment_backend_training.py --output .runtime\backend-augmented-next
```

That command prepares data; it does not train or activate a model. All-layer LoRA reached **25/36 (69.44%) on synthetic development cases**, up from 11/36, but **remained at 4/12 (33.33%) on separate executable regression cases**, equal to the base model, with none of four unknowns identified. The development gain did not transfer to accuracy on those local traces. The adapter remains research-only and was not promoted. Training and the one-shot regression evaluation are finished; Colab remains unrun. Follow the [experiment status and measured reports](model-improvement-status.md) before selecting a checkpoint.

## Reproduce an experimental GPU run

Use an activated Python environment with the project's training dependencies and a compatible CUDA-enabled PyTorch already installed. The ordinary CPU environment does not gain CUDA support from `--device cuda`. The measured local runs used PyTorch 2.6.0 with CUDA 12.4; hardware and drivers still need to support your installed build. Replace the checkpoint path below with your existing local Laya checkpoint and choose an unused output directory.

First inspect the plan without loading Torch or downloading a model:

```powershell
python scripts\train_partial_encoder.py --data datasets\backend-augmented-v1 --base C:\models\laya-multilingual --output models\encoder-adapter-seed17 --encoder-adapter-rank 8 --grad-accum 1 --epochs 3 --seed 17 --device cuda --dry-run
```

Check `data_can_train`, `resources.can_run` and `gpu_resources.can_run`. This dry-run checks dataset isolation, checkpoint metadata and resource availability; it does not establish that Torch can initialize CUDA or that training will succeed. After the checks pass, run the same command without `--dry-run`:

```powershell
python scripts\train_partial_encoder.py --data datasets\backend-augmented-v1 --base C:\models\laya-multilingual --output models\encoder-adapter-seed17 --encoder-adapter-rank 8 --grad-accum 1 --epochs 3 --seed 17 --device cuda
```

This selects rank-eight encoder adapters across all 22 layers, with alpha 16, plus head training. The trainer accepts one to six epochs and defaults to three. To reproduce the longer run, use `--epochs 6` in both commands and a fresh output directory; `--seed 29` selects the replication seed. To reproduce the last-two-layer experiment instead, replace `--encoder-adapter-rank 8` with `--last-layers 2` and select a different output directory. Both use the existing checkpoint offline; the trainer does not download weights. The trainer defaults to CPU when `--device` is omitted. It writes an experimental parameter delta and evaluation/resource reports, leaves the base checkpoint unchanged, and does not activate the result in the plugin. An adapter delta requires the same adapter configuration, exact base checkpoint and reported calibration to reload; it is not a standalone model.

To compare a completed adapter experiment with base Laya on the frozen executable regression packet, first verify the inputs without loading the model:

```powershell
python scripts\evaluate_adapter_regressions.py --base C:\models\laya-multilingual --experiment models\encoder-adapter-seed17 --data datasets\executable-backend-v1 --output .runtime\adapter-regression-review --device cuda --dry-run
```

Then run the same evaluation without `--dry-run`:

```powershell
python scripts\evaluate_adapter_regressions.py --base C:\models\laya-multilingual --experiment models\encoder-adapter-seed17 --data datasets\executable-backend-v1 --output .runtime\adapter-regression-review --device cuda
```

Use your completed experiment directory containing `report.json` and `experimental-delta.safetensors`; choose a new output directory. For the published comparison, that experiment was the six-epoch seed-29 adapter. The evaluator checks dataset and checkpoint hashes, reuses the candidate's existing calibration, and evaluates base and candidate on identical recorded inputs. It neither trains nor reruns the SQLite/queue experiments, and it does not use the final test split. Keep this comparison separate from development-based checkpoint selection.
