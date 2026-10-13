---
license: apache-2.0
base_model: convaiinnovations/laya-multilingual
library_name: laya
language:
  - en
tags:
  - architecture-review
  - experimental
  - merged-lora
---

# ArchRivet 322M — experimental merged model

A complete local decision classifier for backend architecture experiments. The base
Laya weights, trained decision head, merged adapters, tokenizer, encoder configuration
and fitted calibration are included. **No separate base-model or adapter download is
needed.** Python runtime dependencies are installed separately.

**Experimental: 25/36 correct on synthetic development cases, 4/12 on separate
executable-regression cases.** This is not validated as a general system-design expert
or a production approval mechanism. It returns choices and probabilities, not generated
architecture explanations. ArchRivet's rules remain the default; using these weights
is an explicit opt-in.

[Download the full model](https://github.com/Ujjansh05/archrivet/releases/tag/model-v0.1.0)
· [Plugin release](https://github.com/Ujjansh05/archrivet/releases/tag/v0.2.0a3)
· [Measured results](https://github.com/Ujjansh05/archrivet/blob/main/docs/model-improvement-status.md)

## Install and connect your AI

Use Python 3.11 or 3.12 in a virtual environment. Activate it with
`.\.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on Linux.
The project is not on PyPI; the URL below installs the GitHub wheel.

```bash
python -m venv .venv
# Activate the environment using the command above for your operating system.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "architecture-review-plugin[model] @ https://github.com/Ujjansh05/archrivet/releases/download/v0.2.0a3/architecture_review_plugin-0.2.0a3-py3-none-any.whl"
arch-review download-model --variant archrivet-experimental --destination models/archrivet-322m
arch-review configure --checkpoint models/archrivet-322m --backend laya --device cpu --context-tokens 1024
arch-review start
arch-review mcp-config
```

If an ArchRivet service is already running, run `arch-review stop` before `configure`.
Keep the same `ARCH_REVIEW_HOME` across commands. Copy the generated MCP configuration
into your AI host. Windows requires `start` in an independent terminal before connecting
the host. The host calls `get_review_schema`, then `review_design` and `get_review`.
Rules, evidence and model judgments remain separate; model approvals cannot erase
explicit rule contradictions.

The installer streams the download, checks a pinned SHA256 and file size, validates
archive paths and individual files, then installs the complete checkpoint. It refuses
to overwrite an existing directory and does not change your settings. For manual
installation, download the ZIP and `SHA256SUMS.txt` from the model release, verify the
archive and extract it into a new folder. Point `configure` at the folder containing
`rl_agent_config.json`.

The FP32 weight file is **1,287,653,752 bytes (1.29 GB)**. Allow several GB for the
temporary archive, extracted model and Python dependencies. CPU inference is supported;
CUDA is optional with a compatible PyTorch installation. Validation used Windows,
Python 3.12, Torch 2.6.0 CPU and CUDA 12.4, and an RTX 3050 6 GB. Runtime memory depends
on input length and concurrency; this does not guarantee fit on every low-memory machine.

## Use directly in Python

After installing and downloading as above:

```python
from arch_review.checkpoint import configure_checkpoint_loading

configure_checkpoint_loading()  # Windows-safe loading; a no-op elsewhere.
import laya
import torch

agent = laya.load("models/archrivet-322m", device="cpu")
agent.amp_enabled = False
agent.dtype = torch.float32  # Same precision as validation and calibration.
result = agent.predict(
    "A worker commits a database write, then crashes before acknowledging the queue. "
    "The queue redelivers the message. No uniqueness constraint or deduplication exists.",
    {"retry_safety": {
        "type": "choice",
        "instructions": "Does the design prevent repeated delivery from duplicating business effects?",
        "criteria": {
            "supported": "Explicit facts satisfy the requirement.",
            "contradicted": "Explicit facts establish a violation.",
            "insufficient_information": "Required facts are missing or conflicting."
        }
    }},
    max_len=1024,
    head_max_len=256,
)
print(result["answers"])
```

Direct calls expose model judgments without the plugin's rules or workload calculations.
Inspect `result["usage"]` for truncation; do not rely on a truncated answer. Tested context
is 1,024 tokens per decision, including a 256-token question head. Larger document intake
does not enlarge this model window. Direct CUDA callers must also disable autocast as
shown above. The 0.2.0a2 plugin applies the model manifest's FP32 setting automatically.

## Training, merge and limitations

- Base: `convaiinnovations/laya-multilingual`, revision
  `1720e3e3357cfe1e281542e223f8273b0890ca34`, **321,908,995 parameters**.
- Candidate: `backend-lora-cuda-6ep-seed29`, six epochs, selected epoch six on development
  data. Rank-eight adapters, alpha 16, across 88 linear layers in all 22 encoder layers,
  plus the trained decision head.
- Training: 120 constructed training cases plus 240 representation variants. Variants
  are not new independently verified cases. Eight decisions cover retries, acknowledgements,
  queue overload, atomicity, replica consistency, cache freshness, failover and dependency
  resilience. Specialized examples are English; multilingual accuracy has not been evaluated.
- Merge: `W = W_base + (16 / 8) * B @ A`, exported in FP32 with ordinary Laya tensor names.
  No adapter wrappers or external encoder-weight downloads are needed.
- Calibration: fitted on 36 separate synthetic calibration cases; still unvalidated.
  Model probability is not the probability that a production system will fail.

No new training was performed for this release. Original base and adapter files were
preserved. Publishing makes the candidate accessible; it does not promote its quality.
The unchanged 4/12 separate regression accuracy and 0/4 correct unknown controls remain
limitations. Changing option order changed answers on 15/36 development cases. The final
48-case test remains unused. Maximum-context and arbitrary prose extraction are unvalidated.

## Merge verification

The unmerged candidate and merged checkpoint returned identical labels on 48 existing
inputs: 36 development and 12 executable-regression cases. Fresh CPU and CUDA processes
preserved all 48 labels; maximum reported probability drift was approximately 0.0001.
These verify packaging fidelity, not additional independent accuracy. See the
[machine-readable validation](https://github.com/Ujjansh05/archrivet/blob/main/docs/merged-model-validation.json).

Reproduction requires the exact base, adapter and original report. Run
`scripts/merge_laya_adapter.py --help` from the repository. Export into a new directory,
then run `--verify-only` in fresh CPU and CUDA processes. Retain licenses and checksums
when redistributing. License/model-card packaging is separate from the tensor merge.

## License and attribution

The model is distributed under **Apache License 2.0**, following
[Laya's declaration](https://huggingface.co/convaiinnovations/laya-multilingual).
Its mmBERT backbone is [declared MIT upstream](https://huggingface.co/jhu-clsp/mmBERT-base).
The archive contains `LICENSE`, `NOTICE`, upstream model cards and license text.
The plugin's separate MIT software license does not replace these model terms.
This is an ArchRivet adaptation; upstream authors do not endorse its judgments.
