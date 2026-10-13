# Collect web references and fine-tune Laya by yourself

The student is **Laya Multilingual: approximately 322 million parameters (0.322 billion)**. Its encoder has about 307 million parameters. The default experiment freezes that encoder and trains approximately 15 million parameters in the decision head and other non-encoder layers. Full mode also updates the encoder. The [upstream model card](https://huggingface.co/convaiinnovations/laya-multilingual) describes the model and its decision interface.

You do not need a backend engineer or human labeler to run this workflow. The generated labels are explicitly marked `machine_generated_rule_checked`. They are suitable for development experiments; agreement with them does not establish real production reliability.

Laya is a decision model. Raw articles alone are not its training format. This pipeline collects articles, uses an already authenticated AI host to produce source-linked hypothetical examples, and creates three variants of each example: a complete design, a design with one violating fact, and a design missing that fact. Training uses only this new web-derived export. It does not load the old pilot cases or runtime evidence collection.

## Run the three commands

Use the latest repository checkout, Python 3.11/3.12, and the model environment from the main README. Commands below use PowerShell. On Linux, replace `.\.venv\Scripts\python.exe` with your environment's `python` and use `/` in paths.

**1. Collect the configured reference pages.**

```powershell
.\.venv\Scripts\python.exe scripts\scrape_system_design.py --output .runtime\web-corpus
```

The starting sources are The Architecture of Open Source Applications, volumes 1 and 2, and a pinned revision of System Design Primer. They cover storage, distributed systems, caching, replication, queues, concurrency, networking, and example product designs. Historical descriptions and interview exercises are not verified descriptions of a company's current production architecture.

The scraper preserves paragraphs and code text, removes HTML navigation/scripts, records authors and source licenses, groups exact duplicate documents, and writes searchable chunks. It honors robots.txt, source/path boundaries, download limits, page limits and request delays. Redirects are reported rather than followed outside the configured scope. The default limit is 100 attempted pages with link depth 2, not an unrestricted internet crawl. The cache avoids repeated downloads.

Outputs are `documents.jsonl`, `chunks.jsonl`, `licenses.json`, `source-config.json`, `crawl-report.json` and a hash manifest. Read the crawl report to see failed or skipped pages. General concepts remain in the reference corpus even when unsuitable for today's three decisions.

**2. Turn relevant passages into decision examples.**

```powershell
.\.venv\Scripts\python.exe scripts\prepare_web_training.py --corpus .runtime\web-corpus --output .runtime\web-training --max-documents 24
```

This uses the installed, already authenticated Codex CLI as a temporary data-generation teacher. Pass `--codex "C:\path\to\codex.exe"` if it is not on PATH. You can select the teacher with `--model MODEL_ID`; otherwise the host default is recorded as unspecified. Laya remains the student. The teacher is used during data preparation, not for local Laya inference. Calls use your existing account and its usage limits; this script does not provision paid compute or install/sign in to a host for you.

Use `--max-documents 0` to consider every relevant collected document. Candidate retrieval is initially keyword based and selects one passage/check per document. It does **not** turn every paragraph into a training example. Unrelated passages are rejected, and the rejection reasons are retained. Increase scope after inspecting the first run. To use another teacher, use `--prepare-only`, generate families in the schema returned by `arch_review.web_training.response_schema()`, and pass their JSONL using `--responses` with a new output directory.

Document groups and exact shared passages are assigned to train, development, calibration and test before examples are generated. Quote matching, schema validation, one-field changes, and agreement with explicit decision rules are automatic gates. Semantic correctness of a lesson is not independently verified. The initial variants change atomic duplicate protection, durable-commit acknowledgement ordering, or preservation of accepted messages. These are deliberately narrow exercises, not every failure mode in the plugin.

If generation is interrupted, rerun the same command with `--resume` to reuse completed compatible batches. Completed exports are immutable; choose a new output directory to regenerate them. `export/rejections.json` explains rejected families. If screening leaves a split empty, the export fails instead of silently moving examples between splits; collect more relevant sources and create a new export.

**3. Fine-tune and compare against base Laya.**

```powershell
.\.venv\Scripts\python.exe scripts\finetune_web.py --data .runtime\web-training\export --base models\laya-multilingual --output models\web-head-v1 --epochs 1
```

Use your actual existing model directory for `--base`; there is no reason to download it again. `--dry-run` validates the export and tokenizer context lengths without training. The default CPU run has a 30-minute timeout; change `--timeout-seconds` for a larger dataset. It checks disk space before writing and refuses to overwrite an existing artifact. The base model and encoder are verified unchanged. The process uses no paid GPU service.

The output is `experimental-head.safetensors` (about 60 MB) plus `report.json`. The report includes base-model and trained-model development predictions, accuracy, missed defects/false support, calibration diagnostics, and answer-order sensitivity. Calibration is fitted on its own split; the final test split does not choose weights or temperatures and is not evaluated during training. Small or incomplete per-decision splits make these diagnostics unstable.

The head file requires the exact base model and is **not** a standalone plugin checkpoint. It is an economical way to test whether training helps. For a standalone candidate, run the same command with `--mode full` and a new output path; use `--device cuda` only in an environment with a working CUDA PyTorch installation. Full mode needs substantially more training memory and enough free storage for a complete checkpoint. Its runtime cost was not measured in the local smoke test. A full candidate can be selected with `arch-review configure --checkpoint PATH` after stopping the service; its experimental/unvalidated status remains visible.

## Add another website

Edit `configs/web_sources.json` or provide another file with `--config`. Each collection needs exact HTTPS seed URLs, an allowed host and path prefix, a collection-path regex, attribution, license URL, a permission marker in that page, and a license identifier. Only add sites whose terms permit the intended collection and reuse. A marker match is an automated consistency check, not a determination of legal permission. The crawler does not get around logins, blocks or paywalls.

GeeksforGeeks is not enabled: its [published terms](https://www.geeksforgeeks.org/legal/terms-of-use/) restrict systematic extraction without written permission. The two supplied collections have explicit attribution licenses. New source material keeps its own license and attribution; the repository's MIT license applies to the code.

To learn more decision types, define their input fields, labels and failure conditions first, then extend `CHECKS`, the variant builder and the consistency checks. Scraping more chapters alone will not turn a three-decision classifier into a general architecture designer.

## Actual first run

The tested run collected **57 references and 598 chunks**. A 24-document generation pilot accepted **10 families / 30 cases**, rejecting 14 irrelevant candidates. Splits contain 18 training, 6 development, 3 calibration and 3 untouched test cases. No previous pilot examples were included. One CPU training/evaluation epoch took about 113 seconds after model loading; development label agreement stayed at **2/6 before and after**. See [the machine-readable run report](web-training-results.json) for detailed results and limitations. The [exact corpus and export](../datasets/web-solo-v1/README.md) are included, so you can train without rerunning scraping or the teacher. This is a working pipeline and a small experiment, not a validated specialist.
