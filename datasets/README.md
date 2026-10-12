# Architecture reference and decision datasets

This collection has two distinct uses: human-origin reference text for checking claims, and AI-authored scenarios for developing the review model. It does **not** contain independently human-reviewed decision labels.

| Collection | Contents | Appropriate use |
|---|---|---|
| `open-reference-v1` | 18 pinned documents, 372 attributed text chunks | Retrieval, architecture study, source verification |
| `source-linked-v1` | 15 draft lessons and 45 hypothetical decision cases | Experimental training and evaluation; human labeling queue |
| Existing `src/arch_review/data/pilot.jsonl` | 100 synthetic cases | Earlier narrow decision pilot |
| `source-screening.json` | Accepted, rejected, and quarantined candidates | Audit why a source was or was not used |

The references include HDFS, Riak, Graphite, nginx, ZeroMQ, Twisted, MediaWiki, and general distributed-system chapters. The System Design Primer contributes concepts and eight exercises: Twitter-like feeds, Mint-like transaction processing, pastebin, query caching, sales rankings, AWS scaling, social graphs, and web crawling. Exercises describe proposed interview solutions; they are **not verified internal architectures of those companies**.

Book chapters describe historical implementations, principally from 2011–2012. Their principles can be useful, but their product details and reported scale must not be presented as current configuration advice or benchmarks for a new deployment. Source records retain original-work year separately from retrieval date and pinned revision. Named authorship is documented; every later contributor edit has not been audited for AI use.

## Licenses and provenance

The reference text is **CC BY 3.0 or CC BY 4.0**, according to each record. It is not relicensed under this repository's MIT software license. Original license notices are in `open-reference-v1/licenses/`. Every chunk carries authors, title, source URL, exact revision, license, content hash, and character offsets. HTML extraction and normalization are declared transformations. Images and linked third-party content were not downloaded.

Source-linked adaptations carry attribution and the corresponding source license. Their new scenarios, numeric assumptions, and verdicts were authored by AI and are not endorsed by the source authors. A quoted engineering lesson is background evidence, never proof that a proposed deployment satisfies a requirement.

## Reproduce collection and preparation

From a repository checkout with the package installed:

```powershell
python scripts/collect_open_corpus.py
python scripts/build_source_cases.py
python scripts/export_reviewer_packet.py --source datasets/source-linked-v1/cases.jsonl --output datasets/source-linked-v1/reviewer-packet.csv
arch-review dataset audit --source datasets/source-linked-v1/cases.jsonl
arch-review dataset export --source datasets/source-linked-v1/cases.jsonl --output training/reviewed --allow-unreviewed
```

The collector follows a small fixed allowlist, verifies pinned license text, caps document size, caches downloads, and never follows links. It does not scrape GeeksforGeeks or download unlicensed company articles. Collection uses the public raw GitHub distribution of the licensed works. The resulting source corpus is committed so evaluation does not require networking.

`--allow-unreviewed` explicitly marks an experiment. It does not certify labels. Before a quality release, give the blind CSV and [decision specification](../docs/product-specification.md) to backend engineers. Ask them to independently label cases, record unclear facts, and supply their name/date. Resolve disagreements before importing labels. Blank reviewer fields must remain blank until someone actually reviews them.

## Split and evaluation rules

The 45 new cases contain 15 scenario families with three variants each. Their split is 18 train, 9 development, 9 calibration, and 9 test. Each source document and all variants of its scenario remain within one split. The three new labels are balanced overall. Source grouping alone does not eliminate conceptual similarity: these are deliberately related toy cases of three narrow checks, not an independent generalization benchmark.

Combined with the original pilot, there are 145 cases: 78 train, 24 development, 19 calibration, and 24 test. The test split was not evaluated in this experiment. Evaluation labels and rationales are not indexed for retrieval or sent to the general-AI baseline. Optional background lessons are identical across each scenario's variants. The exporter records both evidence and source-case hashes.

Do not train directly on every retrieved paragraph as though it were a correct architecture verdict. Convert relevant mechanisms into requirements, explicit assumptions, a failure timeline, counterfactual variants, a label, source applicability, and required verification tests. Keep workload arithmetic in code. Maintain missing-information cases and outdated-source flags as the collection grows.

See [the experiment report](../docs/research-validation-2026-10-13.md) for measured results and limitations.
