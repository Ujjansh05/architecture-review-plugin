# Published model vs. October preview

The new optional preview got **769/780 (98.6%)** synthetic final-test decisions right. The published model got **294/780 (37.7%)**. Incorrect approvals fell from **222 to 3**.

**This is an experimental improvement, not production certification.** Strict gates remain unmet. Rules stay the default.

## Rules baseline

The deterministic rules score **780/780**, with **zero incorrect approvals**, on these same public synthetic inputs. The evaluation maps the model formatter's cache field aliases back to the runtime schema. It does not include prose extraction or a host assistant.

These finite scenarios closely match the rules. This baseline establishes neither perfect real-world performance nor a benefit from adding Laya. The 780 inputs are now a regression suite; future training needs a separately frozen test. [Per-case rules results](2026-10-rules-baseline.json).

```powershell
python scripts/benchmark_rules_baseline.py --output rules-results.json
```

The 0.3.0a2 runtime adds conservative fact verification and new packaging; it does not change the weights behind the model results below. See [full-review runtime measurements](2026-10-runtime.md) for separately scoped performance data.

## Same-case comparison

| Split | Cases | Published correct | Preview correct | Published wrong approvals | Preview wrong approvals |
|---|---:|---:|---:|---:|---:|
| development | 420 | 163 (38.8%) | 411 (97.9%) | 127 | 5 |
| regressions | 48 | 23 (47.9%) | 47 (97.9%) | 13 | 1 |
| final-test | 780 | 294 (37.7%) | 769 (98.6%) | 222 | 3 |

An incorrect approval means the model answered `supported` when the case was flawed or missing necessary information. The table measures the **model decision**, not the combined rules-and-model review.

## Final test by check

| Check | Cases | Published accuracy | Preview accuracy | Preview wrong approvals | Verdict changes across six answer orders |
|---|---:|---:|---:|---:|---:|
| retry_safety | 210 | 34.3% | 96.7% | 3 | 8/210 |
| ack_durability | 60 | 33.3% | 100.0% | 0 | 0/60 |
| queue_overload | 60 | 26.7% | 100.0% | 0 | 0/60 |
| transaction_atomicity | 60 | 23.3% | 100.0% | 0 | 0/60 |
| replica_consistency | 60 | 43.3% | 100.0% | 0 | 0/60 |
| cache_freshness | 150 | 36.7% | 97.3% | 0 | 7/150 |
| failover_safety | 120 | 60.0% | 100.0% | 0 | 0/120 |
| dependency_resilience | 60 | 31.7% | 100.0% | 0 | 0/60 |

## What still fails

- Three unsafe retry cases were incorrectly approved on the final test. Retry safety therefore fails its strict final-test gate.
- Eight other final-test errors rejected valid designs or treated a missing fact as a contradiction. All 11 errors are in the [per-case report](2026-10-preview-final-test.json).
- The development gates still fail for retry safety and cache freshness. The 48-case regression set contains one incorrect approval: `product_stale_on_error-incomplete`. An unspecified outage-age bound remains unknown in the **combined review**, because the model cannot erase the rule finding.
- Across answer order permutations, 15/780 final-test decisions changed at least once. The canonical accuracy alone does not describe this instability.
- In the actual packaged CPU test, the model incorrectly approved a cache allowing 60-second-old values against a five-second requirement. The combined review correctly retained `contradicted` and the rule/model conflict. This practical failure is another reason to keep the model optional; the GIF uses rules mode.

## How the comparison was run

The published `model-v0.1.0` weights use the original v0.2.0a3 facts/evidence serialization and 1,024-token limit. The new checkpoint uses its bound `archrivet-facts-2` formatter and 2,048-token limit. Both receive the same declared facts and evidence. The new formatter also retains scenario context and names unknown fields; the old deployed formatter omitted context. **This compares two deployment configurations, not the isolated effect of fine-tuning.**

Both use Laya 0.4.6, native `predict`, FP32, four CPU threads and an NVIDIA RTX 3050 Laptop GPU. All six answer orders were scored for every case. Tables use each deployment format's canonical order. There were no failed inference calls. All cases fit their respective contexts.

The checkpoint was fixed before final-test inference: seed 17, selected update 1,350. It was chosen for 47/48 development regressions rather than seed 29's 43/48. Development results were used during selection and are not an untouched estimate. The final test contains 780 balanced labels across eight checks, with more cases for targeted retry/cache/failover mechanisms.

Training used synthetic finite scenarios. Split-specific configurations share simulator mechanisms; 19 fact-only patterns cross split roles even though complete model inputs do not. There are zero independently reviewed real-world scenario families. A high score here does not establish accuracy on arbitrary architectures, unseen companies, or prose extraction.

Temperature was fitted on a separate calibration split and checked on another synthetic split. The final-test ECE is 0.0090; this is not independent confidence calibration or a production failure probability.

## Reproduce the final-test decision counts

The [780 final-test inputs and labels](2026-10-final-test.jsonl), [summary and hashes](2026-10-preview.json), [published predictions](2026-10-published-final-test.json), and [preview predictions](2026-10-preview-final-test.json) are public. Training data, simulator implementation and collection work are not required for replaying inference.

Install the 0.3 runtime with model dependencies, download both checkpoints, and run from the repository:

```powershell
python scripts/reproduce_preview_benchmark.py --checkpoint "C:\Models\old" --format published --device cpu --output old-results.json
python scripts/reproduce_preview_benchmark.py --checkpoint "C:\Models\new" --format preview --device cpu --output new-results.json
```

CUDA is optional if your PyTorch installation supports it. Hardware/library floating-point differences can affect borderline choices. Published test inputs must be retired as an untouched benchmark for future versions.

## Response time and memory

A separate small CPU test used 16 fixed decisions per version, one excluded warmup, FP32, four threads and concurrency one on an Intel Core i7-13700H Windows laptop. These are native classifier calls, not whole reviews or AI-host response time.

| Version | Median decision time | Observed min–max | Sampled process RAM |
|---|---:|---:|---:|
| published | 0.178 s | 0.141–0.237 s | 1.62 GiB |
| preview | 0.299 s | 0.253–0.428 s | 1.71 GiB |

This is one local pass, not a production throughput study. RAM includes Python/libraries and allocator retention. See [raw timing samples](2026-10-cpu-latency.json). The first model load is excluded from these warm timings and can take roughly a minute or more on this laptop. In a separate packaged MCP check, the cold review took 105 seconds and the following two reviews about three seconds each; these are observations, not a controlled throughput benchmark. A more accurate model does not necessarily answer faster. ArchRivet's own speed says nothing about the capacity of the architecture being reviewed.
