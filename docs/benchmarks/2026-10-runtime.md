# Complete local review timing — 0.3.0a2

These measurements cover eight declared checks per review on one Windows laptop. They measure plugin work, including result rendering, serialization, queueing and polling. They exclude the host assistant's answer, package installation and weight downloading. They do not measure application capacity or accuracy.

Each mode used three fresh processes, one first review per process, then ten warm reviews per process (30 warm samples). Laya used the unchanged optional preview checkpoint, CPU FP32 and four threads. Rules ran under Python 3.12.4; Laya under Python 3.12.13. The processor was Intel Family 6 Model 186, the same i7-13700H laptop used previously. OS disk caches were not cleared.

| Measurement | Rules | Optional Laya CPU |
|---|---:|---:|
| Warm complete-review median | 33.0 ms | 2.387 s |
| Warm complete-review p95 (nearest rank) | 41.2 ms | 2.734 s |
| First review median, excluding earlier setup | 39.5 ms | 11.122 s |
| Application initialization through first result, median | 0.517 s | 12.378 s |
| Peak sampled process-tree RSS during review | 49.2 MiB | 2.78 GiB |

Application initialization starts at the benchmark child function, after interpreter and standard-library startup. It includes ArchRivet imports, backend setup, index creation and first review. The first-review timer starts after backend construction and index creation; lazy model loading remains inside that first Laya review. Model startup can vary substantially between runs and packaging environments.

RSS includes the review process and its descendants. Sampling can miss peaks and double-count shared pages. This differs from the older single-process classifier-only measurements; the two RAM figures are not directly comparable. CUDA allocator metrics are null in these CPU runs. A p95 from 30 fixed-input samples is descriptive, not a production latency guarantee.

No previous-runtime full-review benchmark was run under identical conditions, so these numbers do not establish a speedup over 0.3.0a1. Rules and Laya also perform different amounts of work; this is an operating-cost comparison, not an accuracy contest.

## Raw measurements and reproduction

- [Rules samples, stage times and resources](2026-10-runtime-rules.json)
- [Laya CPU samples, worker startup times and resources](2026-10-runtime-laya.json)
- [Separate synthetic decision baseline](2026-10-rules-baseline.json)

With the updated runtime installed:

```powershell
python scripts/benchmark_review_runtime.py --mode rules --checks all --cold-runs 3 --repeats 10 --output rules-runtime.json
python scripts/benchmark_review_runtime.py --mode laya --checkpoint "C:\Models\archrivet-preview" --device cpu --checks all --cold-runs 3 --repeats 10 --output laya-runtime.json
```

Use `--device cuda` only in a runtime with compatible CUDA PyTorch. The desktop Laya bundle is CPU-only. No benchmark command trains or downloads weights. Each child process closes its worker after the trial.
