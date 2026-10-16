"""Measure complete local review jobs, separately from AI-host generation.

Runs sequential subprocesses for process-cold measurements and repeated warm
reviews. Does not install packages, fetch weights, train, or modify settings.
"""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "median_ms": None, "p95_ms": None}
    return {"count": len(values), "median_ms": round(statistics.median(values), 3),
            "p95_ms": round(values[math.ceil(.95 * len(values)) - 1], 3),
            "min_ms": round(values[0], 3), "max_ms": round(values[-1], 3)}


def child(args):
    started = time.perf_counter()
    from arch_review import __version__
    from arch_review.backend_dataset import positive_facts
    from arch_review.backends import make_backend
    from arch_review.documents import flatten
    from arch_review.evidence_v2 import PersistentEvidence
    from arch_review.jobs import ReviewJobs
    from arch_review.registry import REGISTRY
    from arch_review.review_v2 import DesignReviewer
    from arch_review.schema_v2 import CheckInput, DesignReviewRequest
    from arch_review.settings import Settings

    imported = time.perf_counter()
    requested = list(REGISTRY) if args.checks == "all" else [args.checks]
    inputs = []
    for check in requested:
        facts = positive_facts(check)
        inputs.append(CheckInput(check_id=check, facts=facts,
                                 fact_origins={path: "user_declared" for path in flatten(facts)}))
    request = DesignReviewRequest(input_mode="supplied_facts", checks=inputs, requested_checks=requested)
    settings = Settings(backend="laya" if args.mode == "laya" else "rules", checkpoint=args.checkpoint,
                        model_context_tokens=2048, device=args.device, threads=args.threads,
                        inference_timeout_seconds=args.timeout)
    backend = make_backend(settings)
    backend_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="archrivet-runtime-") as directory:
        evidence = PersistentEvidence(directory)
        ready = time.perf_counter()
        jobs = ReviewJobs(DesignReviewer(backend, evidence), review_timeout_seconds=args.timeout)
        results = []
        try:
            for index in range(args.repeats + 1):
                before = time.perf_counter()
                job = jobs.submit(request.model_copy(deep=True))
                deadline = time.monotonic() + args.timeout + 10
                while job["status"] not in {"completed", "failed", "cancelled"}:
                    if time.monotonic() > deadline:
                        jobs.cancel(job["job_id"])
                        raise TimeoutError("Benchmark review exceeded its timeout")
                    time.sleep(.002)
                    job = jobs.get(job["job_id"])
                elapsed = (time.perf_counter() - before) * 1000
                if index == 0:
                    initialization_to_first_result_ms = (time.perf_counter() - started) * 1000
                if job["status"] != "completed":
                    raise RuntimeError(job["error"] or job["status"])
                review = job["result"]
                if review["coverage"].get("inference_errors"):
                    raise RuntimeError("Inference failed; no speed result is accepted: " + str(review["coverage"]["inference_errors"]))
                results.append({"phase": "cold" if index == 0 else "warm", "elapsed_ms": round(elapsed, 3),
                                "job_timings_ms": job.get("timings_ms", {}),
                                "stage_timings_ms": review["coverage"].get("timings_ms", {}),
                                "resources": review["resources"],
                                "model_metrics": review["coverage"].get("model_metrics"),
                                "runtime_metrics": review["backend"].get("runtime_metrics"),
                                "verdicts": {r["check_id"]: r["verdict"] for r in review["checks"]}})
        finally:
            jobs.close()
    report = {"engine_version": __version__, "imports_ms": round((imported - started) * 1000, 3),
              "backend_setup_ms": round((backend_started - imported) * 1000, 3),
              "evidence_index_ms": round((ready - backend_started) * 1000, 3),
              "application_initialization_to_first_result_ms": round(initialization_to_first_result_ms, 3),
              "reviews": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["rules", "laya"], default="rules")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checks", default="all")
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10, help="Warm reviews per cold process")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold_runs < 1 or args.repeats < 1 or args.threads < 1 or args.timeout < 1:
        parser.error("cold-runs, repeats, threads and timeout must be positive")
    if args.mode == "laya" and (not args.checkpoint or not Path(args.checkpoint).is_dir()):
        parser.error("Laya requires an existing local --checkpoint directory")
    if args.child:
        child(args)
        return
    trials = []
    with tempfile.TemporaryDirectory(prefix="archrivet-measurement-") as directory:
        for index in range(args.cold_runs):
            output = Path(directory) / f"trial-{index}.json"
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--mode", args.mode,
                       "--device", args.device, "--threads", str(args.threads), "--checks", args.checks,
                       "--repeats", str(args.repeats), "--timeout", str(args.timeout), "--output", str(output)]
            if args.checkpoint:
                command += ["--checkpoint", args.checkpoint]
            started = time.perf_counter()
            result = subprocess.run(command, check=False, capture_output=True, text=True,
                                    timeout=(args.repeats + 1) * (args.timeout + 15) + 60,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode:
                raise RuntimeError("Benchmark child failed: " + result.stderr[-4000:])
            trial = json.loads(output.read_text(encoding="utf-8"))
            trial["process_total_ms"] = round((time.perf_counter() - started) * 1000, 3)
            trials.append(trial)
            print(f"Finished process {index + 1}/{args.cold_runs}", flush=True)
    reviews = [review for trial in trials for review in trial["reviews"]]
    report = {"schema_version": "archrivet-runtime-benchmark-1", "mode": args.mode,
              "device_preference": args.device, "threads": args.threads, "checks": args.checks,
              "platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
              "cold_reviews": summary([r["elapsed_ms"] for r in reviews if r["phase"] == "cold"]),
              "warm_reviews": summary([r["elapsed_ms"] for r in reviews if r["phase"] == "warm"]),
              "application_initialization_to_first_result": summary([t["application_initialization_to_first_result_ms"] for t in trials]),
              "peak_sampled_process_tree_rss_bytes": max(r["resources"]["peak_sampled_process_tree_rss_bytes"] for r in reviews),
              "trials": trials, "training_started": False,
              "measurement_scope": "Complete local review jobs include queue/poll overhead, resource sampling, final summary rendering and result serialization; exclude backend setup and evidence-index creation. Cold means first review in a fresh process, not cleared OS disk cache. Application initialization starts at child function entry after Python/script standard-library imports; includes ArchRivet imports, backend setup, evidence index and first review, not process launch. No host generation, package install or weight download. RSS sampled only during review; shared pages may be double-counted. CUDA allocator metrics are not total device use.",
              "accuracy_claim": "None: fixed declared performance fixtures, not independent real-world evaluation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
