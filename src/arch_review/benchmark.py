import math
import os
import platform
import statistics
import threading
import time

import psutil

from .client import ensure_service, review_remote


def benchmark(request, runs=5):
    if runs < 2:
        raise ValueError("Use at least two runs to distinguish the first request from warm requests.")
    status = ensure_service()
    server = psutil.Process(status["pid"])
    peak = 0
    stop = threading.Event()
    def sample():
        nonlocal peak
        while not stop.is_set():
            processes = [server, psutil.Process(os.getpid())]
            try:
                processes += server.children(recursive=True)
            except psutil.Error:
                pass
            total = 0
            for process in processes:
                try:
                    total += process.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, total)
            stop.wait(.02)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    elapsed, modes, model_counts = [], [], []
    try:
        for _ in range(runs):
            start = time.perf_counter()
            result = review_remote(request)
            elapsed.append(time.perf_counter() - start)
            modes.append(result.review_mode)
            model_counts.append(sum(check.model_decision is not None for check in result.checks))
    finally:
        stop.set()
        sampler.join()
    warm = sorted(elapsed[1:])
    p95 = warm[max(0, math.ceil(.95 * len(warm)) - 1)]
    valid = all(count == 3 for count in model_counts)
    return {"runs": runs, "first_request_seconds": elapsed[0],
            "first_request_was_cold": status["model_worker_pid"] is None,
            "warm_p95_seconds": p95, "warm_median_seconds": statistics.median(warm),
            "peak_aggregate_rss_bytes": peak, "rss_scope": "HTTP daemon, model descendants and benchmark CLI; summed RSS may include shared pages.",
            "memory_target_met": peak <= 4 * 1024 ** 3 if valid else None,
            "latency_target_met": p95 <= 5 if valid else None,
            "model_benchmark_valid": valid, "model_decisions_per_request": model_counts,
            "all_requests_used_model": valid,
            "review_modes": modes, "hardware": {"os": platform.platform(), "cpu": platform.processor(),
                "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                "total_ram_bytes": psutil.virtual_memory().total}, "durations_seconds": elapsed,
            "warning": "A short local benchmark does not establish production capacity or a stable p95."}
