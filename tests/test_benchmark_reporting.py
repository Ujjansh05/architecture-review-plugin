import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

from arch_review.backends import NoModelBackend
from arch_review.jobs import ReviewJobs

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case():
    return {"id": "one", "questions": {"cache_freshness": {}},
            "expected": {"cache_freshness": "insufficient_information"}, "state": json.dumps({"facts": {}})}


@pytest.mark.parametrize("change", ["duplicate", "multiple_checks", "unknown_label", "missing_label"])
def test_rules_benchmark_rejects_ambiguous_or_duplicate_cases(tmp_path, change):
    row = case()
    if change == "multiple_checks":
        row["questions"]["retry_safety"] = {}
    elif change == "unknown_label":
        row["expected"]["cache_freshness"] = "probably safe"
    elif change == "missing_label":
        row["expected"] = {}
    source = tmp_path / "cases.jsonl"
    source.write_text(json.dumps(row) + "\n" + (json.dumps(row) + "\n" if change == "duplicate" else ""))
    with pytest.raises(ValueError):
        script("benchmark_rules_baseline").run(source)


def test_rules_benchmark_binds_results_to_the_bytes_it_scored(tmp_path):
    import hashlib
    source = tmp_path / "cases.jsonl"
    payload = (json.dumps(case()) + "\n").encode()
    source.write_bytes(payload)
    report = script("benchmark_rules_baseline").run(source)
    assert report["count"] == report["correct"] == 1
    assert report["input_sha256"] == hashlib.sha256(payload).hexdigest()


def test_runtime_percentiles_do_not_hide_small_samples():
    module = script("benchmark_review_runtime")
    assert module.summary([]) == {"count": 0, "median_ms": None, "p95_ms": None}
    assert module.summary([10, 1, 5])["p95_ms"] == 10
    assert module.summary([10, 1, 5])["count"] == 3


class Result:
    def model_dump(self, **kwargs):
        return {"complete": True}


class Reviewer:
    backend = NoModelBackend()
    def review(self, request, cancelled, progress):
        return Result()


def completed(manager, job_id):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(.001)
    raise AssertionError("Job did not finish")


def test_terminal_job_is_never_visible_without_complete_timing():
    manager = ReviewJobs(Reviewer())
    observations = []
    class ObservedLock:
        def __init__(self):
            self.inner = threading.RLock()
        def __enter__(self):
            self.inner.acquire()
            return self
        def __exit__(self, *_):
            for job in manager.jobs.values():
                if job["status"] in {"completed", "failed", "cancelled"}:
                    observations.append("total" in job["timings_ms"])
            self.inner.release()
    manager.lock = ObservedLock()
    try:
        result = completed(manager, manager.submit(None)["job_id"])
        assert result["status"] == "completed"
        assert observations and all(observations)
        assert result["timings_ms"]["total"] >= result["timings_ms"]["queue_wait"]
    finally:
        manager.close()


def test_cancelled_queued_job_retains_queue_timing_and_snapshot():
    execution = threading.Lock()
    execution.acquire()
    manager = ReviewJobs(Reviewer(), execution_lock=execution)
    try:
        submitted = manager.submit(None)
        manager.cancel(submitted["job_id"])
        execution.release()
        result = completed(manager, submitted["job_id"])
        assert result["status"] == "cancelled"
        assert result["timings_ms"]["total"] >= result["timings_ms"]["queue_wait"] >= 0
        assert submitted["timings_ms"] == {}
    finally:
        if execution.locked():
            execution.release()
        manager.close()
