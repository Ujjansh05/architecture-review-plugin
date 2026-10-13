"""Bounded in-memory review jobs. Architecture text is not written to disk."""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor


class ReviewJobs:
    def __init__(self, reviewer, execution_lock=None, max_pending=5, ttl_seconds=3600, review_timeout_seconds=1800):
        self.reviewer = reviewer
        self.execution_lock = execution_lock or threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="architecture-review")
        self.lock = threading.RLock()
        self.jobs = {}
        self.max_pending, self.ttl = max_pending, ttl_seconds
        self.closed = False
        self.review_timeout = review_timeout_seconds
        self.stop_cleanup = threading.Event()
        self.cleanup = threading.Thread(target=self._cleanup_loop, name="review-result-expiry", daemon=True)
        self.cleanup.start()

    def _cleanup_loop(self):
        while not self.stop_cleanup.wait(min(60, max(.01, self.ttl))):
            with self.lock:
                self._expire()

    @property
    def pending(self):
        with self.lock:
            return sum(j["status"] in {"queued", "running", "cancelling"} for j in self.jobs.values())

    def _expire(self):
        now = time.monotonic()
        completed = [key for key, job in self.jobs.items() if job["status"] in {"completed", "failed", "cancelled"}]
        for key in completed:
            if now - self.jobs[key]["updated"] > self.ttl:
                del self.jobs[key]
        while len(self.jobs) >= 24:
            oldest = next((key for key in self.jobs if self.jobs[key]["status"] in {"completed", "failed", "cancelled"}), None)
            if oldest is None:
                break
            del self.jobs[oldest]

    def submit(self, request):
        with self.lock:
            self._expire()
            if self.closed or self.pending >= self.max_pending:
                raise RuntimeError("Review queue is full or shutting down")
            key = uuid.uuid4().hex
            self.jobs[key] = {"job_id": key, "status": "queued", "progress": {}, "result": None,
                              "error": None, "updated": time.monotonic(), "cancel": threading.Event()}
            self.pool.submit(self._run, key, request)
            return self.get(key)

    def _run(self, key, request):
        with self.execution_lock:
            job = self.jobs[key]
            try:
                if job["cancel"].is_set():
                    raise InterruptedError()
                with self.lock:
                    job["status"] = "running"
                deadline = time.monotonic() + self.review_timeout
                def cancelled():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Review exceeded its overall time budget; no complete result was returned")
                    return job["cancel"].is_set()
                def progress(**values):
                    with self.lock:
                        job["progress"].update(values)
                result = self.reviewer.review(request, cancelled=cancelled, progress=progress)
                with self.lock:
                    if cancelled():
                        job["status"] = "cancelled"
                    else:
                        job.update(status="completed", result=result.model_dump(mode="json"))
            except InterruptedError:
                with self.lock:
                    job["status"] = "cancelled"
            except Exception as exc:
                with self.lock:
                    job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                with self.lock:
                    job["updated"] = time.monotonic()

    def get(self, key):
        with self.lock:
            self._expire()
            job = self.jobs[key]
            return {k: v for k, v in job.items() if k not in {"cancel", "updated"}}

    def cancel(self, key):
        with self.lock:
            job = self.jobs[key]
            if job["status"] in {"queued", "running", "cancelling"}:
                job["cancel"].set()
                job["status"] = "cancelling"
            return self.get(key)

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                job["cancel"].set()
        self.pool.shutdown(wait=True)
        self.stop_cleanup.set()
        self.cleanup.join(timeout=1)
        self.reviewer.backend.close()
