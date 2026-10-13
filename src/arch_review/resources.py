"""Sample process-tree RSS without importing a tensor runtime or probing GPU drivers."""
import os
import threading

import psutil


class ResourceSampler:
    def __init__(self, interval_seconds=.25):
        self.interval = interval_seconds
        self.peak = 0
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="review-rss-sampler", daemon=True)

    def sample(self):
        try:
            parent = psutil.Process(os.getpid())
            processes = [parent, *parent.children(recursive=True)]
        except psutil.Error:
            return
        total = 0
        for process in processes:
            try:
                total += process.memory_info().rss
            except psutil.Error:
                pass
        self.peak = max(self.peak, total)
        self.samples += 1

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.sample()

    def report(self):
        return {"peak_sampled_process_tree_rss_bytes": self.peak, "sample_count": self.samples,
                "sample_interval_seconds": self.interval, "gpu_peak_bytes": None,
                "measurement_scope": "Current process and descendants; shared pages may be counted more than once. External llama servers excluded.",
                "notice": "Sampled RSS is not a strict peak or a guarantee the next input will fit."}
