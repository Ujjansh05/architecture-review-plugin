"""Opt-in, resumable model setup with separate verified-file and loaded-worker states."""
import hashlib
import os
import shutil
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from filelock import FileLock, Timeout

from arch_review.merged_download import (
    ARCHIVE_BYTES,
    ARCHIVE_SHA256,
    EXPANDED_BYTES,
    MODEL_ID,
    RELEASE,
    extract_archive,
    verify_directory,
)


class ModelSetup:
    def __init__(self, root, enabled=False, checkpoint=None, loader=None, is_loaded=None, release=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else self.root / "models" / MODEL_ID
        self.enabled = enabled
        self.loader = loader
        self.is_loaded = is_loaded or (lambda: False)
        self.release = release or (lambda: None)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.state = {"status": "not_requested", "downloaded_bytes": 0,
                      "total_bytes": ARCHIVE_BYTES, "error": None, "files_verified": False,
                      "timings_seconds": {}}
        self.closed = False

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def status(self):
        with self.lock:
            loaded = self.enabled and self.is_loaded()
            if self.state["status"] == "ready" and not loaded:
                self.state["status"] = "unloaded"
            return {**self.state, "model_requested": self.enabled,
                    "timings_seconds": dict(self.state["timings_seconds"]),
                    "active_mode": "laya" if loaded else "unavailable" if self.enabled else "rules",
                    "worker_loaded": bool(loaded),
                    "model_id": MODEL_ID, "device": "cpu", "model_quality": "experimental",
                    "download_percent": round(100 * self.state["downloaded_bytes"] / ARCHIVE_BYTES, 1)}

    def start(self):
        with self.lock:
            self.status()  # Refresh readiness after idle unloading or a worker exit.
            if self.closed:
                return self.status()
            if not self.enabled:
                return self.status()
            if self.state["status"] == "ready" or (self.thread and self.thread.is_alive()):
                return self.status()
            self.stop.clear()
            self.update(status="checking", error=None, files_verified=False, timings_seconds={})
            self.thread = threading.Thread(target=self._run, daemon=True, name="archrivet-model-setup")
            self.thread.start()
            return self.status()

    def cancel(self):
        """Pause pending setup, retaining verified files and partial downloads."""
        with self.lock:
            if self.thread and self.thread.is_alive() and self.state["status"] != "ready":
                self.stop.set()
                self.update(status="pausing")
            return self.status()

    def _check_cancelled(self):
        if self.stop.is_set():
            raise InterruptedError()

    def _timed(self, name, function):
        started = time.perf_counter()
        try:
            return function()
        finally:
            with self.lock:
                self.state["timings_seconds"][name] = round(time.perf_counter() - started, 6)

    def _run(self):
        started = time.perf_counter()
        try:
            self._check_cancelled()
            # Another host may already be installing the shared model. Never race extraction.
            with FileLock(str(self.root / "model-install.lock"), timeout=0):
                self._timed("files_setup", self._install)
            self._check_cancelled()
            if self.loader is not None:
                self.update(status="loading_model")
                self._timed("worker_initialization", lambda: self.loader(self.stop.is_set))
                self._check_cancelled()
                if not self.is_loaded():
                    raise RuntimeError("Model worker exited before readiness could be confirmed")
                self.update(status="ready")
        except Timeout:
            self.update(status="waiting_for_other_install", error="Another session is installing the model. Retry shortly.")
        except InterruptedError:
            self.release()
            self.update(status="paused", error="Setup paused; the next launch can resume the download.")
        except Exception as exc:
            self.release()
            self.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            with self.lock:
                self.state["timings_seconds"]["total_setup"] = round(time.perf_counter() - started, 6)

    def _install(self):
        if self.checkpoint.exists():
            self.update(status="verifying")
            self._timed("checkpoint_verification", lambda: verify_directory(self.checkpoint))
            self._check_cancelled()
            self.update(status="files_ready", downloaded_bytes=ARCHIVE_BYTES, files_verified=True)
            return
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = self.root / (MODEL_ID + ".zip.part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > ARCHIVE_BYTES:
            partial.unlink()
            offset = 0
        # Download and extraction may be on different drives when reusing a checkpoint path.
        if shutil.disk_usage(self.root).free < ARCHIVE_BYTES - offset + 256 * 1024**2:
            raise ValueError("Not enough disk space for the model download.")
        if shutil.disk_usage(self.checkpoint.parent).free < EXPANDED_BYTES + ARCHIVE_BYTES - offset + 256 * 1024**2:
            raise ValueError("Not enough disk space to download and extract the model.")
        if offset < ARCHIVE_BYTES:
            headers = {"User-Agent": "ArchRivet-Desktop/0.3.0-alpha.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(f"{RELEASE}/{MODEL_ID}.zip", headers=headers)
            with urllib.request.urlopen(request, timeout=60) as incoming:
                if incoming.status == 206:
                    expected = f"bytes {offset}-{ARCHIVE_BYTES - 1}/{ARCHIVE_BYTES}"
                    if incoming.headers.get("Content-Range") != expected:
                        raise ValueError("Unexpected resumed download range.")
                elif incoming.status == 200:
                    offset = 0
                else:
                    raise ValueError("Unexpected download response.")
                self.update(status="downloading", downloaded_bytes=offset)
                with partial.open("ab" if offset else "wb") as outgoing:
                    while chunk := incoming.read(1024 * 1024):
                        if self.stop.is_set():
                            raise InterruptedError()
                        offset += len(chunk)
                        if offset > ARCHIVE_BYTES:
                            raise ValueError("Download exceeds the pinned model size.")
                        outgoing.write(chunk)
                        self.update(downloaded_bytes=offset)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
        self.update(status="verifying")
        def checksum():
            with partial.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        digest = self._timed("archive_verification", checksum)
        if partial.stat().st_size != ARCHIVE_BYTES or digest != ARCHIVE_SHA256:
            partial.unlink()  # Only this fixed-name, private download is removed.
            raise ValueError("Model checksum verification failed. Retry setup to download a fresh copy.")
        if self.stop.is_set():
            raise InterruptedError()
        self.update(status="extracting")
        with tempfile.TemporaryDirectory(prefix=".install-", dir=self.checkpoint.parent) as temporary:
            staged = Path(temporary) / "checkpoint"
            self._timed("extraction_and_verification", lambda: extract_archive(partial, staged, EXPANDED_BYTES))
            self._check_cancelled()
            if self.checkpoint.exists():
                raise ValueError("Model destination appeared during installation; existing files were preserved.")
            staged.rename(self.checkpoint)
        partial.unlink()
        self.update(status="files_ready", downloaded_bytes=ARCHIVE_BYTES, files_verified=True)

    def close(self):
        with self.lock:
            self.closed = True
            self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
