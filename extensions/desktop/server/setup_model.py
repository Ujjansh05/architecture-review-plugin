"""Opt-in, resumable download of the already published, checksum-pinned model."""
import hashlib
import os
import shutil
import tempfile
import threading
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
    def __init__(self, root, enabled=False, checkpoint=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else self.root / "models" / MODEL_ID
        self.enabled = enabled
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.state = {"status": "not_requested", "downloaded_bytes": 0,
                      "total_bytes": ARCHIVE_BYTES, "error": None}

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def status(self):
        with self.lock:
            return {**self.state, "model_requested": self.enabled,
                    "active_mode": "laya" if self.state["status"] == "ready" else "rules",
                    "model_id": MODEL_ID, "device": "cpu", "model_quality": "experimental",
                    "download_percent": round(100 * self.state["downloaded_bytes"] / ARCHIVE_BYTES, 1)}

    def start(self):
        with self.lock:
            if not self.enabled:
                return self.status()
            if self.state["status"] == "ready" or (self.thread and self.thread.is_alive()):
                return self.status()
            self.stop.clear()
            self.update(status="checking", error=None)
            self.thread = threading.Thread(target=self._run, daemon=True, name="archrivet-model-setup")
            self.thread.start()
            return self.status()

    def _run(self):
        try:
            # Another host may already be installing the shared model. Never race extraction.
            with FileLock(str(self.root / "model-install.lock"), timeout=0):
                self._install()
        except Timeout:
            self.update(status="waiting_for_other_install", error="Another session is installing the model. Retry shortly.")
        except InterruptedError:
            self.update(status="paused", error="Setup paused; the next launch can resume the download.")
        except Exception as exc:
            self.update(status="failed", error=f"{type(exc).__name__}: {exc}")

    def _install(self):
        if self.checkpoint.exists():
            self.update(status="verifying")
            verify_directory(self.checkpoint)
            self.update(status="ready", downloaded_bytes=ARCHIVE_BYTES)
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
            headers = {"User-Agent": "ArchRivet-Desktop/0.1.0-alpha.1"}
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
        with partial.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if partial.stat().st_size != ARCHIVE_BYTES or digest != ARCHIVE_SHA256:
            partial.unlink()  # Only this fixed-name, private download is removed.
            raise ValueError("Model checksum verification failed. Retry setup to download a fresh copy.")
        if self.stop.is_set():
            raise InterruptedError()
        self.update(status="extracting")
        with tempfile.TemporaryDirectory(prefix=".install-", dir=self.checkpoint.parent) as temporary:
            staged = Path(temporary) / "checkpoint"
            extract_archive(partial, staged, EXPANDED_BYTES)
            if self.checkpoint.exists():
                raise ValueError("Model destination appeared during installation; existing files were preserved.")
            staged.rename(self.checkpoint)
        partial.unlink()
        self.update(status="ready", downloaded_bytes=ARCHIVE_BYTES)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
