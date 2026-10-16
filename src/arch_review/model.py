from __future__ import annotations

import contextlib
import json
import math
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

from .decisions import question
from .schema import ModelDecision, ReviewRequest, Verdict
from .settings import Settings, model_metadata


def build_state(req: ReviewRequest, check_id: str, evidence: list[dict]) -> str:
    section = {"retry_safety": "retry", "ack_durability": "acknowledgement", "queue_overload": "queue"}[check_id]
    # Keep all declared requirements and shared flow facts; never silently trim important facts.
    state = {"requirements": req.requirements.model_dump(exclude_none=True),
             "components": req.design.components, "flows": req.design.flows,
             "design_notes": req.design.notes,
             "decision_details": getattr(req.design, section).model_dump(exclude_none=True),
             "evidence": [{"id": r["id"], "conditions": r["conditions"], "lesson": r["lesson"]}
                          for r in evidence]}
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False)


def _worker(connection, settings_dict):
    """Only this subprocess imports torch; exiting it releases the model's RAM."""
    settings = Settings.model_validate(settings_dict)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # This worker uses PyTorch only; MCP hosts may strip optional parent env flags.
    os.environ["USE_TF"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from .checkpoint import configure_checkpoint_loading

            configure_checkpoint_loading()
            import laya
            import torch

            imported = time.perf_counter()
            torch.set_num_threads(settings.threads)
            checkpoint = Path(settings.checkpoint).resolve()
            if not checkpoint.is_dir():
                raise FileNotFoundError("Configured local checkpoint directory does not exist.")
            device = settings.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but this PyTorch environment has no available CUDA device")
            loading = time.perf_counter()
            agent = laya.load(str(checkpoint), device=device)
            loaded = time.perf_counter()
            metadata = model_metadata(str(checkpoint))
            from .model_contract import verify_contract
            contract = verify_contract(checkpoint, settings.model_context_tokens)
            verified = time.perf_counter()
            if metadata.get("inference_precision") == "float32" and metadata.get("autocast") is False:
                # Merged experiments were evaluated in FP32, including their calibration.
                agent.amp_enabled, agent.dtype = False, torch.float32
            connection.send({"ready": True, "device": device,
                             "inference_precision": str(agent.dtype).removeprefix("torch."),
                             "autocast": agent.amp_enabled,
                             "runtime_metrics": {"startup": {
                                 "import_seconds": round(imported - started, 6),
                                 "model_load_seconds": round(loaded - loading, 6),
                                 "contract_verification_seconds": round(verified - loaded, 6),
                                 "total_seconds": round(verified - started, 6),
                                 **_device_memory(torch, device)}}})
            while True:
                if not connection.poll(settings.idle_seconds):
                    break
                payloads = connection.recv()
                if payloads is None:
                    break
                request_started = time.perf_counter()
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                results, timings = {}, {}
                for key, payload in payloads.items():
                    decision_started = time.perf_counter()
                    try:
                        if contract is not None and json.loads(payload["state"]).get("input_contract") != contract["input_contract"]:
                            raise ValueError("Input does not use this model's evaluated fact formatter")
                        result = agent.predict(payload["state"], payload["questions"],
                                               max_len=settings.model_context_tokens, head_max_len=256)
                        usage = result.get("usage", {})
                        if usage.get("truncated") or usage.get("state_tokens_dropped", 0) > 0:
                            raise ValueError("Model context overflow: narrow the design; no truncated verdict was used.")
                        if key in usage.get("options", {}) and usage["options"][key]["distinct"] < 3:
                            raise ValueError("Decision labels collapsed in the model context.")
                        answer = result["answers"][key]
                        label_map = payload.get("label_map", {})
                        label = Verdict(label_map.get(answer["choice"], answer["choice"]))
                        probabilities = {label_map.get(key, key): value for key, value in answer["probabilities"].items()}
                        if set(probabilities) != {v.value for v in Verdict}:
                            raise ValueError("Model returned an unexpected label set.")
                        if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities.values()):
                            raise ValueError("Model returned invalid probabilities.")
                        if abs(sum(probabilities.values()) - 1) > 0.01:
                            raise ValueError("Model probabilities do not sum to one.")
                        results[key] = {"verdict": label.value, "probabilities": probabilities}
                    except Exception as exc:
                        results[key] = {"error": f"{type(exc).__name__}: {exc}"}
                    finally:
                        if device == "cuda":
                            torch.cuda.synchronize()
                        timings[key] = round(time.perf_counter() - decision_started, 6)
                results["_runtime_metrics"] = {
                    "inference_seconds": round(time.perf_counter() - request_started, 6),
                    "decision_seconds": timings, **_device_memory(torch, device)}
                connection.send(results)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({"error": f"{type(exc).__name__}: {exc}"})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


def _device_memory(torch, device):
    """Measure only this worker's CUDA allocator; CPU never probes a driver."""
    if device != "cuda":
        return {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}
    return {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}


class ModelWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process = None
        self.connection = None
        self.lock = threading.Lock()
        self._ready = threading.Event()
        self.metadata = model_metadata(settings.checkpoint)

    @property
    def pid(self):
        process = self.process
        try:
            return process.pid if process is not None and process.is_alive() else None
        except (ValueError, OSError, AssertionError):
            # Status polling must not fail while another thread closes the process.
            return None

    @property
    def is_ready(self):
        """A live process is usable only after its current initialization handshake."""
        process = self.process
        if process is None or not self._ready.is_set():
            return False
        try:
            return process.is_alive() and self.process is process and self._ready.is_set()
        except (ValueError, OSError, AssertionError):
            return False

    def _stop(self):
        self._ready.clear()
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def close(self):
        with self.lock:
            self._stop()

    def _receive(self, deadline, cancelled=None):
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("Model initialization cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Local model load/inference exceeded its time budget.")
            if self.connection.poll(min(remaining, .2) if cancelled is not None else remaining):
                return self.connection.recv()

    def _ensure_ready(self, deadline, cancelled=None):
        if self.process is None or not self.process.is_alive():
            self._stop()
            context = mp.get_context("spawn")
            self.connection, child_connection = context.Pipe()
            self.process = context.Process(target=_worker, args=(child_connection, self.settings.model_dump()), daemon=True)
            self.process.start()
            child_connection.close()
            ready = self._receive(deadline, cancelled)
            if not ready.get("ready"):
                raise RuntimeError(ready.get("error", "Model failed to initialize."))
            self.metadata["active_device"] = ready.get("device", "unknown")
            self.metadata["active_precision"] = ready.get("inference_precision", "unknown")
            self.metadata["active_autocast"] = ready.get("autocast")
            self.metadata["runtime_metrics"] = ready.get("runtime_metrics", {})
            self._ready.set()

    def ensure_ready(self, cancelled=None):
        """Load and verify without scoring a design; initialization errors remain visible."""
        if not self.settings.checkpoint:
            raise RuntimeError("No Laya checkpoint configured")
        with self.lock:
            try:
                self._ensure_ready(time.monotonic() + self.settings.inference_timeout_seconds, cancelled)
                return dict(self.metadata)
            except (OSError, EOFError, RuntimeError, TimeoutError):
                self._stop()
                raise

    def predict(self, payloads: dict) -> tuple[dict[str, ModelDecision], dict[str, str]]:
        if not self.settings.checkpoint:
            return {}, {key: "No Laya checkpoint configured. Rules-only review is incomplete." for key in payloads}
        with self.lock:
            deadline = time.monotonic() + self.settings.inference_timeout_seconds
            try:
                self._ensure_ready(deadline)
                self.connection.send(payloads)
                raw = self._receive(deadline)
                metrics = self.metadata.setdefault("runtime_metrics", {})
                metrics["last_request"] = raw.pop("_runtime_metrics", {})
                metrics["request_count"] = metrics.get("request_count", 0) + 1
                metrics["total_inference_seconds"] = round(metrics.get("total_inference_seconds", 0)
                    + metrics["last_request"].get("inference_seconds", 0), 6)
            except (OSError, EOFError, RuntimeError, TimeoutError) as exc:
                exit_code = self.process.exitcode if self.process is not None else None
                self._stop()
                message = str(exc) or f"Model worker exited unexpectedly (exit code {exit_code}); inspect the service log for load or memory errors."
                return {}, {key: message for key in payloads}
        decisions, errors = {}, {}
        for key in payloads:
            value = raw.get(key, {"error": raw.get("error", "Missing model result.")})
            if "error" in value:
                errors[key] = value["error"]
            else:
                decisions[key] = ModelDecision(**value, checkpoint=self.settings.checkpoint,
                    calibration_status=self.metadata.get("calibration_status", "unvalidated"))
        return decisions, errors


def payload(check_id, state):
    return {"state": state, "questions": question(check_id)}
