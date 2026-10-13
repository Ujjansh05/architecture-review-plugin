import argparse
import importlib.metadata
import json
import multiprocessing
import os
import sys
from pathlib import Path

import httpx

from . import __version__
from .schema import ReviewRequest, ReviewResponse
from .settings import Settings, get_token, load_settings, model_metadata, runtime_dir, save_settings


def emit(value, output=None):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(prog="arch-review", description="Local messaging architecture review plugin")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("review")
    review.add_argument("input", type=Path)
    review.add_argument("--output", type=Path)
    review.add_argument("--rules-only", action="store_true", help="Explicit diagnostic baseline; returns an incomplete rules-only review")
    design = sub.add_parser("review-design", help="Review V2 structured checks or submit a document review job")
    design.add_argument("input", type=Path)
    design.add_argument("--output", type=Path)
    design.add_argument("--local", action="store_true", help="Run synchronously in this process without the shared service")
    design.add_argument("--wait", action="store_true", help="Poll a submitted job until completion")
    get_job = sub.add_parser("get-review")
    get_job.add_argument("job_id")
    cancel_job = sub.add_parser("cancel-review")
    cancel_job.add_argument("job_id")
    schema_v2 = sub.add_parser("schema-v2")
    schema_v2.add_argument("--output", type=Path)
    preflight = sub.add_parser("preflight", help="Read-only hardware/storage inspection; does not install or download")
    preflight.add_argument("--workspace", type=Path, default=Path.cwd())
    preflight.add_argument("--output", type=Path)
    sub.add_parser("start", help="Start the shared service before connecting Windows MCP clients")
    sub.add_parser("serve", help="Run the shared service in the foreground")
    sub.add_parser("stop")
    sub.add_parser("mcp")
    sub.add_parser("doctor")
    sub.add_parser("mcp-config")
    schema = sub.add_parser("schema")
    schema.add_argument("--output", type=Path)
    config = sub.add_parser("configure")
    config.add_argument("--checkpoint", type=Path)
    config.add_argument("--port", type=int)
    config.add_argument("--idle-seconds", type=int)
    config.add_argument("--backend", choices=["rules", "laya", "llama_cpp"])
    config.add_argument("--device", choices=["auto", "cpu", "cuda"])
    config.add_argument("--context-tokens", type=int)
    config.add_argument("--llama-endpoint")
    config.add_argument("--llama-server", type=Path)
    config.add_argument("--gguf", type=Path)
    config.add_argument("--evidence", type=Path)
    download = sub.add_parser("download-model")
    download.add_argument("--destination", required=True, type=Path)
    dataset = sub.add_parser("dataset")
    dataset.add_argument("action", choices=["audit", "export"])
    dataset.add_argument("--source", type=Path)
    dataset.add_argument("--output", type=Path)
    dataset.add_argument("--allow-unreviewed", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--source", type=Path)
    evaluate.add_argument("--split", choices=["train", "development", "calibration", "test", "all"], default="development")
    evaluate.add_argument("--backend", choices=["rules", "laya"], default="rules")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--predictions", type=Path)
    evaluate.add_argument("--perturb", action="store_true")
    evaluate.add_argument("--output", type=Path)
    bench = sub.add_parser("benchmark")
    bench.add_argument("input", type=Path)
    bench.add_argument("--runs", type=int, default=5)
    bench.add_argument("--output", type=Path)
    train = sub.add_parser("train")
    train.add_argument("--data", type=Path, required=True, help="An exported dataset directory")
    train.add_argument("--base", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--timeout-seconds", type=int, default=3600)
    train.add_argument("--hourly-rate-usd", type=float, default=0)
    train.add_argument("--allow-unreviewed", action="store_true")
    train.add_argument("--freeze-encoder", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            from .preflight import inspect_hardware
            emit(inspect_hardware(args.workspace), args.output)
        elif args.command == "schema-v2":
            from .registry import definitions
            from .schema_v2 import DesignReviewRequest, DesignReviewResponse
            emit({"input": DesignReviewRequest.model_json_schema(), "output": DesignReviewResponse.model_json_schema(),
                  "decisions": definitions()}, args.output)
        elif args.command in {"get-review", "cancel-review"}:
            from .client import design_job
            emit(design_job("GET" if args.command == "get-review" else "DELETE", job_id=args.job_id))
        elif args.command == "review-design":
            from .schema_v2 import DesignReviewRequest
            req = DesignReviewRequest.model_validate_json(args.input.read_text(encoding="utf-8-sig"))
            if args.local:
                from .backends import make_backend
                from .evidence_v2 import PersistentEvidence
                from .review_v2 import DesignReviewer
                settings = load_settings()
                backend = make_backend(settings)
                try:
                    import time
                    deadline = time.monotonic() + settings.review_timeout_seconds
                    def cancelled():
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Review exceeded its overall time budget")
                        return False
                    result = DesignReviewer(backend, PersistentEvidence(runtime_dir() / "evidence-index", settings.evidence_path)).review(req, cancelled=cancelled)
                    emit(result.model_dump(mode="json"), args.output)
                finally:
                    backend.close()
            else:
                import time

                from .client import design_job
                job = design_job("POST", request=req)
                while args.wait and job["status"] in {"queued", "running", "cancelling"}:
                    time.sleep(1)
                    job = design_job("GET", job_id=job["job_id"])
                emit(job, args.output)
                if job["status"] in {"failed", "cancelled"}:
                    raise SystemExit(2)
        elif args.command == "review":
            req = ReviewRequest.model_validate_json(args.input.read_text(encoding="utf-8-sig"))
            if args.rules_only:
                from .service import ReviewService
                service = ReviewService()
                try:
                    response = service.review(req, rules_only=True)
                finally:
                    service.close()
            else:
                from .client import review_remote
                response = review_remote(req)
            emit(response.model_dump(mode="json"), args.output)
        elif args.command == "start":
            from .client import ensure_service
            emit(ensure_service())
        elif args.command == "serve":
            import uvicorn
            from filelock import FileLock

            from .api import create_app
            settings = load_settings()
            with FileLock(str(runtime_dir() / "serve.lock"), timeout=0):
                app = create_app(settings, get_token())
                server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=settings.port,
                                                       workers=1, access_log=False, log_level="warning"))
                app.state.shutdown = lambda: setattr(server, "should_exit", True)
                server.run()
        elif args.command == "stop":
            from .client import endpoint, health
            health()
            with httpx.Client(trust_env=False, timeout=10) as client:
                response = client.post(endpoint() + "/v1/shutdown", headers={"Authorization": "Bearer " + get_token()})
                response.raise_for_status()
                emit(response.json())
        elif args.command == "mcp":
            from .mcp_server import run
            # Windows MCP hosts own a job that also kills every service spawned
            # underneath the stdio process. Connect to an independently started daemon.
            previous = os.environ.get("ARCH_REVIEW_MCP_SESSION")
            os.environ["ARCH_REVIEW_MCP_SESSION"] = "1"
            try:
                run()
            finally:
                if previous is None:
                    os.environ.pop("ARCH_REVIEW_MCP_SESSION", None)
                else:
                    os.environ["ARCH_REVIEW_MCP_SESSION"] = previous
        elif args.command == "doctor":
            import psutil

            from .client import health
            settings = load_settings()
            packages = {}
            for name in ["architecture-review-plugin", "mcp", "laya", "torch", "transformers", "pydantic"]:
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            try:
                service_status = health()
            except Exception as exc:
                service_status = {"status": "not_connected", "detail": str(exc)}
            emit({"python": sys.version, "executable": sys.executable, "packages": packages,
                  "settings": settings.model_dump(), "model": model_metadata(settings.checkpoint),
                  "runtime_directory": str(runtime_dir()), "service": service_status,
                  "total_ram_bytes": psutil.virtual_memory().total})
        elif args.command == "mcp-config":
            emit({"mcpServers": {"architecture-review": {"command": sys.executable,
                  "args": ["-m", "arch_review", "mcp"], "env": {"ARCH_REVIEW_HOME": str(runtime_dir())}}}})
        elif args.command == "schema":
            emit({"input": ReviewRequest.model_json_schema(), "output": ReviewResponse.model_json_schema()}, args.output)
        elif args.command == "configure":
            previous = load_settings()
            values = previous.model_dump()
            if args.checkpoint:
                directory = args.checkpoint.resolve()
                if not (directory / "rl_agent_config.json").exists():
                    raise ValueError("Checkpoint directory lacks rl_agent_config.json")
                values["checkpoint"] = str(directory)
                values["model_role"] = model_metadata(str(directory)).get("role", "unregistered")
            if args.port is not None:
                values["port"] = args.port
            if args.idle_seconds is not None:
                values["idle_seconds"] = args.idle_seconds
            for option, field in [("backend", "backend"), ("device", "device"), ("context_tokens", "model_context_tokens"),
                                  ("llama_endpoint", "llama_endpoint"), ("llama_server", "llama_server_path"),
                                  ("gguf", "gguf_path"), ("evidence", "evidence_path")]:
                value = getattr(args, option)
                if value is not None:
                    values[field] = str(value.resolve()) if isinstance(value, Path) else value
            # Avoid a silently stale daemon after settings are changed.
            from filelock import FileLock
            with FileLock(str(runtime_dir() / "serve.lock"), timeout=0):
                settings = Settings.model_validate(values)
                save_settings(settings)
            emit(settings.model_dump())
        elif args.command == "download-model":
            from .download import download_checkpoint
            emit(download_checkpoint(args.destination))
        elif args.command == "dataset":
            from .dataset import export_dataset, load_cases, validate_cases
            if args.action == "audit":
                emit(validate_cases(load_cases(args.source)), args.output)
            else:
                if args.output is None:
                    raise ValueError("dataset export requires --output DIRECTORY")
                emit(export_dataset(args.output, args.source, args.allow_unreviewed))
        elif args.command == "evaluate":
            from .evaluation import evaluate
            checkpoint = str(args.checkpoint.resolve()) if args.checkpoint else load_settings().checkpoint
            if args.backend == "laya" and not checkpoint:
                raise ValueError("Laya evaluation requires a configured checkpoint or --checkpoint")
            emit(evaluate(args.source, args.split, args.backend, checkpoint, args.predictions, args.perturb), args.output)
        elif args.command == "benchmark":
            from .benchmark import benchmark
            req = ReviewRequest.model_validate_json(args.input.read_text(encoding="utf-8-sig"))
            emit(benchmark(req, args.runs), args.output)
        elif args.command == "train":
            from .training import run_training
            emit(run_training(args))
    except (ValueError, RuntimeError, OSError, TimeoutError, httpx.HTTPError) as exc:
        print(f"arch-review: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
