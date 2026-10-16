"""Exercise an extracted desktop bundle over real stdio, optionally reusing local Laya weights."""
import argparse
import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def structured(result):
    body = result.model_dump(by_alias=True)
    if body.get("isError"):
        raise RuntimeError(str(result.content))
    return body.get("structuredContent") or json.loads(result.content[0].text)


def request(bound=None):
    facts = {"max_staleness_ms": 5000, "worst_case_staleness_ms": 3000,
             "bound_covers_failures": True, "stale_on_error": True}
    if bound is not None:
        facts["stale_on_error_bound_ms"] = bound
    return {"input_mode": "supplied_facts", "requested_checks": ["cache_freshness"],
            "checks": [{"check_id": "cache_freshness", "scope_id": "catalog",
                        "facts": facts, "fact_origins": {key: "user_declared" for key in facts}}]}


async def validate(directory, checkpoint=None, uv="uv"):
    directory = directory.resolve(strict=True)
    config = json.loads((directory / "bundle-config.json").read_text(encoding="utf-8"))
    if checkpoint is not None and config["variant"] != "laya-cpu":
        raise ValueError("Only the Laya CPU bundle may load model weights")
    if checkpoint is not None:
        checkpoint = checkpoint.resolve(strict=True)
    command = [uv, "run", "--frozen", "--offline", "--managed-python", "--directory", str(directory)]
    inventory_code = "import importlib.metadata,json; print(json.dumps(sorted(d.metadata['Name'].lower().replace('_','-') for d in importlib.metadata.distributions())))"
    installed = json.loads(subprocess.run(command + ["python", "-c", inventory_code], check=True,
                                         capture_output=True, text=True).stdout)
    if config["variant"] == "rules" and {"torch", "laya", "transformers", "numpy", "safetensors"}.intersection(installed):
        raise ValueError("Rules environment contains model dependencies")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="archrivet-stdio-") as temporary:
        environment = {"ARCHRIVET_EXTENSION_HOME": temporary, "PYTHONPATH": "",
                       "ARCHRIVET_ENABLE_LAYA": "true" if checkpoint is not None else "false"}
        if checkpoint is not None:
            environment["ARCHRIVET_MODEL_DIR"] = str(checkpoint)
        parameters = StdioServerParameters(command=uv, args=command[1:] + ["server/main.py"], env=environment)
        observations, reviews = [], []
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=30) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = sorted(tool.name for tool in tools.tools)
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                assert names == sorted(tool["name"] for tool in manifest["tools"])
                schema = structured(await session.call_tool("get_review_schema", {}))
                assert len(schema["decisions"]) == 8
                deadline = time.monotonic() + 240
                while True:
                    status = structured(await session.call_tool("get_setup_status", {}))
                    state = status["setup"]["status"]
                    if not observations or observations[-1]["status"] != state:
                        observations.append({"status": state, "worker_loaded": status["setup"]["worker_loaded"],
                                             "elapsed_seconds": round(time.perf_counter() - started, 4)})
                    if checkpoint is None:
                        assert status["setup"]["active_mode"] == "rules"
                        break
                    if state == "ready":
                        assert status["setup"]["worker_loaded"]
                        break
                    if state in {"failed", "paused"} or time.monotonic() > deadline:
                        raise RuntimeError(str(status))
                    await asyncio.sleep(.1)
                for bound in (None, 60000):
                    job = structured(await session.call_tool("review_design", {"request": request(bound)}))
                    deadline = time.monotonic() + 240
                    while job["status"] in {"queued", "running", "cancelling"}:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Packaged review timed out")
                        await asyncio.sleep(.05)
                        job = structured(await session.call_tool("get_review", {"job_id": job["job_id"]}))
                    assert job["status"] == "completed", job
                    if checkpoint is not None:
                        assert job["result"]["checks"][0]["model_decision"] is not None, job
                    reviews.append(job)
                assert reviews[0]["result"]["checks"][0]["verdict"] == "insufficient_information"
                assert reviews[1]["result"]["checks"][0]["verdict"] == "contradicted"
                comparison = structured(await session.call_tool("compare_reviews", {
                    "before_job_id": reviews[0]["job_id"], "after_job_id": reviews[1]["job_id"]}))
    return {"variant": config["variant"], "engine_version": config["engine_version"],
            "tools": names, "installed_packages": installed, "setup_observations": observations,
            "last_setup_status": status["setup"], "reviews": reviews, "comparison": comparison,
            "local_checkpoint_reused": checkpoint is not None, "gui_installation_verified": False,
            "publication_performed": False, "passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(validate(args.bundle, args.checkpoint, args.uv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("variant", "engine_version", "local_checkpoint_reused", "passed")}))
