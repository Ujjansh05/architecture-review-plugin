"""Download and exercise a pinned public release without changing host settings.

With --shared-dependencies, the package is newly installed but dependencies are
shared to save disk space. This is NOT a clean dependency installation test.
"""
import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import venv
from pathlib import Path

REPO = "Ujjansh05/archrivet"


def download(url, limit=2_000_000):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "arch-review-release-test"}), timeout=60) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Download exceeds the test's size bound")
    return body


def run(command, env, cwd, timeout=240):
    result = subprocess.run(command, env=env, cwd=cwd, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:] or result.stdout[-3000:])
    return result.stdout


async def check_mcp(executable, runtime, cases):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def client(name, request):
        parameters = StdioServerParameters(command=str(executable), args=["-m", "arch_review", "mcp"],
                                           env={"ARCH_REVIEW_HOME": runtime})
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=180) as session:
                await session.initialize()
                result = await session.call_tool("review_architecture", arguments=request)
                body = result.model_dump(by_alias=True)
                if body.get("isError"):
                    raise RuntimeError("Published MCP server returned an error")
                review = body.get("structuredContent") or json.loads(result.content[0].text)
                if any(check.get("model_decision") is None for check in review["checks"]):
                    raise RuntimeError("Published MCP server omitted model decisions")
                return {"example": name, "verdicts": {check["check_id"]: check["verdict"] for check in review["checks"]}}
    return await asyncio.gather(*(client(name, request) for name, request in cases.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v0.1.1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shared-dependencies", type=Path, action="append", default=[])
    parser.add_argument("--codex", type=Path)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    release = json.loads(download(f"https://api.github.com/repos/{REPO}/releases/tags/{args.tag}"))
    assets = {item["name"]: item for item in release["assets"]}
    checksums = download(assets["SHA256SUMS.txt"]["browser_download_url"]).decode()
    wheel_name = next(name for name in assets if name.endswith(".whl"))
    wheel = download(assets[wheel_name]["browser_download_url"])
    digest = hashlib.sha256(wheel).hexdigest()
    expected = next(line.split()[0] for line in checksums.splitlines() if line.split()[-1] == wheel_name)
    if digest != expected or assets[wheel_name].get("digest") != "sha256:" + digest:
        raise ValueError("Published wheel digest mismatch")
    wheel_path = destination / wheel_name
    wheel_path.write_bytes(wheel)
    environment = destination / "venv"
    venv.EnvBuilder(with_pip=not bool(args.shared_dependencies)).create(environment)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(ARCH_REVIEW_HOME=str(destination / "service"), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    if args.shared_dependencies:
        site = environment / ("Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
        # Process dependency .pth initialization too (notably pywin32 on Windows).
        # The fresh environment's own site-packages stays first; the import check
        # below rejects an accidentally imported workspace/editable package.
        (site / "shared-dependencies.pth").write_text(
            "".join("import site; site.addsitedir(" + repr(str(p.resolve())) + ")\n" for p in args.shared_dependencies), encoding="utf-8")
        installer = [sys.executable, "-m", "pip", "--python", str(executable)]
        extra = ["--no-deps"]
    else:
        installer = [str(executable), "-m", "pip"]
        extra = []
    run(installer + ["install", "--no-cache-dir", "--no-compile", *extra, str(wheel_path)], env, destination)
    imported = json.loads(run([str(executable), "-c", "import json,arch_review; print(json.dumps({'file':arch_review.__file__,'version':arch_review.__version__}))"], env, destination))
    if not Path(imported["file"]).resolve().is_relative_to(environment):
        raise RuntimeError("Release test accidentally imported a workspace package")
    run([str(executable), "-m", "arch_review", "configure", "--checkpoint", str(args.checkpoint.resolve()), "--port", "8767"], env, destination)
    root = Path(__file__).resolve().parents[1]
    cases = {name: json.loads((root / "examples" / f"{name}.json").read_text()) for name in ["complete", "flawed", "incomplete"]}
    report = {"release": release["html_url"], "wheel_sha256": digest, "fresh_package_import_verified": True,
              "dependency_isolation": "shared_existing_dependencies" if args.shared_dependencies else "fresh_install",
              "checkpoint_reused": True, "reviews": []}
    try:
        for name, case in cases.items():
            path = destination / (name + ".json")
            path.write_text(json.dumps(case), encoding="utf-8")
            started = time.monotonic()
            result = json.loads(run([str(executable), "-m", "arch_review", "review", str(path)], env, destination))
            (destination / (name + "-result.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
            if any(item.get("model_decision") is None for item in result["checks"]):
                raise RuntimeError("Release review omitted real model decisions")
            report["reviews"].append({"example": name, "elapsed_seconds": round(time.monotonic() - started, 3),
                                      "verdicts": {item["check_id"]: item["verdict"] for item in result["checks"]}})
        mcp = {"command": str(executable), "args": ["-m", "arch_review", "mcp"], "env": {"ARCH_REVIEW_HOME": env["ARCH_REVIEW_HOME"]}}
        (destination / "mcp.json").write_text(json.dumps({"mcpServers": {"architecture_review": mcp}}, indent=2), encoding="utf-8")
        report["mcp_protocol_clients"] = asyncio.run(check_mcp(executable, env["ARCH_REVIEW_HOME"], cases))
        if args.codex:
            prompt = (
                "You are participating in a controlled integration test, not doing coding work. "
                "Do not inspect files, run shell commands, or delegate. Use tool discovery if needed, then only architecture_review.review_architecture. "
                "Call that MCP tool once for EACH of the following three exact requests. Do not fix input facts before review. "
                "Then report the three returned verdicts per example, any disagreement with the model, missing evidence, "
                "and a short revision of the flawed architecture. Do not present a supported verdict as a proven capacity guarantee. "
                "If the tool is unavailable, say the integration failed.\n" + json.dumps(cases)
            )
            config = {
                "mcp_servers.architecture_review.command": mcp["command"],
                "mcp_servers.architecture_review.args": mcp["args"],
                "mcp_servers.architecture_review.env": mcp["env"],
                "mcp_servers.architecture_review.startup_timeout_sec": 45,
                "mcp_servers.architecture_review.tool_timeout_sec": 180,
                "mcp_servers.architecture_review.enabled": True,
                "mcp_servers.architecture_review.tools.review_architecture.approval_mode": "approve",
                "mcp_servers.architecture_review.tools.review_architecture.output_token_limit": 24000,
                "approval_policy": "never", "features.shell_tool": False,
            }
            command = [str(args.codex), "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--output-last-message", str(destination / "codex-answer.txt")]
            for key, value in config.items():
                # JSON strings/arrays are valid TOML values; tables need TOML '='.
                value_text = "{ " + ", ".join(k + " = " + json.dumps(v) for k, v in value.items()) + " }" if isinstance(value, dict) else json.dumps(value)
                command += ["-c", key + "=" + value_text]
            command += ["-"]
            started = time.monotonic()
            process = subprocess.run(command, input=prompt, text=True, encoding="utf-8", errors="replace", capture_output=True, env=env, cwd=destination, timeout=480)
            (destination / "codex-events.jsonl").write_text(process.stdout, encoding="utf-8")
            (destination / "codex-stderr.txt").write_text(process.stderr, encoding="utf-8")
            events = [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]
            calls = [event["item"] for event in events if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "mcp_tool_call"]
            report["codex"] = {"exit_code": process.returncode, "elapsed_seconds": round(time.monotonic() - started, 3),
                               "completed_mcp_calls": len(calls), "tool_statuses": [item.get("status") for item in calls],
                               "passed": process.returncode == 0 and len(calls) == 3 and all(item.get("status") == "completed" for item in calls)}
    finally:
        try:
            run([str(executable), "-m", "arch_review", "stop"], env, destination)
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
