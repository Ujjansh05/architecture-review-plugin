"""Desktop lifecycle and dependency boundaries, without downloading model weights."""
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "extensions" / "desktop"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


setup_lib = load_module("setup_model", DESKTOP / "server/setup_model.py")
desktop_module = load_module("archrivet_desktop_test", DESKTOP / "server/main.py")
builder = load_module("archrivet_bundle_test", ROOT / "scripts/build_desktop_variants.py")


def finish(setup):
    setup.thread.join(timeout=3)
    assert not setup.thread.is_alive()
    return setup.status()


def test_variants_replace_same_plugin_and_rules_have_no_model_setting():
    rules, laya = builder.manifest_for("rules"), builder.manifest_for("laya-cpu")
    assert rules["name"] == laya["name"] == "archrivet"
    assert rules["version"] == laya["version"]
    assert rules["user_config"] == {}
    assert "ARCHRIVET_ENABLE_LAYA" not in rules["server"]["mcp_config"]["env"]
    assert laya["user_config"]["enable_laya"]["default"] is False
    assert {tool["name"] for tool in rules["tools"]} == {tool["name"] for tool in laya["tools"]}
    project = tomllib.loads(builder.project_for("rules", "0.3.0a2", "engine.whl"))
    assert project["project"]["dependencies"] == ["architecture-review-plugin==0.3.0a2"]
    assert "torch" not in project["tool"]["uv"]["sources"]
    assert "index" not in project["tool"]["uv"]


def test_rules_lock_rejects_transitive_model_dependency(tmp_path):
    (tmp_path / "uv.lock").write_text('[[package]]\nname="torch"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="model libraries"):
        builder.audit_lock(tmp_path, "rules")


def test_files_verified_does_not_mean_worker_ready(tmp_path, monkeypatch):
    checkpoint = tmp_path / "old-model"
    checkpoint.mkdir()
    checked = []
    monkeypatch.setattr(setup_lib, "verify_directory", lambda path: checked.append(path))
    loading, release, loaded = threading.Event(), threading.Event(), threading.Event()

    def loader(cancelled):
        loading.set()
        assert release.wait(2)
        loaded.set()

    setup = setup_lib.ModelSetup(tmp_path, True, checkpoint, loader=loader, is_loaded=loaded.is_set)
    try:
        setup.start()
        assert loading.wait(2)
        status = setup.status()
        assert status["status"] == "loading_model"
        assert status["files_verified"] is True
        assert status["worker_loaded"] is False
        assert status["active_mode"] == "unavailable"
        release.set()
        status = finish(setup)
        assert status["status"] == "ready"
        assert status["worker_loaded"] is True
        assert checked == [checkpoint.resolve()]
        assert status["timings_seconds"]["checkpoint_verification"] >= 0
        loaded.clear()
        assert setup.status()["status"] == "unloaded"
    finally:
        release.set()
        setup.close()


def test_failed_loading_can_retry_and_setup_can_be_cancelled(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    monkeypatch.setattr(setup_lib, "verify_directory", lambda _: None)
    attempts, entered, loaded = [], threading.Event(), threading.Event()

    def loader(cancelled):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("test load failure")
        if len(attempts) == 2:
            entered.set()
            while not cancelled():
                time.sleep(.01)
            raise InterruptedError()
        loaded.set()

    setup = setup_lib.ModelSetup(tmp_path, True, checkpoint, loader=loader, is_loaded=loaded.is_set)
    try:
        setup.start()
        assert finish(setup)["status"] == "failed"
        assert "test load failure" in setup.status()["error"]
        setup.start()
        assert entered.wait(2)
        setup.cancel()
        assert finish(setup)["status"] == "paused"
        assert checkpoint.is_dir()
        setup.start()
        assert finish(setup)["status"] == "ready"
        assert setup.status()["error"] is None
    finally:
        setup.close()


def test_checkpoint_tampering_blocks_loader(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()

    def reject(_):
        raise ValueError("Model checksum mismatch")

    monkeypatch.setattr(setup_lib, "verify_directory", reject)
    setup = setup_lib.ModelSetup(tmp_path, True, checkpoint, loader=lambda _: pytest.fail("must not load"))
    try:
        setup.start()
        assert finish(setup)["status"] == "failed"
        assert setup.status()["files_verified"] is False
    finally:
        setup.close()


def test_rules_does_not_download_or_allow_model_enable(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_lib.ModelSetup, "_install", lambda _: pytest.fail("no download permitted"))
    extension = desktop_module.Extension(tmp_path, variant="rules")
    try:
        assert extension.setup.status()["active_mode"] == "rules"
        assert extension.setup.thread is None
        assert extension.backend is None
        with pytest.raises(ValueError, match="Replace it"):
            desktop_module.Extension(tmp_path, enabled=True, variant="rules")
    finally:
        extension.close()


def test_rules_launch_can_import_with_model_libraries_blocked(tmp_path):
    code = '''import importlib.abc, runpy, sys
class DenyModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'laya', 'transformers', 'numpy', 'safetensors'}:
            raise AssertionError('Rules imported a model library: ' + fullname)
sys.meta_path.insert(0, DenyModels())
sys.path.insert(0, sys.argv[1])
module = runpy.run_path(sys.argv[1] + '/main.py')
extension = module['Extension'](sys.argv[2], variant='rules')
module['create_server'](extension)
assert extension.setup.status()['active_mode'] == 'rules'
extension.close()
print('rules-ready-without-model-libraries')
'''
    result = subprocess.run([sys.executable, "-c", code, str(DESKTOP / "server"), str(tmp_path)],
                            capture_output=True, text=True, timeout=30, check=False,
                            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    assert result.returncode == 0, result.stderr
    assert "rules-ready-without-model-libraries" in result.stdout


def test_two_extension_sessions_isolate_jobs_and_shutdown(tmp_path):
    from arch_review.schema_v2 import DesignReviewRequest

    first = desktop_module.Extension(tmp_path, variant="rules")
    second = desktop_module.Extension(tmp_path, variant="rules")
    request = DesignReviewRequest.model_validate({"checks": [{"check_id": "cache_freshness", "facts": {}}],
                                                  "requested_checks": ["cache_freshness"]})
    try:
        before = first.submit(request)
        with pytest.raises(ValueError, match="different assistant session"):
            second.submit(request)
            second.get(before["job_id"])
        first.close()
        after = second.submit(request)
        deadline = time.monotonic() + 3
        while after["status"] in {"queued", "running"}:
            assert time.monotonic() < deadline
            time.sleep(.01)
            after = second.get(after["job_id"])
        assert after["status"] == "completed"
    finally:
        first.close()
        second.close()


def test_ready_api_preserves_initialization_errors_and_cancellation(monkeypatch):
    from arch_review.model import ModelWorker, _device_memory
    from arch_review.settings import Settings

    worker = ModelWorker(Settings(checkpoint="nonexistent-test-checkpoint"))
    stopped = []

    def fail(*_):
        raise RuntimeError("exact initialization failure")

    monkeypatch.setattr(worker, "_ensure_ready", fail)
    monkeypatch.setattr(worker, "_stop", lambda: stopped.append(True))
    with pytest.raises(RuntimeError, match="exact initialization failure"):
        worker.ensure_ready()
    assert stopped == [True]
    with pytest.raises(InterruptedError):
        worker._receive(time.monotonic() + 10, cancelled=lambda: True)
    assert _device_memory(object(), "cpu") == {
        "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}


def test_live_worker_is_not_ready_before_handshake_and_closed_pid_is_safe():
    from arch_review.model import ModelWorker
    from arch_review.settings import Settings

    class Process:
        pid = 123
        closed = False

        def is_alive(self):
            if self.closed:
                raise ValueError("process object is closed")
            return True

    worker = ModelWorker(Settings())
    worker.process = Process()
    assert worker.pid == 123
    assert worker.is_ready is False
    worker._ready.set()
    assert worker.is_ready is True
    worker._ready.clear()  # A replacement process must handshake again.
    assert worker.is_ready is False
    worker.process.closed = True
    assert worker.pid is None
    assert worker.is_ready is False


@pytest.mark.skipif(os.name != "nt", reason="Desktop bundle currently supports Windows x64")
async def test_desktop_stdio_rules_tools_and_review(tmp_path):
    import asyncio

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(command=sys.executable, args=[str(DESKTOP / "server/main.py")],
        env={"ARCHRIVET_EXTENSION_HOME": str(tmp_path), "ARCHRIVET_ENABLE_LAYA": "false",
             "PYTHONPATH": str(ROOT / "src")})

    def structured(result):
        body = result.model_dump(by_alias=True)
        assert not body.get("isError", False), result.content
        return body.get("structuredContent") or json.loads(result.content[0].text)

    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=30) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {tool["name"] for tool in builder.manifest_for("rules")["tools"]}
            status = structured(await session.call_tool("get_setup_status", {}))
            assert status["variant"] == "rules"
            assert status["setup"]["worker_loaded"] is False
            schema = structured(await session.call_tool("get_review_schema", {}))
            assert len(schema["decisions"]) == 8
            job = structured(await session.call_tool("review_design", {"request": {
                "input_mode": "supplied_facts", "checks": [{"check_id": "cache_freshness", "facts": {}}],
                "requested_checks": ["cache_freshness"]}}))
            deadline = time.monotonic() + 10
            while job["status"] in {"queued", "running"}:
                assert time.monotonic() < deadline
                await asyncio.sleep(.01)
                job = structured(await session.call_tool("get_review", {"job_id": job["job_id"]}))
            assert job["status"] == "completed", job
            assert job["result"]["checks"][0]["verdict"] == "insufficient_information"
