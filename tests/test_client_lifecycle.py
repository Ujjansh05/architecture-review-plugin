"""Shared-service startup checks without a model checkpoint or inference worker."""
import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import psutil
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from arch_review import client


def test_windows_mcp_cannot_autostart_session_bound_service(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH_REVIEW_HOME", str(tmp_path))
    monkeypatch.setenv("ARCH_REVIEW_MCP_SESSION", "1")
    monkeypatch.setattr(client, "sys", SimpleNamespace(platform="win32", executable=sys.executable))
    monkeypatch.setattr(client, "health", Mock(side_effect=httpx.ConnectError("No service")))
    spawn = Mock()
    monkeypatch.setattr(client.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match="python -m arch_review start.*independent terminal.*ARCH_REVIEW_HOME"):
        client.ensure_service()
    spawn.assert_not_called()


def test_windows_mcp_reuses_independently_started_service(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH_REVIEW_HOME", str(tmp_path))
    monkeypatch.setenv("ARCH_REVIEW_MCP_SESSION", "1")
    monkeypatch.setattr(client, "sys", SimpleNamespace(platform="win32", executable=sys.executable))
    status = {"service": "architecture-review-plugin", "pid": 12345}
    monkeypatch.setattr(client, "health", Mock(return_value=status))
    spawn = Mock()
    monkeypatch.setattr(client.subprocess, "Popen", spawn)
    assert client.ensure_service() == status
    spawn.assert_not_called()


async def test_cli_started_service_survives_each_mcp_client_close(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    (tmp_path / "settings.json").write_text(json.dumps({"port": port}), encoding="utf-8")
    token = "independent-service-lifecycle-test"
    (tmp_path / "token").write_text(token, encoding="utf-8")
    env = dict(os.environ, ARCH_REVIEW_HOME=str(tmp_path))
    env.pop("ARCH_REVIEW_MCP_SESSION", None)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = subprocess.run([sys.executable, "-m", "arch_review", "start"], env=env,
                             capture_output=True, text=True, timeout=30, creationflags=flags)
    assert started.returncode == 0, started.stderr
    state = json.loads(started.stdout)
    process = psutil.Process(state["pid"])
    base = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer " + token}
    try:
        assert state["model_worker_pid"] is None
        for _ in range(2):
            parameters = StdioServerParameters(command=sys.executable, args=["-m", "arch_review", "mcp"],
                                               env={"ARCH_REVIEW_HOME": str(tmp_path)})
            async with stdio_client(parameters) as (reader, writer):
                async with ClientSession(reader, writer, read_timeout_seconds=30) as session:
                    await session.initialize()
                    response = await session.call_tool("review_architecture", arguments={
                        "requirements": {}, "design": {"components": ["queue"]},
                        "workload": {}, "deployment": {}})
                    assert not response.model_dump(by_alias=True).get("isError", False)
            # Check after the SDK closes the client and its Windows Job Object.
            with httpx.Client(trust_env=False, timeout=3) as http:
                health = http.get(base + "/health", headers=headers)
                health.raise_for_status()
            assert health.json()["pid"] == state["pid"]
            assert health.json()["model_worker_pid"] is None
    finally:
        try:
            with httpx.Client(trust_env=False, timeout=3) as http:
                http.post(base + "/v1/shutdown", headers=headers).raise_for_status()
            process.wait(timeout=10)
        except (httpx.HTTPError, psutil.TimeoutExpired):
            if process.is_running():
                process.terminate()
                process.wait(timeout=10)
