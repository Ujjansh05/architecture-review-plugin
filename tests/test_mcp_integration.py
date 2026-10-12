"""Real subprocess protocol tests with two clients sharing one loopback daemon."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.fixture
def daemon(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    (tmp_path / "settings.json").write_text(json.dumps({"port": port}), encoding="utf-8")
    token = "stdio-integration-test-token"
    (tmp_path / "token").write_text(token)
    env = dict(os.environ, ARCH_REVIEW_HOME=str(tmp_path))
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log = (tmp_path / "server.log").open("w")
    process = subprocess.Popen([sys.executable, "-m", "arch_review", "serve"], env=env,
                               stdout=log, stderr=log, creationflags=flags)
    base = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer " + token}
    try:
        with httpx.Client(trust_env=False, timeout=.5) as client:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail((tmp_path / "server.log").read_text())
                try:
                    response = client.get(base + "/health", headers=headers)
                    if response.status_code == 200:
                        serving_pid = response.json()["pid"]
                        break
                except httpx.TransportError:
                    time.sleep(.1)
            else:
                pytest.fail("Test daemon did not become healthy")
        yield env, base, headers, serving_pid
    finally:
        try:
            with httpx.Client(trust_env=False, timeout=3) as client:
                client.post(base + "/v1/shutdown", headers=headers)
            process.wait(timeout=10)
        except (httpx.HTTPError, subprocess.TimeoutExpired):
            process.terminate()
            process.wait(timeout=10)
        log.close()


async def test_two_mcp_clients_use_one_service_and_preserve_unknowns(daemon):
    env, base, headers, pid = daemon
    async def review():
        parameters = StdioServerParameters(command=sys.executable, args=["-m", "arch_review", "mcp"],
                                           env={"ARCH_REVIEW_HOME": env["ARCH_REVIEW_HOME"]})
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=30) as session:
                await session.initialize()
                listing = await session.list_tools()
                assert [tool.name for tool in listing.tools] == ["review_architecture"]
                result = await session.call_tool("review_architecture", arguments={
                    "requirements": {}, "design": {"components": ["Queue"]}, "workload": {}, "deployment": {}})
                body = result.model_dump(by_alias=True)
                assert not body.get("isError", False)
                response = body.get("structuredContent")
                if response is None:
                    response = json.loads(result.content[0].text)
                assert response["review_status"] == "partial"
                assert all(check["verdict"] == "insufficient_information" for check in response["checks"])
                return response
    responses = await asyncio.gather(review(), review())
    assert len(responses) == 2
    with httpx.Client(trust_env=False, timeout=3) as client:
        state = client.get(base + "/health", headers=headers).json()
    assert state["pid"] == pid
    assert state["max_active"] == 1


def test_stdio_accepts_raw_jsonrpc_initialize_and_tool_listing(tmp_path):
    # An independent raw JSON-RPC client verifies stdout has no non-protocol text.
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen([sys.executable, "-m", "arch_review", "mcp"],
        env=dict(os.environ, ARCH_REVIEW_HOME=str(tmp_path)), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=flags)
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18",
            "capabilities": {}, "clientInfo": {"name": "raw-integration-client", "version": "1.0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    try:
        stdout, stderr = process.communicate("".join(json.dumps(m) + "\n" for m in messages), timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    output = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    assert any(message.get("id") == 1 and "result" in message for message in output), stderr
    assert any(message.get("id") == 2 and message.get("result", {}).get("tools") for message in output), stderr
