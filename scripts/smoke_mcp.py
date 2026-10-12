"""Exercise two actual stdio MCP clients against the configured local model service.

This verifies the protocol and shared model process, not two AI product integrations.
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from arch_review.client import health
from arch_review.schema import ReviewResponse
from arch_review.settings import runtime_dir

ROOT = Path(__file__).resolve().parents[1]


async def review(name):
    parameters = StdioServerParameters(command=sys.executable, args=["-m", "arch_review", "mcp"],
        env={"ARCH_REVIEW_HOME": str(runtime_dir())})
    request = json.loads((ROOT / "examples" / f"{name}.json").read_text())
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=240) as session:
            await session.initialize()
            result = await session.call_tool("review_architecture", arguments=request)
            body = result.model_dump(by_alias=True)
            if body.get("isError"):
                raise RuntimeError(str(result.content))
            response = body.get("structuredContent") or json.loads(result.content[0].text)
            parsed = ReviewResponse.model_validate(response)
            if any(check.model_decision is None for check in parsed.checks):
                raise RuntimeError("Model smoke check requires three real model decisions: " + str(parsed.warnings))
            return {"example": name, "review_mode": parsed.review_mode,
                    "verdicts": {check.check_id: check.verdict.value for check in parsed.checks}}


async def main():
    responses = await asyncio.gather(review("complete"), review("flawed"))
    state = health()
    if state["model_worker_pid"] is None:
        raise RuntimeError("Expected one live shared model worker")
    report = {"clients": 2, "transport": "stdio", "service_pid": state["pid"],
              "model_worker_pid": state["model_worker_pid"], "responses": responses,
              "actual_ai_hosts_tested": 0}
    output = ROOT / "reports" / "mcp-model-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
