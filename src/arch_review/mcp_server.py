import asyncio

from mcp.server import MCPServer

from . import __version__
from .client import review_remote
from .schema import Deployment, Design, Requirements, ReviewRequest, Workload


def create_server():
    server = MCPServer("architecture-review", version=__version__, instructions=(
        "After drafting a messaging architecture, call review_architecture. Preserve missing information, "
        "review_status, model-role warnings and source limitations when explaining findings. "
        "Use null for unknown facts. Do not infer guarantees from product or company names. "
        "After a relevant revision, one follow-up review is appropriate."))

    @server.tool()
    async def review_architecture(requirements: Requirements, design: Design,
                                  workload: Workload, deployment: Deployment) -> dict:
        """Review retries, acknowledgement durability and queue overload; calculate workload.

        Extract explicit facts only. Unknown fields must be null or omitted. The same
        acknowledged_operation identifier must name the exact boundary in requirements
        and design. Idempotency booleans cover ALL relevant effects throughout the full
        retry window. survives_failures is an exhaustive declared guarantee, not a list
        of examples. This tool cannot certify a deployment or predict outage probability.
        """
        request = ReviewRequest(requirements=requirements, design=design, workload=workload, deployment=deployment)
        result = await asyncio.to_thread(review_remote, request)
        return result.model_dump(mode="json")

    return server


def run():
    create_server().run(transport="stdio")
