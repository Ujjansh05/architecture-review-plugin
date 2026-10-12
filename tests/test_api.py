import asyncio
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from arch_review.api import create_app
from arch_review.service import ReviewService
from arch_review.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "integration-test-token"
HEADERS = {"Authorization": "Bearer " + TOKEN}


def test_authentication_body_limits_and_complete_http_contract():
    with TestClient(create_app(Settings(), TOKEN)) as client:
        assert client.get("/health").status_code == 401
        assert client.get("/health", headers={**HEADERS, "Origin": "https://untrusted.example"}).status_code == 403
        assert client.post("/v1/reviews", headers=HEADERS, content=b"x" * 100001).status_code == 413
        assert client.post("/v1/reviews", headers=HEADERS, json={}).status_code == 422
        payload = (ROOT / "examples/flawed.json").read_text()
        response = client.post("/v1/reviews", headers={**HEADERS, "Content-Type": "application/json"}, content=payload)
        assert response.status_code == 200
        assert response.json()["review_status"] == "partial"
        assert {c["verdict"] for c in response.json()["checks"]} == {"contradicted"}
        assert "requirements" in client.get("/v1/schema", headers=HEADERS).json()["input"]["properties"]


async def test_admission_limits_and_responsive_health():
    class SlowService(ReviewService):
        def review(self, req, **kwargs):
            time.sleep(.15)
            return super().review(req, rules_only=True)
    service = SlowService()
    app = create_app(Settings(), TOKEN, service=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        async def call():
            return await client.post("/v1/reviews", headers=HEADERS, json={"requirements": {}, "design": {}})
        jobs = [asyncio.create_task(call()) for _ in range(7)]
        await asyncio.sleep(.05)
        assert (await client.get("/health", headers=HEADERS)).status_code == 200
        responses = await asyncio.gather(*jobs)
        assert sum(r.status_code == 200 for r in responses) == 5
        assert sum(r.status_code == 503 for r in responses) == 2
        assert (await client.get("/health", headers=HEADERS)).json()["active_or_waiting"] == 0
