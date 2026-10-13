import os
import subprocess
import sys
import time

import httpx
from filelock import FileLock

from .schema import ReviewRequest, ReviewResponse
from .settings import get_token, load_settings, runtime_dir


def endpoint():
    return f"http://127.0.0.1:{load_settings().port}"


def health():
    with httpx.Client(trust_env=False, timeout=2) as client:
        response = client.get(endpoint() + "/health", headers={"Authorization": "Bearer " + get_token()})
        response.raise_for_status()
        result = response.json()
        if result.get("service") != "architecture-review-plugin":
            raise RuntimeError("The configured port belongs to a different service.")
        return result


def ensure_service():
    root = runtime_dir()
    with FileLock(str(root / "start.lock"), timeout=30):
        try:
            return health()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
        # Authentication and protocol failures are deliberately not treated as permission to replace a server.
        if sys.platform == "win32" and os.environ.get("ARCH_REVIEW_MCP_SESSION") == "1":
            raise RuntimeError(
                "Shared service is not running. Run `python -m arch_review start` in an independent terminal "
                "with the same ARCH_REVIEW_HOME before connecting this Windows MCP client. "
                "A service started inside the client would terminate when that client closes."
            )
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        with (root / "service.log").open("ab") as log:
            child = subprocess.Popen([sys.executable, "-m", "arch_review", "serve"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=root, **options)
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError(f"Local service exited; inspect {root / 'service.log'}.")
            try:
                return health()
            except (httpx.ConnectError, httpx.ConnectTimeout):
                time.sleep(0.1)
        raise TimeoutError(f"Local service did not start; inspect {root / 'service.log'}.")


def review_remote(request: ReviewRequest) -> ReviewResponse:
    ensure_service()
    # There can be one active request and four queued requests, each with a bounded inference deadline.
    timeout = 5 * load_settings().inference_timeout_seconds + 15
    with httpx.Client(trust_env=False, timeout=timeout) as client:
        response = client.post(endpoint() + "/v1/reviews", json=request.model_dump(mode="json"),
                               headers={"Authorization": "Bearer " + get_token()})
        response.raise_for_status()
        return ReviewResponse.model_validate(response.json())


def design_job(method, job_id=None, request=None):
    ensure_service()
    if job_id is not None and (len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id)):
        raise ValueError("Invalid review job identifier")
    path = "/v2/reviews" + ("/" + job_id if job_id else "")
    with httpx.Client(trust_env=False, timeout=30) as client:
        kwargs = {"json": request.model_dump(mode="json")} if request is not None else {}
        response = client.request(method, endpoint() + path, headers={"Authorization": "Bearer " + get_token()}, **kwargs)
        response.raise_for_status()
        return response.json()
