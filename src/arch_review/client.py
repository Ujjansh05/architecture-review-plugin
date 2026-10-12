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
