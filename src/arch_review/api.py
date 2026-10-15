import asyncio
import hmac
import os
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .schema import ReviewRequest, ReviewResponse
from .schema_v2 import MAX_DOCUMENT_BYTES, DesignReviewRequest, DesignReviewResponse
from .service import ReviewService
from .settings import Settings


def create_app(settings: Settings, token: str, service=None):
    if not token:
        raise ValueError("A non-empty local authentication token is required.")
    review_service = service or ReviewService(settings)
    active_or_waiting = 0
    serial = asyncio.Semaphore(1)
    execution_lock = threading.Lock()
    from .backends import make_backend
    from .evidence_v2 import PersistentEvidence
    from .jobs import ReviewJobs
    from .review_v2 import DesignReviewer
    from .settings import runtime_dir

    backend = make_backend(settings, review_service.worker)
    reviewer = DesignReviewer(backend, PersistentEvidence(runtime_dir() / "evidence-index", settings.evidence_path))
    jobs = ReviewJobs(reviewer, execution_lock=execution_lock, review_timeout_seconds=settings.review_timeout_seconds)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(jobs.close)
        review_service.close()

    app = FastAPI(title="Architecture Review", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    async def authorize(request: Request):
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + token):
            raise HTTPException(status_code=401, detail="Local authentication required.")
        origin = request.headers.get("origin")
        if origin is not None:
            raise HTTPException(status_code=403, detail="Browser-origin requests are not supported.")

    @app.middleware("http")
    async def bounded_body(request, call_next):
        if request.method in {"POST", "PUT", "PATCH"}:
            # Authenticate before buffering. Never log architecture bodies or credentials.
            try:
                await authorize(request)
            except HTTPException as exc:
                return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                limit = MAX_DOCUMENT_BYTES if request.url.path == "/v2/reviews" else 100_000
                if size > limit:
                    return JSONResponse({"detail": f"Request body exceeds {limit} bytes."}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        return await call_next(request)

    @app.get("/health", dependencies=[Depends(authorize)])
    async def health():
        return {"service": "architecture-review-plugin", "version": __version__, "pid": os.getpid(),
                "model_worker_pid": review_service.worker.pid, "checkpoint": settings.checkpoint,
                "model_role": review_service.worker.metadata.get("role"),
                "active_or_waiting": active_or_waiting + jobs.pending, "max_active": 1, "max_waiting": 4}

    @app.get("/v1/schema", dependencies=[Depends(authorize)])
    async def schema():
        return {"input": ReviewRequest.model_json_schema(), "output": ReviewResponse.model_json_schema()}

    @app.get("/v2/schema", dependencies=[Depends(authorize)])
    async def schema_v2():
        from .registry import definitions
        return {"input": DesignReviewRequest.model_json_schema(), "output": DesignReviewResponse.model_json_schema(),
                "decisions": definitions()}

    @app.post("/v2/reviews", status_code=202, dependencies=[Depends(authorize)])
    async def submit_review(req: DesignReviewRequest):
        try:
            if active_or_waiting + jobs.pending >= 5:
                raise RuntimeError("Review queue is full; retry later.")
            return jobs.submit(req)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from None

    @app.get("/v2/reviews/{job_id}", dependencies=[Depends(authorize)])
    async def get_review(job_id: str):
        try:
            return jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown or expired review job") from None

    @app.delete("/v2/reviews/{job_id}", dependencies=[Depends(authorize)])
    async def cancel_review(job_id: str):
        try:
            return jobs.cancel(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown or expired review job") from None

    @app.get("/v2/comparison", dependencies=[Depends(authorize)])
    async def compare_reviews(before_job_id: str, after_job_id: str):
        try:
            return jobs.compare(before_job_id, after_job_id)
        except KeyError:
            raise HTTPException(404, "Unknown or expired review job") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/v1/shutdown", dependencies=[Depends(authorize)])
    async def shutdown():
        callback = getattr(app.state, "shutdown", None)
        if callback is None:
            raise HTTPException(409, "This application has no managed server shutdown callback.")
        asyncio.get_running_loop().call_later(.2, callback)
        return {"status": "stopping"}

    @app.post("/v1/reviews", response_model=ReviewResponse, dependencies=[Depends(authorize)])
    async def review(req: ReviewRequest):
        nonlocal active_or_waiting
        if settings.backend == "llama_cpp" and not settings.llama_server_path and settings.checkpoint:
            raise HTTPException(409, "Stop the externally managed generative server and select the Laya backend for V1 model reviews.")
        if active_or_waiting + jobs.pending >= 5:
            raise HTTPException(503, "Review queue is full; retry later.", headers={"Retry-After": "1"})
        active_or_waiting += 1
        try:
            async with serial:
                def run_review():
                    with execution_lock:
                        if backend.name == "llama_cpp":
                            backend.close()
                        return review_service.review(req)
                task = asyncio.create_task(asyncio.to_thread(run_review))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # A disconnected HTTP caller does not release an in-flight model slot early.
                    await task
                    raise
        finally:
            active_or_waiting -= 1

    return app
