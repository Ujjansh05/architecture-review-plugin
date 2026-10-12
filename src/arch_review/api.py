import asyncio
import hmac
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .schema import ReviewRequest, ReviewResponse
from .service import ReviewService
from .settings import Settings


def create_app(settings: Settings, token: str, service=None):
    if not token:
        raise ValueError("A non-empty local authentication token is required.")
    review_service = service or ReviewService(settings)
    active_or_waiting = 0
    serial = asyncio.Semaphore(1)

    @asynccontextmanager
    async def lifespan(app):
        yield
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
                if size > 100_000:
                    return JSONResponse({"detail": "Review body exceeds 100 KB."}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        return await call_next(request)

    @app.get("/health", dependencies=[Depends(authorize)])
    async def health():
        return {"service": "architecture-review-plugin", "version": __version__, "pid": os.getpid(),
                "model_worker_pid": review_service.worker.pid, "checkpoint": settings.checkpoint,
                "model_role": review_service.worker.metadata.get("role"),
                "active_or_waiting": active_or_waiting, "max_active": 1, "max_waiting": 4}

    @app.get("/v1/schema", dependencies=[Depends(authorize)])
    async def schema():
        return {"input": ReviewRequest.model_json_schema(), "output": ReviewResponse.model_json_schema()}

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
        if active_or_waiting >= 5:
            raise HTTPException(503, "Review queue is full; retry later.", headers={"Retry-After": "1"})
        active_or_waiting += 1
        try:
            async with serial:
                task = asyncio.create_task(asyncio.to_thread(review_service.review, req))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # A disconnected HTTP caller does not release an in-flight model slot early.
                    await task
                    raise
        finally:
            active_or_waiting -= 1

    return app
