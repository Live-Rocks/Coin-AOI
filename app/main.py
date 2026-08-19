"""FastAPI entry point for the Coin-AOI portfolio demo."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import BoundedSemaphore
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.rate_limit import FixedWindowRateLimiter
from app.yolo_service import InvalidImageError, ModelConfigurationError, Prediction, YoloService


LOGGER = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = APP_ROOT.parent
STATIC_DIRECTORY = APP_ROOT / "static"
EXAMPLE_IMAGE_PATH = STATIC_DIRECTORY / "examples" / "coin-dent.webp"
ARTIFACTS_DIRECTORY = REPOSITORY_ROOT / "artifacts"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
MAX_FORM_FILES = 1
MAX_FORM_FIELDS = 0
PREDICTION_RATE_LIMIT = 5
PREDICTION_RATE_WINDOW_SECONDS = 60
ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}


class PredictionService(Protocol):
    """Small boundary that lets web tests avoid loading a real model."""

    def ensure_ready(self) -> None: ...

    def predict_bytes(self, contents: bytes) -> Prediction: ...


def create_app(service: PredictionService | None = None) -> FastAPI:
    """Build the app with an optional injectable prediction service."""
    app = FastAPI(
        title="Coin-AOI Portfolio Demo",
        description="Research prototype for visualizing YOLO defect detections.",
        version="0.1.0",
    )
    app.state.prediction_service = service or YoloService()
    app.state.inference_slot = BoundedSemaphore(value=1)
    app.state.rate_limiter = FixedWindowRateLimiter(
        PREDICTION_RATE_LIMIT, PREDICTION_RATE_WINDOW_SECONDS
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")
    app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIRECTORY), name="artifacts")

    @app.middleware("http")
    async def reject_oversized_prediction(
        request: Request, call_next
    ) -> JSONResponse | object:
        """Reject declared oversized multipart bodies before form parsing."""
        if request.method == "POST" and request.url.path == "/api/predict":
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "Invalid Content-Length header."},
                    )
                if declared_size > MAX_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={"detail": "The request body is too large."},
                    )
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        """Report ready only after the configured checkpoint can be loaded."""
        try:
            await run_in_threadpool(app.state.prediction_service.ensure_ready)
        except ModelConfigurationError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
            ) from error
        except Exception as error:
            LOGGER.exception("YOLO readiness check failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="YOLO checkpoint could not be loaded.",
            ) from error
        return {"status": "ready"}

    @app.post("/api/predict")
    async def predict(request: Request) -> dict[str, object]:
        async with request.form(
            max_files=MAX_FORM_FILES,
            max_fields=MAX_FORM_FIELDS,
            max_part_size=MAX_UPLOAD_BYTES,
        ) as form:
            image = form.get("image")
            if not isinstance(image, UploadFile):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Please choose one image file.",
                )
            contents = await _read_upload(image)
        return await _run_limited_prediction(app, request, contents)

    @app.post("/api/predict-example")
    async def predict_example(request: Request) -> dict[str, object]:
        if not EXAMPLE_IMAGE_PATH.is_file():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The bundled example image is unavailable.",
            )
        return await _run_limited_prediction(
            app, request, await run_in_threadpool(EXAMPLE_IMAGE_PATH.read_bytes)
        )

    return app


async def _read_upload(image: UploadFile) -> bytes:
    """Validate upload metadata and read no more than the configured byte limit."""
    if image.content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG, PNG, and WebP images are supported.",
        )
    contents = await image.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Please choose an image file.",
        )
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The image must be 10 MB or smaller.",
        )
    return contents


async def _run_limited_prediction(
    app: FastAPI, request: Request, contents: bytes
) -> dict[str, object]:
    """Reject overloaded callers and move blocking work off the event loop."""
    client_key = request.client.host if request.client else "unknown"
    if not app.state.rate_limiter.allow(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Prediction rate limit reached. Please try again later.",
        )
    if not app.state.inference_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="The model is busy. Please try again shortly.",
        )
    try:
        return await run_in_threadpool(
            _predict_with_service, app.state.prediction_service, contents
        )
    finally:
        app.state.inference_slot.release()


def _predict_with_service(
    service: PredictionService, contents: bytes
) -> dict[str, object]:
    """Translate predictable inference failures into clear HTTP responses."""
    try:
        return service.predict_bytes(contents).to_dict()
    except InvalidImageError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except ModelConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


app = create_app()
