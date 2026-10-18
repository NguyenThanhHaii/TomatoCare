"""Stateless image prediction API. Images live only for the duration of a request."""

import io
import logging
import os
import threading
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from tomatocare.inference import load_predictor

MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
logger = logging.getLogger(__name__)


class BodyLimitMiddleware:
    """Bound multipart parsing as well as the final uploaded file size."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        maximum = MAX_BYTES + 64 * 1024
        headers = dict(scope.get("headers", []))
        try:
            if int(headers.get(b"content-length", b"0")) > maximum:
                return await JSONResponse({"detail": "Image exceeds the 10 MB limit"}, status_code=413)(scope, receive, send)
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
        total = 0
        exceeded = False
        async def bounded_receive():
            nonlocal total, exceeded
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > maximum:
                    exceeded = True
                    # FastAPI may catch this while parsing; bounded_send replaces that response.
                    raise HTTPException(413, "Image exceeds the 10 MB limit")
            return message
        async def bounded_send(message):
            if exceeded and message["type"] == "http.response.start":
                message = {**message, "status": 413, "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]}
            if exceeded and message["type"] == "http.response.body":
                message = {**message, "body": b'{"detail":"Image exceeds the 10 MB limit"}', "more_body": False}
            await send(message)
        await self.app(scope, bounded_receive, bounded_send)


def decode_image(payload):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as original:
                if original.format not in ("JPEG", "PNG"):
                    raise ValueError("Only valid JPG and PNG images are accepted")
                if original.width * original.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError("Image dimensions exceed the safe 20 megapixel limit")
                original.load()
                return ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(422, f"Invalid image: {exc}") from exc


def create_app(predictor_factory=load_predictor):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        app.state.model_error = None
        try:
            app.state.predictor = await run_in_threadpool(predictor_factory)
        except Exception as exc:
            app.state.model_error = str(exc)
            logger.error("Model initialization failed: %s", exc)
        yield
        app.state.predictor = None

    app = FastAPI(title="TomatoCare", version="0.3.0", lifespan=lifespan)
    origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "").split(",") if item.strip()]
    if "*" in origins:
        raise ValueError("Set explicit CORS_ORIGINS; wildcard origins are not supported")
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    admission = threading.Lock()

    @app.middleware("http")
    async def private_responses(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health")
    async def health():
        if app.state.predictor is None:
            return JSONResponse({"status": "unavailable", "detail": app.state.model_error or "Model initializing"}, status_code=503)
        engine = app.state.predictor.engine
        return {"status": "ready", "model_identifier": engine.identifier,
                "runtime": os.getenv("INFERENCE_RUNTIME", "onnx"),
                "profile": getattr(engine, "profile", None),
                "metadata_source": getattr(engine, "metadata_source", None)}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        try:
            if file.content_type not in ("image/jpeg", "image/png") or not (file.filename or "").lower().endswith((".jpg", ".jpeg", ".png")):
                raise HTTPException(415, "Choose a JPG or PNG image")
            if app.state.predictor is None:
                raise HTTPException(503, app.state.model_error or "Model is not ready; retry shortly")
            if not admission.acquire(blocking=False):
                raise HTTPException(429, "The prediction service is busy. Please try again shortly.")
            try:
                payload = await file.read(MAX_BYTES + 1)
                if not payload:
                    raise HTTPException(422, "The image file is empty")
                if len(payload) > MAX_BYTES:
                    raise HTTPException(413, "Image exceeds the 10 MB limit")
                image = await run_in_threadpool(decode_image, payload)
                try:
                    return await run_in_threadpool(app.state.predictor.predict, image)
                finally:
                    image.close()
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Inference failed")
                raise HTTPException(503, "Prediction failed; please retry or contact the service maintainer") from exc
            finally:
                admission.release()
        finally:
            await file.close()

    return app


app = create_app()
