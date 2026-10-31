"""Experimental hybrid app. Run separately on port 8001, never replace backend.main."""

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

# Reuse baseline validation only; its app lifespan is NOT started by this import.
from backend.main import BodyLimitMiddleware, MAX_BYTES, decode_image
from tomatocare.hybrid_onnx import load_hybrid_predictor

logger = logging.getLogger(__name__)


def create_app(predictor_factory=load_hybrid_predictor):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor, app.state.error = None, None
        try:
            app.state.predictor = await run_in_threadpool(predictor_factory)
        except Exception as exc:
            app.state.error = f"Hybrid model unavailable: {exc}"
            logger.error("%s", app.state.error)
        yield
        app.state.predictor = None

    app = FastAPI(title="TomatoCare Hybrid Experimental", version="0.5.0", lifespan=lifespan)
    origins = [o.strip() for o in os.getenv("HYBRID_CORS_ORIGINS", "").split(",") if o.strip()]
    if "*" in origins:
        raise ValueError("Use explicit HYBRID_CORS_ORIGINS")
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    busy = threading.Lock()

    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health")
    async def health():
        if app.state.predictor is None:
            return JSONResponse({"status": "unavailable", "detail": app.state.error}, status_code=503)
        return {"status": "ready", "experimental": True, "model_version": app.state.predictor.model_version,
                "runtime": "onnx-cpu", "sha256_verified": True}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        try:
            if file.content_type not in ("image/jpeg", "image/png") or not (file.filename or "").lower().endswith((".jpg", ".jpeg", ".png")):
                raise HTTPException(415, "Choose a valid JPG or PNG image")
            if app.state.predictor is None:
                raise HTTPException(503, app.state.error or "Hybrid model not ready")
            if not busy.acquire(blocking=False):
                raise HTTPException(429, "Hybrid service is busy; retry shortly")
            try:
                payload = await file.read(MAX_BYTES + 1)
                if not payload or len(payload) > MAX_BYTES:
                    raise HTTPException(413 if payload else 422, "Image exceeds 10 MiB" if payload else "Empty image file")
                image = await run_in_threadpool(decode_image, payload)
                try:
                    if max(image.size) > 8192:
                        raise HTTPException(422, "Image side exceeds experimental 8192-pixel limit")
                    return await run_in_threadpool(app.state.predictor.predict, image)
                finally:
                    image.close()
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Hybrid inference failed")
                raise HTTPException(503, "Hybrid inference failed; inspect model/manifest compatibility") from exc
            finally:
                busy.release()
        finally:
            await file.close()

    return app


app = create_app()
