"""FastAPI entry point for the PLM arbitrary-image web application."""

from __future__ import annotations

import asyncio
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .jobs import JobAccessDenied, JobNotFound, UploadJobManager
from .settings import WebSettings

ALLOWED_COORDINATORS = {"openai"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF", "BMP"}


def _job_urls(job_id: str, token: str) -> dict[str, str]:
    query = f"?token={token}"
    return {
        "status_url": f"/api/jobs/{job_id}{query}",
        "events_url": f"/api/jobs/{job_id}/events{query}",
        "result_url": f"/api/jobs/{job_id}/result{query}",
        "download_url": f"/api/jobs/{job_id}/download{query}",
        "cancel_url": f"/api/jobs/{job_id}/cancel{query}",
    }


def _get_job(manager: UploadJobManager, job_id: str, token: str):
    try:
        return manager.get(job_id, token)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    except JobAccessDenied as exc:
        raise HTTPException(status_code=403, detail="Invalid job token.") from exc


def _normalize_image(data: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as source:
            if (source.format or "").upper() not in ALLOWED_FORMATS:
                raise HTTPException(status_code=415, detail="Unsupported image format.")
            source.load()
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="The upload is not a valid image.") from exc
    if image.width < 32 or image.height < 32:
        raise HTTPException(status_code=422, detail="Image must be at least 32×32 pixels.")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def create_app(
    *,
    settings: WebSettings | None = None,
    engine: Any | None = None,
) -> FastAPI:
    web_settings = settings or WebSettings.from_env()
    Image.MAX_IMAGE_PIXELS = web_settings.max_image_pixels

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        web_settings.prepare()
        active_engine = engine
        if active_engine is None:
            from PLM_Agent.upload_inference import PLMUploadInference

            active_engine = PLMUploadInference(web_settings.artifacts)
        manager = UploadJobManager(
            active_engine,
            data_dir=web_settings.data_dir,
            result_ttl_seconds=web_settings.result_ttl_seconds,
            max_queue_size=web_settings.max_queue_size,
            job_timeout_seconds=web_settings.job_timeout_seconds,
        )
        manager.start()
        application.state.engine = active_engine
        application.state.job_manager = manager
        application.state.settings = web_settings
        try:
            yield
        finally:
            manager.stop()

    application = FastAPI(
        title="PLM Arbitrary-Image Interpreter",
        version="0.1.0",
        description="Nearest-neighbor, vision, and dictionary-atom microscopy interpretation.",
        lifespan=lifespan,
        docs_url=None if web_settings.public_prototype else "/docs",
        redoc_url=None if web_settings.public_prototype else "/redoc",
    )

    @application.middleware("http")
    async def production_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data:; connect-src 'self'; "
            "script-src 'self'; style-src 'self'; base-uri 'self'; form-action 'self'; "
            "frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response
    application.mount(
        "/static",
        StaticFiles(directory=web_settings.static_dir, check_dir=False),
        name="static",
    )

    @application.get("/", include_in_schema=False)
    def homepage() -> FileResponse:
        return FileResponse(web_settings.static_dir / "index.html")

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": application.state.engine.artifact_status(),
            "coordinator": {"backend": "openai", "model": "gpt-5.1"},
            "openai_configured": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
        }

    @application.post("/api/jobs", status_code=202)
    async def submit_job(
        image: UploadFile = File(...),
        coordinator: str = Form("openai"),
    ) -> dict[str, Any]:
        if coordinator not in ALLOWED_COORDINATORS:
            raise HTTPException(status_code=422, detail="Unsupported coordinator.")
        max_bytes = web_settings.max_upload_bytes
        data = await image.read(max_bytes + 1)
        await image.close()
        if not data:
            raise HTTPException(status_code=422, detail="Choose an image to upload.")
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit.",
            )
        normalized = _normalize_image(data)
        try:
            job = application.state.job_manager.submit(
                image_bytes=normalized,
                original_filename=image.filename or "uploaded-image",
                extension=".png",
                coordinator=coordinator,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "job_id": job.id,
            "token": job.token,
            "status": job.status,
            **_job_urls(job.id, job.token),
        }

    @application.get("/api/jobs/{job_id}")
    def job_status(job_id: str, token: str) -> dict[str, Any]:
        job = _get_job(application.state.job_manager, job_id, token)
        payload = job.public()
        payload.update(_job_urls(job.id, token))
        return payload

    @application.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str, token: str) -> dict[str, Any]:
        job = _get_job(application.state.job_manager, job_id, token)
        if job.status == "failed":
            raise HTTPException(status_code=500, detail=job.error or "Job failed.")
        if job.status != "complete" or job.result is None:
            raise HTTPException(status_code=409, detail="Result is not ready.")
        return job.result

    @application.get("/api/jobs/{job_id}/download")
    def download_result(job_id: str, token: str) -> FileResponse:
        job = _get_job(application.state.job_manager, job_id, token)
        if job.status != "complete" or not job.result_path.is_file():
            raise HTTPException(status_code=409, detail="Result is not ready.")
        return FileResponse(
            job.result_path,
            media_type="application/json",
            filename=f"plm-{job.id}.json",
        )

    @application.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, token: str) -> dict[str, Any]:
        try:
            job = application.state.job_manager.cancel(job_id, token)
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc
        except JobAccessDenied as exc:
            raise HTTPException(status_code=403, detail="Invalid job token.") from exc
        return job.public()

    @application.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, token: str) -> StreamingResponse:
        job = _get_job(application.state.job_manager, job_id, token)

        async def stream():
            cursor = 0
            while True:
                current = _get_job(application.state.job_manager, job.id, token)
                while cursor < len(current.events):
                    event = current.events[cursor]
                    cursor += 1
                    yield f"data: {json.dumps(event)}\n\n"
                if current.status in {"complete", "failed", "cancelled"}:
                    return
                await asyncio.sleep(0.75)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "PLM_Agent.web.api:app",
        host=os.getenv("PLM_WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("PLM_WEB_PORT", "8080")),
        reload=False,
        access_log=not WebSettings.from_env().public_prototype,
    )

