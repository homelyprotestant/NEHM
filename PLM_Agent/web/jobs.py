"""Single-worker queue for long-running uploaded-image interpretations."""

from __future__ import annotations

import json
import queue
import secrets
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PLM_Agent.upload_inference import PLMUploadInference


class JobNotFound(KeyError):
    pass


class JobAccessDenied(PermissionError):
    pass


class JobCancelled(RuntimeError):
    pass


class JobTimedOut(RuntimeError):
    pass


@dataclass
class UploadJob:
    id: str
    token: str
    original_filename: str
    image_path: Path
    result_path: Path
    coordinator: str
    created_at: float
    expires_at: float
    status: str = "queued"
    stage: str = "queued"
    percent: int = 0
    message: str = "Waiting for the PLM inference engine."
    result: dict[str, Any] | None = None
    error: str | None = None
    cancellation_requested: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("token", None)
        data.pop("result", None)
        data["image_path"] = self.image_path.name
        data["result_path"] = self.result_path.name
        return data


class UploadJobManager:
    def __init__(
        self,
        engine: "PLMUploadInference",
        *,
        data_dir: Path,
        result_ttl_seconds: int = 86_400,
        max_queue_size: int = 10,
        job_timeout_seconds: int = 1_800,
    ) -> None:
        self.engine = engine
        self.data_dir = Path(data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.result_dir = self.data_dir / "results"
        self.job_dir = self.data_dir / "jobs"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.result_ttl_seconds = int(result_ttl_seconds)
        self.job_timeout_seconds = int(job_timeout_seconds)
        self._jobs: dict[str, UploadJob] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue_size)
        self._worker: threading.Thread | None = None
        self._stopping = False

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self.cleanup_orphaned_files()
        self._load_persisted_jobs()
        self._stopping = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="plm-upload-worker",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._stopping = True
        with suppress(queue.Full):
            self._queue.put_nowait(None)
        if self._worker:
            self._worker.join(timeout=5)

    def submit(
        self,
        *,
        image_bytes: bytes,
        original_filename: str,
        extension: str,
        coordinator: str,
    ) -> UploadJob:
        self.cleanup_expired()
        job_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(24)
        image_path = self.upload_dir / f"{job_id}{extension}"
        result_path = self.result_dir / f"{job_id}.json"
        image_path.write_bytes(image_bytes)
        now = time.time()
        job = UploadJob(
            id=job_id,
            token=token,
            original_filename=original_filename,
            image_path=image_path,
            result_path=result_path,
            coordinator=coordinator,
            created_at=now,
            expires_at=now + self.result_ttl_seconds,
        )
        self._append_event(job)
        with self._lock:
            self._jobs[job_id] = job
        try:
            self._queue.put_nowait(job_id)
        except queue.Full:
            image_path.unlink(missing_ok=True)
            with self._lock:
                self._jobs.pop(job_id, None)
            raise RuntimeError("The local inference queue is full.")
        return job

    def get(self, job_id: str, token: str) -> UploadJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        if not secrets.compare_digest(job.token, token):
            raise JobAccessDenied(job_id)
        return job

    def cancel(self, job_id: str, token: str) -> UploadJob:
        job = self.get(job_id, token)
        with self._lock:
            job.cancellation_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.message = "Interpretation cancelled."
                job.percent = 100
                job.image_path.unlink(missing_ok=True)
            self._append_event(job)
        return job

    def cleanup_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                job_id
                for job_id, job in self._jobs.items()
                if job.expires_at <= now
            ]
            for job_id in expired:
                job = self._jobs.pop(job_id)
                job.image_path.unlink(missing_ok=True)
                job.result_path.unlink(missing_ok=True)
                (self.job_dir / f"{job_id}.json").unlink(missing_ok=True)

    def cleanup_orphaned_files(self) -> None:
        now = time.time()
        limits = (
            (self.upload_dir, self.job_timeout_seconds),
            (self.result_dir, self.result_ttl_seconds),
            (self.job_dir, self.result_ttl_seconds),
        )
        for directory, max_age in limits:
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                with suppress(OSError):
                    if now - path.stat().st_mtime >= max_age:
                        path.unlink(missing_ok=True)

    def _ensure_within_timeout(self, job: UploadJob) -> None:
        if time.time() - job.created_at > self.job_timeout_seconds:
            raise JobTimedOut(
                f"Interpretation exceeded the {self.job_timeout_seconds}-second timeout."
            )

    def _append_event(self, job: UploadJob) -> None:
        job.events.append(
            {
                "index": len(job.events),
                "status": job.status,
                "stage": job.stage,
                "percent": job.percent,
                "message": job.message,
                "timestamp": time.time(),
            }
        )
        self._persist(job)

    def _persist(self, job: UploadJob) -> None:
        payload = asdict(job)
        payload["image_path"] = job.image_path.name
        payload["result_path"] = job.result_path.name
        payload["result"] = None
        path = self.job_dir / f"{job.id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(path)

    def _load_persisted_jobs(self) -> None:
        now = time.time()
        for path in self.job_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if float(payload["expires_at"]) <= now:
                    path.unlink(missing_ok=True)
                    continue
                job = UploadJob(
                    id=str(payload["id"]),
                    token=str(payload["token"]),
                    original_filename=str(payload["original_filename"]),
                    image_path=self.upload_dir / Path(payload["image_path"]).name,
                    result_path=self.result_dir / Path(payload["result_path"]).name,
                    coordinator=str(payload["coordinator"]),
                    created_at=float(payload["created_at"]),
                    expires_at=float(payload["expires_at"]),
                    status=str(payload.get("status", "failed")),
                    stage=str(payload.get("stage", "failed")),
                    percent=int(payload.get("percent", 100)),
                    message=str(payload.get("message", "Recovered after restart.")),
                    error=payload.get("error"),
                    cancellation_requested=bool(
                        payload.get("cancellation_requested", False)
                    ),
                    events=list(payload.get("events") or []),
                )
                if job.status == "complete" and job.result_path.is_file():
                    job.result = json.loads(job.result_path.read_text(encoding="utf-8"))
                elif job.status in {"queued", "running"}:
                    job.status = "failed"
                    job.stage = "failed"
                    job.percent = 100
                    job.error = "Interpretation was interrupted by a server restart."
                    job.message = job.error
                    job.image_path.unlink(missing_ok=True)
                    self._append_event(job)
                self._jobs[job.id] = job
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                path.unlink(missing_ok=True)

    def _progress(self, job: UploadJob, stage: str, percent: int, message: str) -> None:
        with self._lock:
            if job.cancellation_requested:
                raise JobCancelled("Interpretation cancelled.")
            self._ensure_within_timeout(job)
            job.stage = stage
            job.percent = max(job.percent, min(100, int(percent)))
            job.message = message
            self._append_event(job)

    @staticmethod
    def _coordinator(choice: str) -> tuple[str, str]:
        if choice == "openai":
            return "openai", "gpt-5.1"
        raise ValueError(f"Unsupported coordinator: {choice!r}")

    def _worker_loop(self) -> None:
        while not self._stopping:
            job_id = self._queue.get()
            if job_id is None:
                return
            with self._lock:
                job = self._jobs.get(job_id)
            if job is None or job.status == "cancelled":
                continue
            try:
                self._ensure_within_timeout(job)
                backend, model = self._coordinator(job.coordinator)
                with self._lock:
                    job.status = "running"
                    job.stage = "validation"
                    job.percent = 5
                    job.message = "Validated image; preparing inference."
                    self._append_event(job)
                result = self.engine.describe_upload(
                    job.image_path,
                    job.result_path,
                    coordinator_backend=backend,
                    coordinator_model=model,
                    progress=lambda stage, percent, message: self._progress(
                        job, stage, percent, message
                    ),
                )
                with self._lock:
                    job.result = result
                    job.status = "complete"
                    job.stage = "complete"
                    job.percent = 100
                    job.message = "Microscopy interpretation complete."
                    self._append_event(job)
            except JobCancelled:
                with self._lock:
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    job.percent = 100
                    job.message = "Interpretation cancelled."
                    self._append_event(job)
            except JobTimedOut as exc:
                with self._lock:
                    job.status = "failed"
                    job.stage = "failed"
                    job.percent = 100
                    job.error = str(exc)
                    job.message = "Interpretation timed out."
                    self._append_event(job)
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.stage = "failed"
                    job.percent = 100
                    job.error = str(exc)
                    job.message = "Interpretation failed."
                    self._append_event(job)
            finally:
                job.image_path.unlink(missing_ok=True)
                if job.result is not None and not job.result_path.is_file():
                    job.result_path.write_text(
                        json.dumps(job.result, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

