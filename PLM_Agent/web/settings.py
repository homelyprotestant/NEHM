"""Configuration for the local PLM image web application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PLM_Agent.config import load_env


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().casefold()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class WebSettings:
    project_root: Path
    data_dir: Path
    static_dir: Path
    max_upload_bytes: int
    result_ttl_seconds: int
    max_queue_size: int
    job_timeout_seconds: int = 1_800
    max_image_pixels: int = 80_000_000
    public_prototype: bool = False

    @classmethod
    def from_env(cls) -> "WebSettings":
        load_env()
        package_root = Path(__file__).resolve().parents[1]
        project_root = Path(
            os.getenv("PLM_PROJECT_ROOT", str(package_root.parent))
        ).expanduser().resolve()
        data_dir = Path(
            os.getenv(
                "PLM_WEB_DATA_DIR",
                str(project_root / "NEHM_RESULTS" / "plm_web"),
            )
        ).expanduser().resolve()
        static_dir = Path(
            os.getenv(
                "PLM_WEB_STATIC_DIR",
                str(Path(__file__).resolve().parent / "static"),
            )
        ).expanduser().resolve()
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            static_dir=static_dir,
            max_upload_bytes=_env_int("PLM_WEB_MAX_UPLOAD_MB", 25) * 1024 * 1024,
            result_ttl_seconds=_env_int("PLM_WEB_RESULT_TTL_SECONDS", 86_400),
            max_queue_size=_env_int("PLM_WEB_MAX_QUEUE_SIZE", 10),
            job_timeout_seconds=_env_int("PLM_WEB_JOB_TIMEOUT_SECONDS", 1_800),
            max_image_pixels=_env_int("PLM_WEB_MAX_IMAGE_PIXELS", 80_000_000),
            public_prototype=_env_bool("PLM_WEB_PUBLIC_PROTOTYPE", False),
        )

    @property
    def artifacts(self):
        from PLM_Agent.upload_inference import UploadArtifacts

        return UploadArtifacts.from_project_root(self.project_root)

    def prepare(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "results").mkdir(parents=True, exist_ok=True)

