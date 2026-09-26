"""Local configuration for PLM_Agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
ENV_PATH = PACKAGE_ROOT / ".env"


def load_env(path: Path | None = None) -> Path:
    """Load PLM_Agent's private environment without overriding shell variables."""
    env_path = Path(path or ENV_PATH).expanduser().resolve()
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    return env_path


def ollama_base_url() -> str:
    base = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    ).rstrip("/")
    return base
