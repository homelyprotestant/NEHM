"""Direct OpenAI and local Ollama clients for PLM_Agent."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

import requests
from openai import OpenAI

from .config import load_env, ollama_base_url


class ModelNotFoundError(RuntimeError):
    pass


def _request_timeout() -> float:
    return float(os.getenv("PLM_OPENAI_TIMEOUT_SECONDS", "180"))


def _max_retries() -> int:
    return int(os.getenv("PLM_OPENAI_MAX_RETRIES", "2"))


def list_ollama_models() -> list[str]:
    load_env()
    response = requests.get(f"{ollama_base_url()}/api/tags", timeout=10)
    response.raise_for_status()
    return [str(item.get("name", "")) for item in response.json().get("models", [])]


def ollama_model_available(model: str) -> bool:
    try:
        names = list_ollama_models()
    except Exception:
        return False
    return model in names or any(name.startswith(f"{model}:") for name in names)


def _client(backend: str) -> OpenAI:
    load_env()
    if backend == "ollama":
        base = ollama_base_url()
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return OpenAI(
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=base,
            timeout=_request_timeout(),
            max_retries=_max_retries(),
        )
    if backend == "openai":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from the runtime environment")
        return OpenAI(
            api_key=api_key,
            timeout=_request_timeout(),
            max_retries=_max_retries(),
        )
    raise ValueError(f"Unsupported PLM_Agent backend: {backend!r}")


def _ensure_model(backend: str, model: str) -> None:
    if backend == "ollama" and not ollama_model_available(model):
        raise ModelNotFoundError(
            f"Ollama model {model!r} is unavailable; run: ollama pull {model}"
        )


def chat_text(
    prompt: str,
    *,
    backend: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    _ensure_model(backend, model)
    client = _client(backend)
    kwargs: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if backend == "ollama":
        kwargs.update(
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"think": False},
        )
    else:
        kwargs.update(max_completion_tokens=max_tokens)
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def chat_vision(
    prompt: str,
    image_path: Path,
    *,
    backend: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    _ensure_model(backend, model)
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    image_url = f"data:{mime_type};base64,{encoded}"
    kwargs: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    if backend == "ollama":
        kwargs.update(
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"think": False},
        )
    else:
        kwargs.update(max_completion_tokens=max_tokens)
    response = _client(backend).chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()
