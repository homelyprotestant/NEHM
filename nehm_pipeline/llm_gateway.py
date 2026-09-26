"""
HKU Azure-style LLM gateway client.

Speaks the **Azure OpenAI** wire format directly via ``requests``, so it works regardless of
whether ``openai`` is the legacy 0.x SDK or the modern 1.x SDK. Mirrors the configuration
shown in ``api-gateway-test.ipynb``::

    api_type    = "azure"
    api_base    = "https://api.hku.hk"
    api_version = "2024-06-01"
    engine      = "gpt-5.5"   # or "gpt-4o", "chatgpt-4", etc.

Endpoint URL pattern::

    POST {api_base}/openai/deployments/{engine}/chat/completions?api-version={api_version}
    Headers: api-key: <key>, Content-Type: application/json

Features:
- Strict JSON mode (``response_format={"type":"json_object"}``) with a graceful fallback
  prompt if the deployment rejects the parameter.
- Retries with jittered exponential backoff on 429 / 5xx / connection errors.
- Cumulative token / cost meter (prompt, completion, total) accessible via
  :attr:`HKUGateway.usage`.
- Thread-safe: a single client instance can be shared across worker threads.

Example
-------
>>> gw = HKUGateway(engine="gpt-5.5", api_key=os.environ["HKU_API_KEY"])
>>> reply = gw.chat_json(
...     system="You are a helpful assistant.",
...     user="Return JSON: {\"hello\": \"world\"}",
... )
>>> print(reply)
{'hello': 'world'}
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

__all__ = ["HKUGateway", "HKUGatewayError", "GatewayUsage"]

_LOG = logging.getLogger(__name__)


class HKUGatewayError(RuntimeError):
    """Raised when the gateway returns a non-recoverable error or repeated failures."""


@dataclass
class GatewayUsage:
    """Cumulative token usage; updated atomically across threads."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    retry_count: int = 0
    error_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, *, prompt: int, completion: int, total: int) -> None:
        with self._lock:
            self.prompt_tokens += int(prompt)
            self.completion_tokens += int(completion)
            self.total_tokens += int(total)
            self.request_count += 1

    def add_retry(self) -> None:
        with self._lock:
            self.retry_count += 1

    def add_error(self) -> None:
        with self._lock:
            self.error_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "request_count": self.request_count,
                "retry_count": self.retry_count,
                "error_count": self.error_count,
            }


def _resolve_api_key(explicit: Optional[str], env_var: str) -> str:
    if explicit:
        return explicit
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    cfg = Path.home() / ".nehm" / "hku_api_key"
    if cfg.is_file():
        return cfg.read_text(encoding="utf-8").strip()
    raise HKUGatewayError(
        f"No API key found. Set env var {env_var}, pass api_key=..., "
        f"or write the key to {cfg}."
    )


class HKUGateway:
    """Synchronous HKU/Azure-style chat completions client."""

    def __init__(
        self,
        *,
        engine: str = "gpt-5.5",
        api_base: str = "https://api.hku.hk",
        api_version: str = "2024-06-01",
        api_key: Optional[str] = None,
        api_key_env: str = "HKU_API_KEY",
        timeout_s: float = 90.0,
        max_retries: int = 6,
        backoff_base: float = 2.0,
        backoff_cap: float = 60.0,
        min_request_interval_s: float = 0.0,
    ) -> None:
        self.engine = str(engine)
        self.api_base = str(api_base).rstrip("/")
        self.api_version = str(api_version)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self.min_request_interval_s = float(min_request_interval_s)
        self._api_key = _resolve_api_key(api_key, api_key_env)
        self._session = requests.Session()
        self._session.headers.update(
            {"api-key": self._api_key, "Content-Type": "application/json"}
        )
        self._rate_lock = threading.Lock()
        self._last_request_t = 0.0
        self._json_mode_supported: Optional[bool] = None
        # Newer reasoning-style models (gpt-5.x, o1, ...) reject ``max_tokens`` and require
        # ``max_completion_tokens``. None = unknown; True/False are sticky after first probe.
        self._needs_max_completion_tokens: Optional[bool] = None
        self.usage = GatewayUsage()

    def _url(self, engine: Optional[str]) -> str:
        eng = engine or self.engine
        return (
            f"{self.api_base}/openai/deployments/{eng}/chat/completions"
            f"?api-version={self.api_version}"
        )

    def _throttle(self) -> None:
        if self.min_request_interval_s <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = self.min_request_interval_s - (now - self._last_request_t)
            if wait > 0:
                time.sleep(wait)
            self._last_request_t = time.monotonic()

    def _backoff_sleep(self, attempt: int, retry_after: Optional[float] = None) -> None:
        if retry_after is not None and retry_after > 0:
            time.sleep(min(float(retry_after), self.backoff_cap))
            return
        delay = min(self.backoff_base ** attempt, self.backoff_cap)
        jitter = random.uniform(0, delay * 0.25)
        time.sleep(delay + jitter)

    @staticmethod
    def _retry_after(resp: requests.Response) -> Optional[float]:
        ra = resp.headers.get("Retry-After")
        if not ra:
            return None
        try:
            return float(ra)
        except ValueError:
            return None

    def chat(
        self,
        *,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1500,
        engine: Optional[str] = None,
        response_format: Optional[dict] = None,
        extra_params: Optional[dict] = None,
    ) -> dict:
        """Low-level: returns the full parsed JSON response from the gateway."""
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": float(temperature),
        }
        # Pick the right token-budget parameter for this deployment. Sticky once probed.
        token_param = "max_completion_tokens" if self._needs_max_completion_tokens else "max_tokens"
        body[token_param] = int(max_tokens)
        if response_format is not None:
            body["response_format"] = response_format
        if extra_params:
            body.update(extra_params)

        url = self._url(engine)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.post(url, json=body, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_err = exc
                if attempt >= self.max_retries:
                    self.usage.add_error()
                    raise HKUGatewayError(f"Network error after {attempt} retries: {exc}") from exc
                self.usage.add_retry()
                _LOG.warning("HKU gateway: network error %s, retrying (attempt %d)", exc, attempt + 1)
                self._backoff_sleep(attempt)
                continue

            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError as exc:
                    self.usage.add_error()
                    raise HKUGatewayError(f"Non-JSON response: {resp.text[:300]}") from exc
                u = payload.get("usage") or {}
                self.usage.add(
                    prompt=int(u.get("prompt_tokens", 0) or 0),
                    completion=int(u.get("completion_tokens", 0) or 0),
                    total=int(u.get("total_tokens", 0) or 0),
                )
                return payload

            if resp.status_code == 400:
                try:
                    err_body = resp.json()
                    err_msg = json.dumps(err_body).lower()
                except Exception:
                    err_msg = resp.text.lower()

                # Newer reasoning models reject 'max_tokens'; swap to 'max_completion_tokens'.
                if (
                    "max_tokens" in err_msg
                    and "max_completion_tokens" in err_msg
                    and not self._needs_max_completion_tokens
                ):
                    _LOG.warning(
                        "HKU gateway: deployment requires max_completion_tokens; switching parameter"
                    )
                    self._needs_max_completion_tokens = True
                    val = body.pop("max_tokens", int(max_tokens))
                    body["max_completion_tokens"] = int(val)
                    continue

                if response_format is not None and ("response_format" in err_msg or "json_object" in err_msg):
                    _LOG.warning(
                        "HKU gateway: deployment rejected response_format; falling back to plain mode"
                    )
                    self._json_mode_supported = False
                    body.pop("response_format", None)
                    continue

                # Some reasoning deployments also reject custom temperatures; honor that.
                if "temperature" in err_msg and "unsupported" in err_msg:
                    _LOG.warning(
                        "HKU gateway: deployment rejected custom temperature; using default"
                    )
                    body.pop("temperature", None)
                    continue

            if resp.status_code in (408, 425, 429, 500, 502, 503, 504):
                last_err = HKUGatewayError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                if attempt >= self.max_retries:
                    self.usage.add_error()
                    raise last_err
                self.usage.add_retry()
                _LOG.warning(
                    "HKU gateway: HTTP %d, retrying (attempt %d/%d)",
                    resp.status_code,
                    attempt + 1,
                    self.max_retries,
                )
                self._backoff_sleep(attempt, self._retry_after(resp))
                continue

            self.usage.add_error()
            raise HKUGatewayError(
                f"HKU gateway returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        self.usage.add_error()
        raise HKUGatewayError(f"Exhausted retries; last error: {last_err}")

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        engine: Optional[str] = None,
        try_native_json_mode: bool = True,
    ) -> dict:
        """High-level: returns a parsed JSON object from the assistant content.

        Tries native ``response_format={"type":"json_object"}`` first; if the deployment
        rejects it, retries without and relies on prompt-level JSON discipline.
        """
        rf = {"type": "json_object"} if (try_native_json_mode and self._json_mode_supported is not False) else None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            engine=engine,
            response_format=rf,
        )
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise HKUGatewayError(f"Unexpected response shape: {payload}") from exc

        text = (content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise HKUGatewayError(f"Could not parse JSON from response: {text[:500]}")
