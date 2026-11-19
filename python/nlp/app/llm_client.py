"""LLM adapter (Member 2).

Single place that knows how to talk to a model, so no other module depends on a
particular SDK/endpoint. Configuration is read from environment or constructor.

Telemetry is intentionally metadata-only: prompts, questions, context and API keys
are never stored in call_history.
"""

import json
import os
import socket
import time
import urllib.error
import urllib.request

from nlp.app.prompt_builder import SENTINEL

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
REAL_PROVIDERS = ("openai", "api")
API_ERRORS = (
    urllib.error.URLError,
    socket.timeout,
    TimeoutError,
    OSError,
    KeyError,
    IndexError,
    ValueError,
)


SYSTEM_MESSAGE = (
    "Bạn là trợ lý môn học Xử lý Ngôn ngữ Tự nhiên (CO3085). "
    "Chỉ trả lời dựa trên CONTEXT được cung cấp và luôn trích dẫn [Nguồn: file / section]. "
    "Nếu CONTEXT không đủ thông tin, chỉ trả về đúng từ khoá " + SENTINEL + "."
)


def _make_mock():
    """Reuse the extractive mock from assistant.py (imported lazily: no import cycle)."""
    try:
        from nlp.app.assistant import MockLLMClient
        return MockLLMClient()
    except Exception:  # pragma: no cover - assistant.py unavailable
        return _MinimalMock()


class _MinimalMock:
    """Last-resort offline stand-in: never invents content, just reports insufficiency."""

    model_name = "mock-minimal-v1"

    def generate(self, prompt, question="", chunks=None, semantic=None):
        return SENTINEL


class LLMClient:
    """Provider-agnostic client with per-generation, privacy-safe telemetry."""

    def __init__(self, provider=None, model=None, api_key=None, base_url=None, timeout=None):
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "mock")).strip().lower()
        self._requested_model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = float(timeout or os.environ.get("LLM_TIMEOUT", 30))
        self._mock = _make_mock()
        self.last_error = None
        self.call_history = []
        self.generation_count = 0

        self.use_api = self.provider in REAL_PROVIDERS or (
            self.provider == "auto" and bool(self._api_key)
        )
        self.model_name = self._requested_model if self.use_api else self._mock.model_name

    @property
    def last_call(self):
        """Metadata for the most recent generation invocation, if any."""
        return self.call_history[-1] if self.call_history else None

    def _record_call(
        self, started, *, used_api, attempted_api, fallback=False, error=None
    ):
        self.generation_count += 1
        self.call_history.append({
            "provider": self.provider,
            "requested_model": self._requested_model,
            "model_name": self.model_name,
            "used_api": bool(used_api),
            "attempted_api": bool(attempted_api),
            "fallback": bool(fallback),
            "error": error,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        })

    # -- public ---------------------------------------------------------------
    def generate(self, prompt, question="", chunks=None, semantic=None):
        started = time.perf_counter()
        self.last_error = None

        if not self.use_api:
            self.model_name = self._mock.model_name
            try:
                return self._mock.generate(
                    prompt, question=question, chunks=chunks, semantic=semantic
                )
            finally:
                self._record_call(
                    started, used_api=False, attempted_api=False, fallback=False
                )

        if not self._api_key:
            try:
                return self._fallback(
                    "OPENAI_API_KEY not set", prompt, question, chunks, semantic
                )
            finally:
                self._record_call(
                    started, used_api=False, attempted_api=False,
                    fallback=True, error="OPENAI_API_KEY not set",
                )

        try:
            text = self._call_api(prompt)
        except API_ERRORS as exc:
            reason = type(exc).__name__
            try:
                return self._fallback(reason, prompt, question, chunks, semantic)
            finally:
                self._record_call(
                    started, used_api=False, attempted_api=True,
                    fallback=True, error=reason,
                )

        self.model_name = self._requested_model
        self._record_call(started, used_api=True, attempted_api=True)
        return text

    # -- internals ------------------------------------------------------------
    def _fallback(self, reason, prompt, question, chunks, semantic):
        self.last_error = reason
        self.model_name = "{} [FALLBACK: {}, requested {}]".format(
            self._mock.model_name, reason, self._requested_model
        )
        return self._mock.generate(
            prompt, question=question, chunks=chunks, semantic=semantic
        )

    def _call_api(self, prompt):
        payload = {
            "model": self._requested_model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            "{}/chat/completions".format(self.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(self._api_key),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()

    def __repr__(self):  # never expose the key
        return "LLMClient(provider={!r}, model={!r}, use_api={})".format(
            self.provider, self._requested_model, self.use_api
        )
