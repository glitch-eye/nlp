"""LLM adapter (Member 2).

Single place that knows how to talk to a model, so no other module depends on a
particular SDK/endpoint.  ``assistant.create_llm_client()`` forwards the selected provider/model to
``LLMClient``; unset values are read from the environment:

    LLM_PROVIDER     mock (default) | openai | api | auto   (auto = real only if a key is set)
    LLM_MODEL        model id, default gpt-4o-mini
    OPENAI_API_KEY   credential; never logged, never written to output
    OPENAI_BASE_URL  OpenAI-compatible endpoint, default https://api.openai.com/v1
    LLM_TIMEOUT      seconds, default 30

Interface (same as ``BaseLLMClient`` in assistant.py):
    generate(prompt, question="", chunks=None, semantic=None) -> str
    model_name   -> string recorded in output/answer.txt
The model_name states honestly when an answer did NOT come from the requested model
(mock mode or API failure), and ``last_error`` says why.
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
    """Provider-agnostic client: offline mock by default, OpenAI-compatible API on request."""

    def __init__(self, provider=None, model=None, api_key=None, base_url=None, timeout=None):
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "mock")).strip().lower()
        self._requested_model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = float(timeout or os.environ.get("LLM_TIMEOUT", 30))
        self._mock = _make_mock()
        self.last_error = None
        self.generation_count = 0
        self.call_history = []
        self.last_call = {}

        self.use_api = self.provider in REAL_PROVIDERS or (
            self.provider == "auto" and bool(self._api_key)
        )
        self.model_name = self._requested_model if self.use_api else self._mock.model_name

    # -- public ---------------------------------------------------------------
    def generate(self, prompt, question="", chunks=None, semantic=None):
        started = time.perf_counter()
        self.generation_count += 1
        self.last_error = None
        try:
            if not self.use_api:
                self.model_name = self._mock.model_name
                return self._mock.generate(prompt, question=question, chunks=chunks, semantic=semantic)
            if not self._api_key:
                return self._fallback("OPENAI_API_KEY not set", prompt, question, chunks, semantic)
            try:
                text = self._call_api(prompt)
            except Exception as exc:
                # Record only the exception class; exception messages may contain request details.
                return self._fallback(type(exc).__name__, prompt, question, chunks, semantic)
            self.model_name = self._requested_model
            return text
        finally:
            call = {
                "provider": self.provider,
                "requested_model": self._requested_model,
                "model_name": self.model_name,
                "used_api": bool(self.use_api and self.last_error is None),
                "fallback": bool(self.last_error),
                "error": self.last_error,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
            self.last_call = call
            self.call_history.append(dict(call))

    # -- internals ------------------------------------------------------------
    def _fallback(self, reason, prompt, question, chunks, semantic):
        self.last_error = reason
        self.model_name = "{} [FALLBACK: {}, requested {}]".format(
            self._mock.model_name, reason, self._requested_model
        )
        return self._mock.generate(prompt, question=question, chunks=chunks, semantic=semantic)

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
