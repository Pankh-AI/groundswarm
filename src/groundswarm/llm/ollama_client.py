"""Thin stdlib-only client for an Ollama server (local or remote/tunneled)."""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field

# Reads OLLAMA_HOST at import time so pointing at a remote instance (e.g. a
# Colab-hosted Ollama behind a cloudflared tunnel) is a one-line env var
# change, not a code edit -- `export OLLAMA_HOST=https://xxxx.trycloudflare.com`
# before running any scenario script.
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "dolphin3:latest"


class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama server can't be reached."""


@dataclass
class CallLog:
    """One record of a single generate() call, success or failure.

    prompt/completion token counts and perplexity come straight from Ollama's
    own OpenAI-compatible response (usage.*, choices[0].logprobs), not an
    estimate. perplexity is exp(-mean(token logprob)) over the generated
    tokens; None if the server returned no logprobs for that call.
    """
    label: str
    model: str
    prompt_chars: int
    success: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_s: float | None = None
    perplexity: float | None = None
    error: str | None = None


@dataclass
class OllamaClient:
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    timeout_s: float = 120.0
    # None leaves Ollama's own default sampling in place (representative of
    # real usage). Scenario scripts doing a code A/B comparison should pass
    # temperature=0.0 explicitly -- otherwise two runs of identical code
    # produce different claim sets purely from sampling noise, which is not
    # a code effect and will be misread as one. See ADR-013.
    temperature: float | None = None
    calls: list[CallLog] = field(default_factory=list, repr=False, compare=False)

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def generate(self, prompt: str, *, system: str | None = None,
                  model: str | None = None, json_mode: bool = False,
                  label: str = "unlabeled") -> str:
        """Single-shot generation. Raises OllamaUnavailable on any transport failure.

        Uses Ollama's OpenAI-compatible /v1/chat/completions endpoint (not the
        native /api/generate) specifically because it's the one that returns
        per-token logprobs, which /api/generate does not. Every call, success
        or failure, is appended to self.calls so callers can inspect real
        token usage, latency, and perplexity after the fact. `label`
        identifies which component made the call (e.g. "extract:worker_a",
        "judge:worker_b") for observability grouping; it has no effect on the
        request itself.
        """
        used_model = model or self.model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": used_model,
            "messages": messages,
            "logprobs": True,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            self.calls.append(CallLog(
                label=label, model=used_model, prompt_chars=len(prompt), success=False,
                duration_s=time.monotonic() - start, error=str(exc),
            ))
            raise OllamaUnavailable(f"could not reach Ollama at {self.host}: {exc}") from exc

        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})
        token_logprobs = [
            t["logprob"] for t in (choice.get("logprobs") or {}).get("content", []) or []
        ]
        perplexity = math.exp(-sum(token_logprobs) / len(token_logprobs)) if token_logprobs else None

        self.calls.append(CallLog(
            label=label, model=used_model, prompt_chars=len(prompt), success=True,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            duration_s=time.monotonic() - start,
            perplexity=perplexity,
        ))
        return content
