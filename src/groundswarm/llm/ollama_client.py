"""Thin stdlib-only client for a local Ollama server."""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "dolphin3:latest"


class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama server can't be reached."""


@dataclass
class OllamaClient:
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    timeout_s: float = 120.0

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def generate(self, prompt: str, *, system: str | None = None,
                  model: str | None = None, json_mode: bool = False) -> str:
        """Single-shot generation. Raises OllamaUnavailable on any transport failure."""
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise OllamaUnavailable(f"could not reach Ollama at {self.host}: {exc}") from exc
        return data.get("response", "")
