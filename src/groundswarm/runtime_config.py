"""Groundswarm-side selection of which Worker backend to construct and,
when it's 'openclaw', which execution harness (see ops/openclaw_config.py)
the Gateway should be configured for before Groundswarm starts it.

Stdlib-only, env-var-fallback-at-construction-time -- matching
OpenClawGatewayManager's pattern, not OllamaClient's import-time
DEFAULT_HOST, so tests can override os.environ per-test rather than at
import. No file-based config format is introduced: nothing in this
codebase has ever needed one (pyproject.toml has zero dependencies), and
every existing consumer of this config is a Python script that can pass
literal kwargs directly.

Deliberately does not decide HOW harness prerequisites (e.g. which OpenAI
model id) get applied to OpenClaw's own config -- harness_options is a
generic passthrough bag forwarded verbatim to ops/openclaw_config.py's
apply_harness(), so adding a future harness there never requires a change
here. It never carries a secret value -- see openclaw_config.py's
SecretRef-based handling of the codex harness's API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

WORKER_BACKENDS = ("ollama", "openclaw")
OPENCLAW_HARNESSES = ("direct", "codex", "claude_cli")

_HARNESS_OPTION_ENV_PREFIX = "GROUNDSWARM_HARNESS_OPTION_"


class InvalidRuntimeConfig(RuntimeError):
    """Raised when worker_backend/openclaw_harness is not a recognized value."""


@dataclass
class RuntimeConfig:
    worker_backend: str = "ollama"      # "ollama" | "openclaw"
    openclaw_harness: str = "direct"    # "direct" | "codex" | "claude_cli" -- only read when worker_backend == "openclaw"
    # Generic passthrough for harness-specific prerequisites (e.g.
    # {"openai_model": "openai/gpt-5.1"}) forwarded verbatim to
    # ops.openclaw_config.apply_harness()'s kwargs. Never a secret value.
    harness_options: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "RuntimeConfig":
        e = env if env is not None else os.environ
        cfg = cls(
            worker_backend=e.get("GROUNDSWARM_WORKER_BACKEND", "ollama").strip().lower(),
            openclaw_harness=e.get("GROUNDSWARM_OPENCLAW_HARNESS", "direct").strip().lower(),
            harness_options=collect_harness_options(e),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.worker_backend not in WORKER_BACKENDS:
            raise InvalidRuntimeConfig(
                f"unknown worker_backend {self.worker_backend!r}; must be one of {WORKER_BACKENDS}"
            )
        if self.worker_backend == "openclaw" and self.openclaw_harness not in OPENCLAW_HARNESSES:
            raise InvalidRuntimeConfig(
                f"unknown openclaw_harness {self.openclaw_harness!r}; must be one of {OPENCLAW_HARNESSES}"
            )


def collect_harness_options(env: dict[str, str]) -> dict[str, str]:
    # e.g. GROUNDSWARM_HARNESS_OPTION_OPENAI_MODEL=openai/gpt-5.1 -> {"openai_model": "openai/gpt-5.1"}
    return {
        key[len(_HARNESS_OPTION_ENV_PREFIX):].lower(): value
        for key, value in env.items()
        if key.startswith(_HARNESS_OPTION_ENV_PREFIX)
    }
