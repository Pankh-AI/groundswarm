"""The single construction point for a Worker, selected by RuntimeConfig
instead of scenario scripts hand-constructing OllamaSimWorker/OpenClawWorker
directly. For the "openclaw" backend this owns the full lifecycle around a
real Gateway: harness config mutation (ops/openclaw_config.py) strictly
before OpenClawGatewayManager.start(), teardown on exit regardless of how
the `with` block exits.
"""
from __future__ import annotations

import contextlib
import os
from typing import Iterator

from ..ops.openclaw_manager import OpenClawGatewayError, OpenClawGatewayManager
from ..runtime_config import RuntimeConfig
from .base import Worker
from .openclaw_worker import OpenClawWorker
from .sim_worker import OllamaSimWorker


@contextlib.contextmanager
def worker_session(
    config: RuntimeConfig | None = None,
    *,
    manager: OpenClawGatewayManager | None = None,
    gateway_token: str | None = None,
) -> Iterator[Worker]:
    """Yields a Worker matching `config`. For "ollama" this is a
    lifecycle-free wrapper (no process involved). For "openclaw" it
    configures the Gateway's harness (ops.openclaw_config.apply_harness,
    called unconditionally -- including for openclaw_harness == "direct" --
    so a Gateway previously left on codex/claude_cli is always reverted
    rather than only reverted when something happens to ask for it),
    starts the Gateway, yields an OpenClawWorker against it, and stops the
    Gateway on exit.
    """
    cfg = config or RuntimeConfig.from_env()
    cfg.validate()

    if cfg.worker_backend == "ollama":
        yield OllamaSimWorker()
        return

    gw = manager or OpenClawGatewayManager()

    # Deferred import: an "ollama" caller (the common case) never needs
    # ops.openclaw_config to exist at all.
    from ..ops.openclaw_config import apply_harness

    apply_harness(cfg.openclaw_harness, gw, **cfg.harness_options)

    token = gateway_token or os.environ.get("OPENCLAW_GATEWAY_TOKEN", "").strip()
    if not token:
        raise OpenClawGatewayError(
            "OPENCLAW_GATEWAY_TOKEN is not set. Export the dev profile's gateway.auth.token first."
        )

    with gw:
        yield OpenClawWorker(base_url=gw.base_url, token=token)
