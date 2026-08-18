"""Groundswarm-owned mutation of a real OpenClaw Gateway's own `openclaw.json`
config -- the mechanism that actually selects which agent-execution harness
(local Ollama, a real Codex CLI subprocess, or a real Claude Code CLI
subprocess) a running Gateway uses. Per investigation of OpenClaw's own
source (src/agents/openai-routing.ts, src/agents/harness/policy.ts,
extensions/anthropic/cli-backend.ts), this routing is entirely config-driven
on OpenClaw's side; OpenClawWorker/OpenClawGatewayManager need no changes to
benefit from it.

Deliberately a thin wrapper around OpenClaw's own `openclaw config patch
--stdin` CLI rather than a hand-rolled deep-merge/backup implementation --
that CLI already provides file locking, atomic writes, schema validation,
and 5-deep backup rotation (src/config/mutate.ts, backup-rotation.ts,
io.write.ts -- this is what already produces the .bak.1-.bak.4 files in
~/.openclaw-dev/, not a manual artifact). This module's only jobs are: build
the right patch, skip the subprocess call entirely when nothing would
change (idempotency), and use OpenClaw's own SecretRef indirection so a real
API key is never written to disk in plaintext.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .openclaw_manager import OpenClawGatewayManager

Harness = Literal["direct", "codex", "claude_cli"]
OpenAiAuthMode = Literal["api-key", "subscription"]

DEFAULT_CLAUDE_MODEL = "claude-cli/claude-opus-5"
DEFAULT_OLLAMA_MODEL = "ollama/dolphin3:latest"


class OpenClawConfigError(RuntimeError):
    """Raised when a config mutation is rejected or the CLI call fails."""


class OpenClawConfigGatewayRunningError(OpenClawConfigError):
    """Raised when apply_harness() is called while the Gateway is still up.

    Config mutation only ever happens while the Gateway is confirmed
    stopped -- a harness/plugin change needs a restart to take effect
    anyway (observed live: a stale "plugin registration is missing" error
    from a prepared run that predates the config change), so mutating
    underneath a running process would just be discarded work.
    """


class OpenClawConfigValidationError(OpenClawConfigError):
    """Raised when `config patch --dry-run` rejects the computed patch."""


@dataclass(frozen=True)
class HarnessApplyResult:
    harness: Harness
    changed: bool
    config_path: Path
    restart_required: bool  # always True when changed=True; no hot-reload classification attempted


def apply_harness(
    harness: Harness,
    gateway_manager: OpenClawGatewayManager,
    *,
    agent_id: str = "dev",
    openai_auth_mode: OpenAiAuthMode = "api-key",
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_model: str | None = None,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    timeout_s: float = 600.0,
) -> HarnessApplyResult:
    """Configures a stopped OpenClaw Gateway's `openclaw.json` so its next
    start() executes the requested harness. Idempotent: if the live config
    already matches, no subprocess is run at all -- always call this for
    every worker_backend=="openclaw" run, "direct" included, rather than
    skipping the call for "direct"; a skip would let a Gateway previously
    switched to codex/claude_cli stay that way forever on a later "direct"
    run since nothing would ever revert it.

    Never writes a plaintext secret -- when openai_auth_mode=="api-key" (the
    default), the codex harness's API key is stored as an OpenClaw SecretRef
    (`{"source": "env", ...}`) that OpenClaw itself resolves from the named
    environment variable at call time; the real key value only needs to be
    set in this process's own environment, since
    OpenClawGatewayManager.start() already forwards os.environ to the
    spawned Gateway subprocess.

    openai_auth_mode=="subscription" skips the apiKey patch entirely --
    `config patch --dry-run` eagerly resolves every SecretRef it's given
    (confirmed live: it rejects the patch outright with "Environment
    variable ... is missing or empty" if openai_api_key_env isn't set, even
    though nothing about the actual codex CLI run needs it), so a
    subscription-only host (`codex login` with a ChatGPT account, no
    OPENAI_API_KEY at all -- OpenClaw's own auth-requirement matrix accepts
    a "chatgpt" account whenever authRequirement is left unset, which this
    patch never sets) could never get a patch applied at all under
    "api-key"'s always-include-the-ref behavior.
    """
    if gateway_manager.is_available():
        raise OpenClawConfigGatewayRunningError(
            "apply_harness() requires a stopped Gateway -- a harness/plugin change "
            "needs a fresh start to take effect. Call gateway_manager.stop() first."
        )

    patch = _patch_for(
        harness,
        agent_id=agent_id,
        openai_auth_mode=openai_auth_mode,
        openai_api_key_env=openai_api_key_env,
        openai_model=openai_model,
        claude_model=claude_model,
        ollama_model=ollama_model,
    )

    config_path = gateway_manager.state_dir / "openclaw.json"
    if _already_satisfied(config_path, patch):
        return HarnessApplyResult(harness=harness, changed=False, config_path=config_path, restart_required=False)

    payload = json.dumps(patch)

    dry_run = _run_config_cli(
        gateway_manager.repo_path,
        ["patch", "--stdin", "--dry-run", "--json"],
        stdin=payload,
        timeout_s=timeout_s,
    )
    _check_dry_run(dry_run)

    applied = _run_config_cli(
        gateway_manager.repo_path,
        ["patch", "--stdin"],
        stdin=payload,
        timeout_s=timeout_s,
    )
    if applied.returncode != 0:
        raise OpenClawConfigError(f"config patch failed (exit {applied.returncode}): {applied.stderr}")

    return HarnessApplyResult(harness=harness, changed=True, config_path=config_path, restart_required=True)


def _patch_for(
    harness: Harness,
    *,
    agent_id: str,
    openai_auth_mode: OpenAiAuthMode,
    openai_api_key_env: str,
    openai_model: str | None,
    claude_model: str,
    ollama_model: str,
) -> dict:
    if harness == "codex":
        if not openai_model:
            raise ValueError("codex harness requires openai_model (a catalog-known 'openai/<model>' ref)")
        patch = {
            "plugins": {"entries": {"codex": {"enabled": True}}},
            "agents": {"entries": {agent_id: {"model": openai_model}}},
        }
        if openai_auth_mode == "api-key":
            patch["models"] = {"providers": {"openai": {
                "apiKey": {"source": "env", "provider": "default", "id": openai_api_key_env},
            }}}
        return patch
    if harness == "claude_cli":
        return {"agents": {"entries": {agent_id: {"model": claude_model}}}}
    if harness == "direct":
        return {"agents": {"entries": {agent_id: {"model": ollama_model}}}}
    raise ValueError(f"unknown harness {harness!r}")


def _already_satisfied(config_path: Path, patch: dict) -> bool:
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = {}
    return _leaves_match(existing, patch)


def _leaves_match(existing: object, patch: object) -> bool:
    if not isinstance(patch, dict):
        return existing == patch
    if not isinstance(existing, dict):
        return False
    return all(_leaves_match(existing.get(key), value) for key, value in patch.items())


def _run_config_cli(repo_path: Path, args: list[str], *, stdin: str, timeout_s: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", "scripts/run-node.mjs", "--dev", "config", *args],
        cwd=str(repo_path),
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_s,
        check=False,
    )


def _check_dry_run(result: subprocess.CompletedProcess) -> None:
    if result.returncode != 0:
        raise OpenClawConfigValidationError(
            f"config patch --dry-run failed (exit {result.returncode}): {result.stderr}"
        )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise OpenClawConfigValidationError(
            f"config patch --dry-run returned non-JSON output: {result.stdout!r}"
        ) from exc
    if not parsed.get("ok"):
        raise OpenClawConfigValidationError(f"config patch --dry-run rejected the patch: {parsed}")