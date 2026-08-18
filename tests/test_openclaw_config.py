import json
import subprocess

import pytest

from groundswarm.ops import openclaw_config
from groundswarm.ops.openclaw_config import (
    HarnessApplyResult,
    OpenClawConfigError,
    OpenClawConfigGatewayRunningError,
    OpenClawConfigValidationError,
    _patch_for,
    apply_harness,
)


class FakeGatewayManager:
    """Duck-typed stand-in for OpenClawGatewayManager -- apply_harness()
    only ever touches .is_available(), .repo_path, and .state_dir."""

    def __init__(self, tmp_path, *, available=False, existing_config=None):
        self.repo_path = tmp_path / "openclaw-repo"
        self.state_dir = tmp_path / "openclaw-dev"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._available = available
        (self.state_dir / "openclaw.json").write_text(
            json.dumps(existing_config or {}), encoding="utf-8"
        )

    def is_available(self):
        return self._available


def _forbid_subprocess(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run should not have been called")

    monkeypatch.setattr(openclaw_config.subprocess, "run", _boom)


def test_apply_harness_raises_when_gateway_running(tmp_path, monkeypatch):
    _forbid_subprocess(monkeypatch)
    manager = FakeGatewayManager(tmp_path, available=True)
    with pytest.raises(OpenClawConfigGatewayRunningError):
        apply_harness("direct", manager)


def test_apply_harness_is_a_noop_when_already_satisfied(tmp_path, monkeypatch):
    _forbid_subprocess(monkeypatch)
    existing = {"agents": {"entries": {"dev": {"model": "ollama/dolphin3:latest"}}}}
    manager = FakeGatewayManager(tmp_path, existing_config=existing)

    result = apply_harness("direct", manager)

    assert result == HarnessApplyResult(
        harness="direct", changed=False,
        config_path=manager.state_dir / "openclaw.json", restart_required=False,
    )


def test_apply_harness_applies_when_config_differs(tmp_path, monkeypatch):
    manager = FakeGatewayManager(
        tmp_path, existing_config={"agents": {"entries": {"dev": {"model": "ollama/dolphin3:latest"}}}}
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "input": kwargs.get("input")})
        if "--dry-run" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": True}), stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(openclaw_config.subprocess, "run", fake_run)

    result = apply_harness("claude_cli", manager)

    assert result.changed is True
    assert result.restart_required is True
    assert len(calls) == 2  # dry-run, then real apply
    assert "--dry-run" in calls[0]["cmd"] and "--json" in calls[0]["cmd"]
    assert "--dry-run" not in calls[1]["cmd"]
    patched = json.loads(calls[0]["input"])
    assert patched["agents"]["entries"]["dev"]["model"] == "claude-cli/claude-opus-5"


def test_apply_harness_raises_on_dry_run_rejection(tmp_path, monkeypatch):
    manager = FakeGatewayManager(tmp_path)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": False, "errors": ["bad"]}), stderr="")

    monkeypatch.setattr(openclaw_config.subprocess, "run", fake_run)

    with pytest.raises(OpenClawConfigValidationError):
        apply_harness("claude_cli", manager)


def test_apply_harness_raises_on_apply_failure(tmp_path, monkeypatch):
    manager = FakeGatewayManager(tmp_path)

    def fake_run(cmd, **kwargs):
        if "--dry-run" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": True}), stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="disk full")

    monkeypatch.setattr(openclaw_config.subprocess, "run", fake_run)

    with pytest.raises(OpenClawConfigError):
        apply_harness("claude_cli", manager)


def test_codex_harness_requires_openai_model(tmp_path, monkeypatch):
    _forbid_subprocess(monkeypatch)
    manager = FakeGatewayManager(tmp_path)
    with pytest.raises(ValueError):
        apply_harness("codex", manager)


def test_codex_patch_never_carries_a_literal_secret():
    patch = _patch_for(
        "codex", agent_id="dev", openai_auth_mode="api-key", openai_api_key_env="OPENAI_API_KEY",
        openai_model="openai/gpt-5.1", claude_model="claude-cli/claude-opus-5",
        ollama_model="ollama/dolphin3:latest",
    )
    api_key_field = patch["models"]["providers"]["openai"]["apiKey"]
    assert api_key_field == {"source": "env", "provider": "default", "id": "OPENAI_API_KEY"}
    assert "sk-" not in json.dumps(patch)


def test_codex_subscription_auth_mode_omits_api_key_entirely():
    patch = _patch_for(
        "codex", agent_id="dev", openai_auth_mode="subscription", openai_api_key_env="OPENAI_API_KEY",
        openai_model="openai/gpt-5.1", claude_model="claude-cli/claude-opus-5",
        ollama_model="ollama/dolphin3:latest",
    )
    assert "models" not in patch
    assert patch == {
        "plugins": {"entries": {"codex": {"enabled": True}}},
        "agents": {"entries": {"dev": {"model": "openai/gpt-5.1"}}},
    }
