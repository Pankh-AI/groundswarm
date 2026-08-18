import pytest

from groundswarm.ops import openclaw_config
from groundswarm.ops.openclaw_manager import OpenClawGatewayError
from groundswarm.runtime_config import RuntimeConfig
from groundswarm.workers.factory import worker_session
from groundswarm.workers.openclaw_worker import OpenClawWorker
from groundswarm.workers.sim_worker import OllamaSimWorker


class FakeManager:
    """Duck-typed stand-in for OpenClawGatewayManager -- worker_session()
    only ever calls it as a context manager and reads .base_url once inside."""

    def __init__(self):
        self.base_url = "http://127.0.0.1:19001"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_backend_yields_sim_worker_and_never_touches_openclaw(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("apply_harness should not be called for worker_backend='ollama'")

    monkeypatch.setattr(openclaw_config, "apply_harness", _boom)

    config = RuntimeConfig(worker_backend="ollama")
    with worker_session(config) as worker:
        assert isinstance(worker, OllamaSimWorker)


def test_openclaw_backend_calls_apply_harness_for_direct_too(monkeypatch):
    """Regression test for the harness-switching bug: apply_harness() must be
    called unconditionally for worker_backend=='openclaw', including when
    openclaw_harness=='direct' -- otherwise a Gateway previously left on
    codex/claude_cli is never reverted by a later 'direct' run."""
    calls = []

    def fake_apply_harness(harness, gateway_manager, **kwargs):
        calls.append(harness)

    monkeypatch.setattr(openclaw_config, "apply_harness", fake_apply_harness)
    monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "test-token")

    manager = FakeManager()
    config = RuntimeConfig(worker_backend="openclaw", openclaw_harness="direct")

    with worker_session(config, manager=manager) as worker:
        assert isinstance(worker, OpenClawWorker)
        assert worker.base_url == manager.base_url

    assert calls == ["direct"]


def test_openclaw_backend_forwards_harness_options(monkeypatch):
    calls = []

    def fake_apply_harness(harness, gateway_manager, **kwargs):
        calls.append((harness, kwargs))

    monkeypatch.setattr(openclaw_config, "apply_harness", fake_apply_harness)
    monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "test-token")

    manager = FakeManager()
    config = RuntimeConfig(
        worker_backend="openclaw", openclaw_harness="codex",
        harness_options={"openai_model": "openai/gpt-5.1"},
    )

    with worker_session(config, manager=manager):
        pass

    assert calls == [("codex", {"openai_model": "openai/gpt-5.1"})]


def test_openclaw_backend_raises_when_token_missing(monkeypatch):
    monkeypatch.setattr(openclaw_config, "apply_harness", lambda *a, **k: None)
    monkeypatch.delenv("OPENCLAW_GATEWAY_TOKEN", raising=False)

    manager = FakeManager()
    config = RuntimeConfig(worker_backend="openclaw")

    with pytest.raises(OpenClawGatewayError):
        with worker_session(config, manager=manager):
            pass


def test_gateway_token_param_overrides_env(monkeypatch):
    monkeypatch.setattr(openclaw_config, "apply_harness", lambda *a, **k: None)
    monkeypatch.delenv("OPENCLAW_GATEWAY_TOKEN", raising=False)

    manager = FakeManager()
    config = RuntimeConfig(worker_backend="openclaw")

    with worker_session(config, manager=manager, gateway_token="explicit-token") as worker:
        assert worker.token == "explicit-token"
