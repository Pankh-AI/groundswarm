import pytest

from groundswarm.runtime_config import InvalidRuntimeConfig, RuntimeConfig, collect_harness_options


def test_from_env_defaults_to_ollama_direct():
    cfg = RuntimeConfig.from_env(env={})
    assert cfg.worker_backend == "ollama"
    assert cfg.openclaw_harness == "direct"
    assert cfg.harness_options == {}


def test_from_env_reads_worker_backend_and_harness_case_insensitively():
    cfg = RuntimeConfig.from_env(env={
        "GROUNDSWARM_WORKER_BACKEND": "OpenClaw",
        "GROUNDSWARM_OPENCLAW_HARNESS": "Claude_Cli",
    })
    assert cfg.worker_backend == "openclaw"
    assert cfg.openclaw_harness == "claude_cli"


def test_from_env_collects_only_harness_option_vars():
    cfg = RuntimeConfig.from_env(env={
        "GROUNDSWARM_WORKER_BACKEND": "openclaw",
        "GROUNDSWARM_HARNESS_OPTION_OPENAI_MODEL": "openai/gpt-5.1",
        "UNRELATED_VAR": "ignored",
    })
    assert cfg.harness_options == {"openai_model": "openai/gpt-5.1"}


def test_collect_harness_options_lowercases_keys():
    options = collect_harness_options({"GROUNDSWARM_HARNESS_OPTION_CLAUDE_MODEL": "claude-cli/claude-opus-5"})
    assert options == {"claude_model": "claude-cli/claude-opus-5"}


def test_invalid_worker_backend_raises():
    with pytest.raises(InvalidRuntimeConfig):
        RuntimeConfig(worker_backend="bogus").validate()


def test_invalid_openclaw_harness_only_checked_when_backend_is_openclaw():
    # "ollama" never reads openclaw_harness, so a bogus value there is not an error.
    RuntimeConfig(worker_backend="ollama", openclaw_harness="bogus").validate()

    with pytest.raises(InvalidRuntimeConfig):
        RuntimeConfig(worker_backend="openclaw", openclaw_harness="bogus").validate()


def test_from_env_validates_and_raises_for_unknown_backend():
    with pytest.raises(InvalidRuntimeConfig):
        RuntimeConfig.from_env(env={"GROUNDSWARM_WORKER_BACKEND": "bogus"})
