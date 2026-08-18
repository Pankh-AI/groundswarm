"""Groundswarm-owned lifecycle management for a local OpenClaw Gateway.

Per ADR-018, Groundswarm doesn't just call into an externally-run OpenClaw
instance -- it starts, health-checks, and stops the Gateway process itself,
the same way it already owns pointing at a local/remote Ollama server. This
module is the process-supervision half of that; OpenClawWorker (in
workers/openclaw_worker.py) is the request half.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class OpenClawGatewayError(RuntimeError):
    """Raised when the Gateway process fails to start or become reachable."""


class OpenClawGatewayManager:
    def __init__(
        self,
        repo_path: str | Path | None = None,
        *,
        state_dir: str | Path | None = None,
        host: str = "127.0.0.1",
        startup_timeout_s: float = 600.0,
        log_path: str | Path | None = None,
    ):
        self.repo_path = Path(repo_path or os.environ.get("OPENCLAW_REPO", ""))
        if not self.repo_path or not (self.repo_path / "package.json").exists():
            raise OpenClawGatewayError(
                f"OpenClaw repo not found at {self.repo_path!r}; pass repo_path or set OPENCLAW_REPO"
            )
        self.state_dir = Path(state_dir or (Path.home() / ".openclaw-dev"))
        self.host = host
        self.startup_timeout_s = startup_timeout_s
        self.log_path = Path(log_path) if log_path else self.state_dir / "gateway-managed.log"
        self.port: int | None = None
        self._proc: subprocess.Popen | None = None
        self._lock_pid: int | None = None

    def _lock_dir(self) -> Path:
        return self.state_dir / "tmp" / "openclaw"

    def _find_lock(self, newer_than: float) -> dict | None:
        lock_dir = self._lock_dir()
        if not lock_dir.exists():
            return None
        # A couple seconds of slack absorbs filesystem mtime-resolution rounding
        # so a lock written in the same wall-clock second we started isn't missed.
        candidates = sorted(
            (p for p in lock_dir.glob("gateway.*.lock") if p.stat().st_mtime >= newer_than - 2.0),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in candidates:
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
        return None

    def _http_ready(self) -> bool:
        if self.port is None:
            return False
        try:
            req = urllib.request.Request(f"http://{self.host}:{self.port}/")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def is_available(self) -> bool:
        """Quick reachability check, mirroring OllamaClient.is_available()."""
        return self._proc is not None and self._proc.poll() is None and self._http_ready()

    def start(self) -> str:
        """Starts the Gateway dev process (a no-op if already running under
        this manager), waits for it to publish a bound port and accept HTTP,
        and returns its base URL. Raises OpenClawGatewayError on timeout or
        an early process exit -- fail closed rather than hand back a worker
        pointed at a Gateway that never came up.
        """
        if self._proc is not None and self._proc.poll() is None:
            return self.base_url

        started_at = time.time()
        env = dict(os.environ)
        env["OPENCLAW_SKIP_CHANNELS"] = "1"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(self.log_path, "a", encoding="utf-8")
        self._proc = subprocess.Popen(
            ["node", "scripts/run-node.mjs", "--dev", "gateway"],
            cwd=str(self.repo_path),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        deadline = started_at + self.startup_timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise OpenClawGatewayError(
                    f"gateway process exited early (code {self._proc.returncode}); see {self.log_path}"
                )
            lock = self._find_lock(newer_than=started_at)
            if lock and isinstance(lock.get("port"), int):
                self.port = lock["port"]
                self._lock_pid = lock.get("pid")
                if self._http_ready():
                    return self.base_url
            time.sleep(1.0)

        self.stop()
        raise OpenClawGatewayError(
            f"gateway did not become ready within {self.startup_timeout_s}s; see {self.log_path}"
        )

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise OpenClawGatewayError("gateway not started")
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        """Kills the Gateway's whole process tree via its own lock-file PID
        (not just the process we Popen'd) so a build-step child node spawned
        along the way can't survive as an orphan after Groundswarm exits.
        """
        pid = self._lock_pid or (self._proc.pid if self._proc else None)
        if pid is not None:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self.port = None
        self._lock_pid = None

    def __enter__(self) -> "OpenClawGatewayManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
