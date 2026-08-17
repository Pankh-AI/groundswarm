"""A Worker backed by a real OpenClaw Gateway agent run, not a bare LLM call.

Hitting the Gateway's OpenAI-compatible /v1/chat/completions endpoint doesn't
bypass OpenClaw -- the Gateway translates the request into a real
agentCommandFromIngress() agent run (openai-http.ts) and hands back the
agent's final text in chat-completion shape. The `user` field maps directly
to a Gateway session key (sessionPrefix "openai"), so each call's `user` is
built from BOTH the worker_id and a per-worker_id attempt counter -- not
worker_id alone. Measured live (ADR-019): reusing the same `user` across two
calls to the same worker_id threads them through one ongoing OpenClaw
session, and the agent anchors on its first answer -- a corrected system
prompt on a same-session retry came back byte-for-byte identical. A fresh
session per attempt is what makes a feedback-loop caution note (see
ops/feedback.py) able to change the output at all. See ADR-018/019.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .base import Claim, WorkerOutput, WorkerTask

EXTRACTION_SYSTEM_PROMPT = (
    "You extract findings from an incident postmortem. Read ONLY the source "
    "text given to you. Respond with strict JSON: "
    '{"claims": [{"text": "...", "quote": <QUOTE_OR_NULL>, "computation": <COMPUTATION_OR_NULL>}]}. '
    "quote: copy a supporting span exactly from the source text, character "
    "for character, with no leading or trailing ellipsis and no text from "
    "outside that one span. If no single span supports the claim, set "
    "quote to null. Never write out the words 'a verbatim quote from the "
    "source' or any other instruction text as the quote value.\n"
    "computation: only set this if the claim's number is something YOU "
    "computed from two or more source values (a duration, sum, or "
    "difference) rather than a number stated outright in the source. "
    'Format: {"operation": "duration_minutes" | "sum" | "difference", '
    '"operands": [{"value": "...", "quote": "verbatim source span this '
    'operand came from"}, ...], "result": <the number your claim asserts>}. '
    "For duration_minutes, operands are [start_time, end_time]. If the "
    "claim's number appears directly in the source, leave computation null "
    "and use quote instead."
)


class OpenClawUnavailable(RuntimeError):
    """Raised when the Gateway can't be reached or rejects the request."""


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*)\n\s*```\s*$", re.DOTALL)


def _strip_code_fence(raw: str) -> str:
    """A real agent turn (unlike a raw Ollama json_mode call) sometimes wraps
    its JSON answer in a markdown code fence despite being told to respond
    with strict JSON. Rather than losing the whole claim set to a
    JSONDecodeError, unwrap a fence if present; otherwise return raw as-is.
    """
    match = _CODE_FENCE_RE.match(raw.strip())
    return match.group(1) if match else raw


@dataclass
class CallLog:
    label: str
    prompt_chars: int
    success: bool
    duration_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


@dataclass
class OpenClawWorker:
    """Implements the Worker protocol against a running OpenClaw Gateway."""

    base_url: str
    token: str
    model: str = "openclaw"
    timeout_s: float = 180.0
    max_tokens: int = 1024
    calls: list[CallLog] = field(default_factory=list, repr=False, compare=False)
    # Feedback loop (see ops/feedback.py): keyed by worker_id, an instruction
    # appended to that worker's next system prompt. Groundswarm's own
    # Verifier output writes into this between dispatches -- the same worker
    # identity's next OpenClaw agent turn is shaped by how its last one was
    # independently checked, not just re-run unchanged.
    source_notes: dict[str, str] = field(default_factory=dict, repr=False)
    # Per-worker_id call count, folded into the Gateway session's `user` key.
    # Measured live (ADR-019): re-dispatching the same worker_id with the
    # SAME `user` reuses OpenClaw's own session, and the agent anchors on its
    # first turn -- a corrected system prompt on a same-session retry got a
    # byte-for-byte identical answer back. A fresh session per attempt is
    # required for a caution note to have any chance of changing the output.
    _attempts: dict[str, int] = field(default_factory=dict, repr=False)

    def _chat(self, prompt: str, *, system: str, user: str, label: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "user": user,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
            method="POST",
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            self.calls.append(CallLog(
                label=label, prompt_chars=len(prompt), success=False,
                duration_s=time.monotonic() - start, error=str(exc),
            ))
            raise OpenClawUnavailable(f"could not reach OpenClaw Gateway at {self.base_url}: {exc}") from exc

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self.calls.append(CallLog(
            label=label, prompt_chars=len(prompt), success=True,
            duration_s=time.monotonic() - start,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        ))
        return content

    def run(self, task: WorkerTask) -> WorkerOutput:
        source_text = Path(task.source_path).read_text(encoding="utf-8")
        prompt = f"{task.instruction}\n\n---SOURCE---\n{source_text}\n---END SOURCE---"
        note = self.source_notes.get(task.worker_id)
        system = f"{EXTRACTION_SYSTEM_PROMPT}\n\n{note}" if note else EXTRACTION_SYSTEM_PROMPT
        attempt = self._attempts.get(task.worker_id, 0)
        self._attempts[task.worker_id] = attempt + 1
        try:
            raw = self._chat(
                prompt,
                system=system,
                user=f"groundswarm:{task.worker_id}:{attempt}",
                label=f"extract:{task.worker_id}",
            )
        except OpenClawUnavailable:
            # Same fail-degrade contract as OllamaSimWorker: one worker's transport
            # failure yields zero claims, not a batch-wide crash.
            return WorkerOutput(worker_id=task.worker_id, claims=[], raw_response="")
        claims: list[Claim] = []
        try:
            parsed = json.loads(_strip_code_fence(raw))
            for c in parsed.get("claims", []):
                claims.append(
                    Claim(
                        text=str(c.get("text", "")).strip(),
                        quote=(c.get("quote") or None),
                        source_path=task.source_path,
                        worker_id=task.worker_id,
                        computation=(c.get("computation") or None),
                    )
                )
        except (json.JSONDecodeError, AttributeError):
            pass
        return WorkerOutput(worker_id=task.worker_id, claims=claims, raw_response=raw)
