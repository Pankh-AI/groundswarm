from __future__ import annotations
import json
from pathlib import Path

from ..llm.ollama_client import OllamaClient, OllamaUnavailable
from .base import Claim, Worker, WorkerOutput, WorkerTask

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


class OllamaSimWorker:
    """A worker stand-in for local development: reads a fixture source file
    and uses a local Ollama model to extract claims from it. Implements the
    same Worker interface a real OpenClaw-backed worker will implement.
    """

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def run(self, task: WorkerTask) -> WorkerOutput:
        source_text = Path(task.source_path).read_text(encoding="utf-8")
        prompt = f"{task.instruction}\n\n---SOURCE---\n{source_text}\n---END SOURCE---"
        try:
            raw = self.client.generate(prompt, system=EXTRACTION_SYSTEM_PROMPT, json_mode=True,
                                         label=f"extract:{task.worker_id}")
        except OllamaUnavailable:
            # A transient network/transport failure on this one worker's call must not
            # take down the whole batch -- degrade to zero claims, same as an
            # unparseable response, and let the other workers in the batch proceed.
            return WorkerOutput(worker_id=task.worker_id, claims=[], raw_response="")
        claims: list[Claim] = []
        try:
            parsed = json.loads(raw)
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
