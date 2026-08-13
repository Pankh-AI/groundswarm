from __future__ import annotations
import json
from pathlib import Path

from ..llm.ollama_client import OllamaClient
from .base import Claim, Worker, WorkerOutput, WorkerTask

EXTRACTION_SYSTEM_PROMPT = (
    "You extract findings from an incident postmortem. Read ONLY the source "
    "text given to you. Respond with strict JSON: "
    '{"claims": [{"text": "...", "quote": "a verbatim quote from the source '
    'supporting this claim, or null if none applies"}]}. '
    "Every quote must be copied exactly from the source text, do not "
    "paraphrase inside a quote field. If you cannot find a supporting quote "
    "for a claim, set quote to null rather than inventing one."
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
        raw = self.client.generate(prompt, system=EXTRACTION_SYSTEM_PROMPT, json_mode=True)
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
                    )
                )
        except (json.JSONDecodeError, AttributeError):
            pass
        return WorkerOutput(worker_id=task.worker_id, claims=claims, raw_response=raw)
