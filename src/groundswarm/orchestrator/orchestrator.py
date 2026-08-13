from __future__ import annotations
from dataclasses import dataclass

from ..workers.base import Worker, WorkerOutput, WorkerTask
from ..llm.ollama_client import OllamaClient

SYNTHESIS_SYSTEM_PROMPT = (
    "You write a short risk briefing from a list of extracted findings. "
    "Use only the findings given to you, do not add facts that are not in "
    "the list. Keep it to 3-5 sentences."
)


@dataclass
class OrchestrationResult:
    worker_outputs: list[WorkerOutput]
    briefing: str


class Orchestrator:
    """Decomposes a task into bounded per-worker scopes, runs them, and only
    starts the synthesis step once every extraction worker has reported.
    """

    def __init__(self, worker: Worker, synth_client: OllamaClient | None = None):
        self.worker = worker
        self.synth_client = synth_client or OllamaClient()

    def run(self, tasks: list[WorkerTask]) -> OrchestrationResult:
        worker_outputs = [self.worker.run(t) for t in tasks]

        all_claims = [c for out in worker_outputs for c in out.claims]
        if not all_claims:
            return OrchestrationResult(worker_outputs=worker_outputs, briefing="")

        findings_block = "\n".join(f"- ({c.worker_id}) {c.text}" for c in all_claims)
        briefing = self.synth_client.generate(
            f"Findings:\n{findings_block}", system=SYNTHESIS_SYSTEM_PROMPT
        )
        return OrchestrationResult(worker_outputs=worker_outputs, briefing=briefing)
