from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class WorkerTask:
    worker_id: str
    instruction: str
    source_path: str  # local fixture standing in for a live primary source during simulation


@dataclass
class Claim:
    text: str
    quote: str | None  # a verbatim quote from the source supporting this claim, if any
    source_path: str
    worker_id: str
    # If the claim asserts a value derived by computing over source data (a
    # duration, sum, or difference) rather than one stated outright, this
    # carries {"operation": str, "operands": [{"value": str, "quote": str}, ...],
    # "result": float}. The Verifier recomputes it independently rather than
    # trusting the worker's arithmetic. None for claims with a plain quote.
    computation: dict | None = None


@dataclass
class WorkerOutput:
    worker_id: str
    claims: list[Claim]
    raw_response: str


class Worker(Protocol):
    """Anything that can execute a WorkerTask and report structured claims.

    Real OpenClaw-backed workers implement this same interface later; the
    orchestrator and verifier never need to know the difference.
    """

    def run(self, task: WorkerTask) -> WorkerOutput: ...
