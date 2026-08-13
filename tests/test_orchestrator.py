from dataclasses import dataclass, field

from groundswarm.workers.base import Claim, WorkerOutput, WorkerTask
from groundswarm.orchestrator.orchestrator import Orchestrator


class FakeWorker:
    def __init__(self, claims_by_worker_id: dict[str, list[Claim]]):
        self.claims_by_worker_id = claims_by_worker_id
        self.run_calls: list[WorkerTask] = []

    def run(self, task: WorkerTask) -> WorkerOutput:
        self.run_calls.append(task)
        claims = self.claims_by_worker_id.get(task.worker_id, [])
        return WorkerOutput(worker_id=task.worker_id, claims=claims, raw_response="{}")


@dataclass
class FakeSynthClient:
    response: str = "synthesized briefing"
    calls: list[dict] = field(default_factory=list)

    def generate(self, prompt, *, system=None, model=None, json_mode=False):
        self.calls.append({"prompt": prompt, "system": system, "model": model, "json_mode": json_mode})
        return self.response


def test_orchestrator_runs_all_workers_before_synthesis():
    claim_a = Claim(text="finding A", quote="q-a", source_path="a.txt", worker_id="w1")
    claim_b = Claim(text="finding B", quote="q-b", source_path="b.txt", worker_id="w2")
    worker = FakeWorker({"w1": [claim_a], "w2": [claim_b]})
    synth = FakeSynthClient()
    orchestrator = Orchestrator(worker=worker, synth_client=synth)

    tasks = [
        WorkerTask(worker_id="w1", instruction="extract", source_path="a.txt"),
        WorkerTask(worker_id="w2", instruction="extract", source_path="b.txt"),
    ]
    result = orchestrator.run(tasks)

    assert len(worker.run_calls) == 2
    assert len(synth.calls) == 1
    assert "finding A" in synth.calls[0]["prompt"]
    assert "finding B" in synth.calls[0]["prompt"]
    assert result.briefing == "synthesized briefing"


def test_orchestrator_skips_synthesis_when_no_claims():
    worker = FakeWorker({"w1": [], "w2": []})
    synth = FakeSynthClient()
    orchestrator = Orchestrator(worker=worker, synth_client=synth)

    tasks = [
        WorkerTask(worker_id="w1", instruction="extract", source_path="a.txt"),
        WorkerTask(worker_id="w2", instruction="extract", source_path="b.txt"),
    ]
    result = orchestrator.run(tasks)

    assert len(synth.calls) == 0
    assert result.briefing == ""
