from groundswarm.workers.base import WorkerTask
from groundswarm.workers.sim_worker import OllamaSimWorker
from groundswarm.llm.ollama_client import OllamaUnavailable


class FailingClient:
    """Stands in for OllamaClient.generate() raising on a transport failure --
    e.g. the mid-run read timeout that crashed the whole N=20 batch before this
    fix (ADR-015): one worker's network hiccup must not take down the others."""

    def generate(self, prompt, *, system=None, model=None, json_mode=False, label="unlabeled"):
        raise OllamaUnavailable("could not reach Ollama: The read operation timed out")


def test_worker_degrades_to_empty_claims_on_transport_failure(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("The incident lasted 10 minutes.", encoding="utf-8")

    worker = OllamaSimWorker(client=FailingClient())
    task = WorkerTask(worker_id="w1", instruction="extract", source_path=str(source))

    output = worker.run(task)

    assert output.worker_id == "w1"
    assert output.claims == []
    assert output.raw_response == ""
