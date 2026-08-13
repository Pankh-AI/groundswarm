import json
from dataclasses import dataclass, field

from groundswarm.workers.base import Claim
from groundswarm.llm.ollama_client import OllamaUnavailable
from groundswarm.verifier.verifier import Verifier


@dataclass
class FakeJudgeClient:
    response: str = ""
    raise_unavailable: bool = False
    calls: list[dict] = field(default_factory=list)

    def generate(self, prompt, *, system=None, model=None, json_mode=False):
        self.calls.append({"prompt": prompt, "system": system, "model": model, "json_mode": json_mode})
        if self.raise_unavailable:
            raise OllamaUnavailable("simulated: judge unreachable")
        return self.response


def unquoted_claim(source_path: str) -> Claim:
    return Claim(text="a claim with no quote", quote=None, source_path=source_path, worker_id="w1")


def test_judge_confirms_when_supported_true(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("the failover took 6 minutes and 40 seconds", encoding="utf-8")
    judge = FakeJudgeClient(response=json.dumps({"supported": True, "reason": "matches source"}))
    verifier = Verifier(judge_client=judge)

    result = verifier.verify_claim(unquoted_claim(str(source)))

    assert result.verdict == "confirmed"
    assert result.method == "llm_judge"
    assert len(judge.calls) == 1


def test_judge_rejects_when_supported_false(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("the failover took 6 minutes and 40 seconds", encoding="utf-8")
    judge = FakeJudgeClient(response=json.dumps({"supported": False, "reason": "not in source"}))
    verifier = Verifier(judge_client=judge)

    result = verifier.verify_claim(unquoted_claim(str(source)))

    assert result.verdict == "rejected"
    assert result.method == "llm_judge"


def test_judge_fails_closed_when_ollama_unavailable(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("some source text", encoding="utf-8")
    judge = FakeJudgeClient(raise_unavailable=True)
    verifier = Verifier(judge_client=judge)

    result = verifier.verify_claim(unquoted_claim(str(source)))

    assert result.verdict == "unverified"
    assert result.method == "fail_closed"


def test_judge_fails_closed_on_unparseable_response(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("some source text", encoding="utf-8")
    judge = FakeJudgeClient(response="not valid json at all")
    verifier = Verifier(judge_client=judge)

    result = verifier.verify_claim(unquoted_claim(str(source)))

    assert result.verdict == "unverified"
    assert result.method == "fail_closed"


def test_judge_fails_closed_on_missing_supported_key(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("some source text", encoding="utf-8")
    judge = FakeJudgeClient(response=json.dumps({"reason": "forgot the key"}))
    verifier = Verifier(judge_client=judge)

    result = verifier.verify_claim(unquoted_claim(str(source)))

    assert result.verdict == "unverified"
    assert result.method == "fail_closed"


def test_judge_fails_closed_on_missing_source_file():
    judge = FakeJudgeClient(response=json.dumps({"supported": True, "reason": "n/a"}))
    verifier = Verifier(judge_client=judge)

    claim = unquoted_claim("/nonexistent/path/does_not_exist.txt")
    result = verifier.verify_claim(claim)

    assert result.verdict == "unverified"
    assert result.method == "fail_closed"
    assert len(judge.calls) == 0


def test_deterministic_check_short_circuits_before_judge(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("the failover took 6 minutes and 40 seconds", encoding="utf-8")
    judge = FakeJudgeClient(response=json.dumps({"supported": True, "reason": "n/a"}))
    verifier = Verifier(judge_client=judge)

    claim = Claim(
        text="the failover took 6 minutes and 40 seconds",
        quote="the failover took 6 minutes and 40 seconds",
        source_path=str(source),
        worker_id="w1",
    )
    result = verifier.verify_claim(claim)

    assert result.method == "deterministic"
    assert len(judge.calls) == 0


def test_verify_all_runs_every_claim(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("fact one. fact two.", encoding="utf-8")
    judge = FakeJudgeClient(response=json.dumps({"supported": True, "reason": "n/a"}))
    verifier = Verifier(judge_client=judge)

    claims = [
        Claim(text="fact one", quote="fact one", source_path=str(source), worker_id="w1"),
        unquoted_claim(str(source)),
    ]
    results = verifier.verify_all(claims)

    assert len(results) == 2
    assert results[0].method == "deterministic"
    assert results[1].method == "llm_judge"
