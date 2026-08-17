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

    def generate(self, prompt, *, system=None, model=None, json_mode=False, label="unlabeled"):
        self.calls.append({"prompt": prompt, "system": system, "model": model, "json_mode": json_mode, "label": label})
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


def _duration_claim(source_path: str, claimed_minutes: float) -> Claim:
    return Claim(
        text=f"The backlog drain completed within {claimed_minutes} minutes after throttling.",
        quote=None,
        source_path=source_path,
        worker_id="w1",
        computation={
            "operation": "duration_minutes",
            "operands": [
                {"value": "12:11 UTC", "quote": "throttled the partner's webhook concurrency at 12:11 UTC"},
                {"value": "12:52 UTC", "quote": "Full backlog drain completed at 12:52 UTC"},
            ],
            "result": claimed_minutes,
        },
    )


def test_computation_confirms_correct_arithmetic(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text(
        "throttled the partner's webhook concurrency at 12:11 UTC, which stabilized lag. "
        "Full backlog drain completed at 12:52 UTC.",
        encoding="utf-8",
    )
    judge = FakeJudgeClient(response=json.dumps({"supported": True, "reason": "n/a"}))
    verifier = Verifier(judge_client=judge)

    result = verifier.verify_claim(_duration_claim(str(source), 41))

    assert result.verdict == "confirmed"
    assert result.method == "computation"
    assert len(judge.calls) == 0  # never needed the judge for a computed claim


def test_computation_rejects_wrong_arithmetic_the_real_bug_case(tmp_path):
    # scenarios/incident_monitoring_swarm, solace_queue_r11: a worker once
    # claimed "17 minutes" here when the source timestamps put it at 41.
    # It was rejected on a quote-format technicality before; this is the
    # dedicated check that catches the actual number being wrong.
    source = tmp_path / "source.txt"
    source.write_text(
        "throttled the partner's webhook concurrency at 12:11 UTC, which stabilized lag. "
        "Full backlog drain completed at 12:52 UTC.",
        encoding="utf-8",
    )
    verifier = Verifier(judge_client=FakeJudgeClient())

    result = verifier.verify_claim(_duration_claim(str(source), 17))

    assert result.verdict == "rejected"
    assert result.method == "computation"
    assert "41" in result.reason and "17" in result.reason


def test_computation_rejects_when_an_operand_quote_is_invented(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("throttled the partner's webhook concurrency at 12:11 UTC.", encoding="utf-8")
    verifier = Verifier(judge_client=FakeJudgeClient())

    result = verifier.verify_claim(_duration_claim(str(source), 41))

    assert result.verdict == "rejected"
    assert result.method == "computation"


def test_computation_fails_closed_on_unsupported_operation(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("throttled at 12:11 UTC.", encoding="utf-8")
    claim = Claim(
        text="a claim with an unsupported operation",
        quote=None,
        source_path=str(source),
        worker_id="w1",
        computation={
            "operation": "percentage_change",
            "operands": [{"value": "12:11 UTC", "quote": "throttled at 12:11 UTC"}],
            "result": 5,
        },
    )
    verifier = Verifier(judge_client=FakeJudgeClient())

    result = verifier.verify_claim(claim)

    assert result.verdict == "unverified"
    assert result.method == "fail_closed"


def test_computation_fails_closed_when_an_operand_is_missing_its_quote_key(tmp_path):
    # Real crash from the first live 55-worker re-run after ADR-009: dolphin3
    # started populating "computation" but sometimes emitted an operand dict
    # with no "quote" key at all, and computation_operands_round_trip() did
    # operand["quote"] unguarded, raising KeyError out of verify_all() and
    # killing the entire run instead of failing closed on that one claim.
    source = tmp_path / "source.txt"
    source.write_text("throttled at 12:11 UTC. drain completed at 12:52 UTC.", encoding="utf-8")
    claim = Claim(
        text="a claim whose worker forgot to quote one operand",
        quote=None,
        source_path=str(source),
        worker_id="w1",
        computation={
            "operation": "duration_minutes",
            "operands": [
                {"value": "12:11 UTC"},  # no "quote" key
                {"value": "12:52 UTC", "quote": "drain completed at 12:52 UTC"},
            ],
            "result": 41,
        },
    )
    verifier = Verifier(judge_client=FakeJudgeClient())

    result = verifier.verify_claim(claim)

    assert result.verdict == "unverified"
    assert result.method == "fail_closed"


def test_computation_fails_closed_when_an_operand_is_missing_its_value_key(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("throttled at 12:11 UTC. drain completed at 12:52 UTC.", encoding="utf-8")
    claim = Claim(
        text="a claim whose worker forgot to state one operand's value",
        quote=None,
        source_path=str(source),
        worker_id="w1",
        computation={
            "operation": "duration_minutes",
            "operands": [
                {"quote": "throttled at 12:11 UTC"},  # no "value" key
                {"value": "12:52 UTC", "quote": "drain completed at 12:52 UTC"},
            ],
            "result": 41,
        },
    )
    verifier = Verifier(judge_client=FakeJudgeClient())

    result = verifier.verify_claim(claim)

    assert result.verdict == "unverified"
    assert result.method == "fail_closed"


def test_computation_fails_closed_on_malformed_payload(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("some source text", encoding="utf-8")
    claim = Claim(
        text="a claim with a malformed computation payload",
        quote=None,
        source_path=str(source),
        worker_id="w1",
        computation={"operation": "duration_minutes"},  # missing operands/result
    )
    verifier = Verifier(judge_client=FakeJudgeClient())

    result = verifier.verify_claim(claim)

    assert result.verdict == "unverified"
    assert result.method == "fail_closed"


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
