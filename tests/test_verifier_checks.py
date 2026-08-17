from groundswarm.workers.base import Claim
from groundswarm.verifier.checks import computation_operands_round_trip, quote_round_trips


def test_quote_round_trip_passes_for_real_quote(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("The outage lasted 47 minutes and no data was lost.", encoding="utf-8")
    claim = Claim(
        text="The outage lasted 47 minutes.",
        quote="The outage lasted 47 minutes",
        source_path=str(source),
        worker_id="w1",
    )
    result = quote_round_trips(claim)
    assert result is not None
    assert result.passed


def test_quote_round_trip_fails_for_fabricated_quote(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("The outage lasted 47 minutes and no data was lost.", encoding="utf-8")
    claim = Claim(
        text="The outage caused permanent data loss.",
        quote="permanent data loss occurred across all buckets",
        source_path=str(source),
        worker_id="w1",
    )
    result = quote_round_trips(claim)
    assert result is not None
    assert not result.passed


def test_quote_round_trip_none_when_no_quote(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("some source text", encoding="utf-8")
    claim = Claim(text="a claim with no quote", quote=None, source_path=str(source), worker_id="w1")
    assert quote_round_trips(claim) is None


def test_quote_round_trip_fails_closed_on_missing_source():
    claim = Claim(
        text="a claim",
        quote="a quote",
        source_path="/nonexistent/path/does_not_exist.txt",
        worker_id="w1",
    )
    result = quote_round_trips(claim)
    assert result is not None
    assert not result.passed


def test_quote_round_trip_tolerates_curly_apostrophe_the_model_wrote(tmp_path):
    # Real case from the 55-worker swarm run (solace_queue_r11): the source
    # uses a straight apostrophe, the model wrote a curly one in its quote,
    # and the claim was otherwise word-for-word correct.
    source = tmp_path / "source.txt"
    source.write_text("the queue's retention window of 72 hours was never approached.", encoding="utf-8")
    claim = Claim(
        text="The queue's retention window was never approached.",
        quote="the queue’s retention window of 72 hours was never approached.",
        source_path=str(source),
        worker_id="w1",
    )
    result = quote_round_trips(claim)
    assert result is not None
    assert result.passed


def test_quote_round_trip_tolerates_curly_quotes_and_em_dash(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text('The team said "we will fix it" - a promise kept.', encoding="utf-8")
    claim = Claim(
        text="the team promised to fix it",
        quote="the team said “we will fix it” — a promise kept.",
        source_path=str(source),
        worker_id="w1",
    )
    result = quote_round_trips(claim)
    assert result is not None
    assert result.passed


def test_quote_round_trip_fails_closed_on_a_dict_quote(tmp_path):
    # Live 200-worker run (N=200 sweep): dolphin3 emitted a JSON object for
    # "quote" instead of a string. _normalize() calls .replace() directly on
    # claim.quote and crashed the whole batch with AttributeError before this
    # fix. Never seen below N=200 -- a longer sweep surfaced a rarer shape.
    source = tmp_path / "source.txt"
    source.write_text("The outage lasted 47 minutes and no data was lost.", encoding="utf-8")
    claim = Claim(
        text="The outage lasted 47 minutes.",
        quote={"span": "The outage lasted 47 minutes", "confidence": 0.9},
        source_path=str(source),
        worker_id="w1",
    )
    result = quote_round_trips(claim)
    assert result is not None
    assert not result.passed


def test_computation_operands_round_trip_passes_when_both_operands_are_real(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text(
        "throttled the specific partner's webhook concurrency at 12:11 UTC, which stabilized "
        "lag within 8 minutes. Full backlog drain completed at 12:52 UTC.",
        encoding="utf-8",
    )
    operands = [
        {"value": "12:11 UTC", "quote": "throttled the specific partner's webhook concurrency at 12:11 UTC"},
        {"value": "12:52 UTC", "quote": "Full backlog drain completed at 12:52 UTC"},
    ]
    result = computation_operands_round_trip(operands, str(source))
    assert result.passed


def test_computation_operands_round_trip_fails_when_a_quote_is_invented(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("throttled the partner's webhook concurrency at 12:11 UTC.", encoding="utf-8")
    operands = [
        {"value": "12:11 UTC", "quote": "throttled the partner's webhook concurrency at 12:11 UTC"},
        {"value": "12:52 UTC", "quote": "drain completed at 12:52 UTC"},  # not in this source
    ]
    result = computation_operands_round_trip(operands, str(source))
    assert not result.passed


def test_computation_operands_round_trip_fails_when_value_not_inside_its_own_quote(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("throttled the partner's webhook concurrency at 12:11 UTC.", encoding="utf-8")
    operands = [
        {"value": "12:52 UTC", "quote": "throttled the partner's webhook concurrency at 12:11 UTC"},
    ]
    result = computation_operands_round_trip(operands, str(source))
    assert not result.passed


def test_computation_operands_round_trip_fails_closed_on_a_null_quote(tmp_path):
    # Live 55-worker run (post ADR-009): a worker sometimes emits JSON null
    # for an operand's "quote" instead of omitting the key. str(None) turns
    # into the literal text "None", which then fails the substring check
    # with a misleading "quote does not appear in the source" reason -- as
    # if the worker had invented text, when really it provided nothing.
    # Caught explicitly now so the reason says what actually happened.
    source = tmp_path / "source.txt"
    source.write_text("throttled the partner's webhook concurrency at 12:11 UTC.", encoding="utf-8")
    operands = [
        {"value": "12:11 UTC", "quote": None},
    ]
    result = computation_operands_round_trip(operands, str(source))
    assert not result.passed
    assert "null" in result.reason


def test_computation_operands_round_trip_fails_closed_on_a_null_value(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("throttled the partner's webhook concurrency at 12:11 UTC.", encoding="utf-8")
    operands = [
        {"value": None, "quote": "throttled the partner's webhook concurrency at 12:11 UTC"},
    ]
    result = computation_operands_round_trip(operands, str(source))
    assert not result.passed
    assert "null" in result.reason
