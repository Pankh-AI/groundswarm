from groundswarm.workers.base import Claim
from groundswarm.verifier.checks import quote_round_trips


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
