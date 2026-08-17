from groundswarm.verifier.arithmetic import apply_operation


def test_duration_minutes_matches_the_real_solace_queue_case():
    # scenarios/incident_monitoring_swarm: throttled at 12:11 UTC, drain
    # completed at 12:52 UTC = 41 minutes. A worker once claimed "17
    # minutes" here and it slipped through on a quote-format technicality
    # rather than being caught on the arithmetic itself. This is that case.
    assert apply_operation("duration_minutes", ["12:11 UTC", "12:52 UTC"]) == 41


def test_duration_minutes_handles_plain_hhmm():
    assert apply_operation("duration_minutes", ["03:31", "03:41"]) == 10


def test_duration_minutes_wraps_past_midnight():
    assert apply_operation("duration_minutes", ["23:50", "00:10"]) == 20


def test_duration_minutes_returns_none_on_unparseable_operand():
    assert apply_operation("duration_minutes", ["around noon", "12:52 UTC"]) is None


def test_duration_minutes_returns_none_on_wrong_operand_count():
    assert apply_operation("duration_minutes", ["12:11 UTC"]) is None


def test_sum_adds_all_operands():
    assert apply_operation("sum", ["100", "200", "3.5"]) == 303.5


def test_sum_strips_thousands_separators():
    assert apply_operation("sum", ["1,000", "340,000"]) == 341000


def test_difference_subtracts_second_from_first():
    assert apply_operation("difference", ["340,000", "100,000"]) == 240000


def test_difference_returns_none_for_wrong_operand_count():
    assert apply_operation("difference", ["1", "2", "3"]) is None


def test_unsupported_operation_returns_none():
    assert apply_operation("percentage_change", ["1", "2"]) is None


def test_unparseable_number_returns_none():
    assert apply_operation("sum", ["not a number"]) is None
