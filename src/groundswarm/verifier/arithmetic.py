from __future__ import annotations
import re

_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _parse_clock_minutes(value: str) -> int | None:
    """Parses an 'HH:MM' (optionally with a UTC/date suffix) into minutes
    since midnight. Returns None if no HH:MM pattern is found.
    """
    match = _TIME_RE.search(value)
    if match is None:
        return None
    hours, minutes = int(match.group(1)), int(match.group(2))
    if hours > 23 or minutes > 59:
        return None
    return hours * 60 + minutes


def _parse_number(value: str) -> float | None:
    match = _NUMBER_RE.search(value.replace(",", ""))
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def apply_operation(operation: str, operand_values: list[str]) -> float | None:
    """Independently recomputes a claimed result from its operand strings.
    Returns None if the operation is unsupported or an operand doesn't
    parse - the caller must treat that as "can't verify", not "verified
    false", and fail closed.
    """
    if operation == "duration_minutes":
        if len(operand_values) != 2:
            return None
        start, end = (_parse_clock_minutes(v) for v in operand_values)
        if start is None or end is None:
            return None
        return (end - start) % (24 * 60)

    if operation in ("sum", "difference"):
        numbers = [_parse_number(v) for v in operand_values]
        if not numbers or any(n is None for n in numbers):
            return None
        if operation == "sum":
            return sum(numbers)
        if len(numbers) != 2:
            return None
        return numbers[0] - numbers[1]

    return None
