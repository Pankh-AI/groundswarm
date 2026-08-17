from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

from ..workers.base import Claim


_TYPOGRAPHIC_EQUIVALENTS = {
    "‘": "'", "’": "'",   # curly single quotes -> straight
    "“": '"', "”": '"',  # curly double quotes -> straight
    "–": "-", "—": "-",  # en/em dash -> hyphen
    "…": "...",               # ellipsis character -> three dots
}


def _normalize(text: str) -> str:
    for curly, straight in _TYPOGRAPHIC_EQUIVALENTS.items():
        text = text.replace(curly, straight)
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str


def quote_round_trips(claim: Claim) -> CheckResult | None:
    """Deterministic check: does the claim's quote appear verbatim (after
    whitespace/case normalization) in the source it claims to come from?
    Returns None if the claim has no quote to check.
    """
    if not claim.quote:
        return None
    try:
        source_text = Path(claim.source_path).read_text(encoding="utf-8")
    except OSError as exc:
        return CheckResult(False, f"could not re-read source: {exc}")
    # str()-coerced, same as computation_operands_round_trip does for operand
    # value/quote -- a live 200-worker run produced a JSON object for "quote"
    # instead of a string; str(dict) will never round-trip into plain source
    # text, so this naturally rejects instead of crashing in _normalize().
    quote = str(claim.quote)
    if _normalize(quote) in _normalize(source_text):
        return CheckResult(True, "quote round-trips into the live-refetched source")
    return CheckResult(False, "quote does not appear in the source text, invented or altered")


def computation_operands_round_trip(operands: list[dict], source_path: str) -> CheckResult:
    """Deterministic check for a computed claim's operands: does every
    operand's quote round-trip into the source (like quote_round_trips),
    and does the operand's claimed value actually appear inside that
    quote? This only validates the operands; the arithmetic itself is
    redone separately (see arithmetic.apply_operation).
    """
    try:
        source_text = Path(source_path).read_text(encoding="utf-8")
    except OSError as exc:
        return CheckResult(False, f"could not re-read source: {exc}")
    normalized_source = _normalize(source_text)
    for i, operand in enumerate(operands):
        value, quote = operand["value"], operand["quote"]
        if value is None or quote is None:
            return CheckResult(False, f"operand {i} has a null value or quote (worker didn't provide one)")
        value, quote = str(value), str(quote)
        if _normalize(quote) not in normalized_source:
            return CheckResult(False, f"operand {i} quote does not appear in the source text: {quote!r}")
        if _normalize(value) not in _normalize(quote):
            return CheckResult(False, f"operand {i} value {value!r} does not appear within its own quote {quote!r}")
    return CheckResult(True, "all operand quotes round-trip and contain their claimed values")
