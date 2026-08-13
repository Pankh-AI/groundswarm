from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

from ..workers.base import Claim


def _normalize(text: str) -> str:
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
    if _normalize(claim.quote) in _normalize(source_text):
        return CheckResult(True, "quote round-trips into the live-refetched source")
    return CheckResult(False, "quote does not appear in the source text, invented or altered")
