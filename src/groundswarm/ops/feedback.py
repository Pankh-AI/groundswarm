"""Groundswarm -> OpenClaw feedback loop: verification outcomes reshape the
next dispatch instead of flowing one way from OpenClaw to Groundswarm.

Deliberately narrow: tracks per-worker confirm/reject/unverified counts from
real VerificationResults and produces a short caution note to inject into
that worker's next system prompt once its confirmed rate drops below
CAUTION_THRESHOLD. Does not touch OpenClaw's own config, channel routing, or
model choice -- reshaping the actual Gateway/agent config from Groundswarm is
follow-on scope, not built here. See ADR-018/019.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..verifier.verifier import VerificationResult

# Below this confirmed-fraction, a worker's next dispatch gets a caution note.
# Deliberately permissive: at the sample sizes one swarm run produces (single
# digits per worker), a stricter threshold fires on noise as often as on a
# real problem. Not a tuned value -- see the ops writeup before treating it
# as validated.
CAUTION_THRESHOLD = 0.8


@dataclass
class SourceReliability:
    worker_id: str
    confirmed: int = 0
    rejected: int = 0
    unverified: int = 0

    @property
    def total(self) -> int:
        return self.confirmed + self.rejected + self.unverified

    @property
    def confirmed_fraction(self) -> float | None:
        return None if self.total == 0 else self.confirmed / self.total

    @property
    def needs_caution(self) -> bool:
        frac = self.confirmed_fraction
        return frac is not None and frac < CAUTION_THRESHOLD


def compute_reliability(verifications: list[VerificationResult]) -> dict[str, SourceReliability]:
    """Groups verification results by worker_id and tallies verdicts."""
    by_worker: dict[str, SourceReliability] = {}
    for v in verifications:
        rel = by_worker.setdefault(v.claim.worker_id, SourceReliability(worker_id=v.claim.worker_id))
        if v.verdict == "confirmed":
            rel.confirmed += 1
        elif v.verdict == "rejected":
            rel.rejected += 1
        else:
            rel.unverified += 1
    return by_worker


def caution_note(rel: SourceReliability) -> str:
    """A short instruction appended to a flagged worker's next system prompt.
    Names the observed rate, not a guessed cause -- the Verifier's per-claim
    rejection reasons aren't threaded through here, only the tally.
    """
    return (
        f"CAUTION: your last {rel.total} claim(s) from this source had "
        f"{rel.rejected} rejected and {rel.unverified} unverified by "
        f"independent fact-checking (only {rel.confirmed} confirmed). Quote "
        "text exactly, character for character, with no summarizing or "
        "ellipsis inside a quote span, and leave quote null rather than "
        "guess at one."
    )
