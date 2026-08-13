from __future__ import annotations
from dataclasses import dataclass

from .schema import MemoryEntry, Scope


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    reason: str


def validate(entry: MemoryEntry) -> GateResult:
    """Deterministic structural check. No LLM involved by design: an LLM
    deciding whether its own write is trustworthy is exactly the failure
    mode this gate exists to avoid.
    """
    if not entry.content or not entry.content.strip():
        return GateResult(False, "empty content")
    if not isinstance(entry.scope, Scope):
        return GateResult(False, f"invalid scope: {entry.scope!r}")
    if not entry.scope_id or not entry.scope_id.strip():
        return GateResult(False, "missing scope_id")
    if not entry.provenance or not entry.provenance.strip():
        return GateResult(False, "missing provenance")
    if not (0.0 <= entry.confidence <= 1.0):
        return GateResult(False, f"confidence out of range: {entry.confidence}")
    if entry.ttl_seconds is not None and entry.ttl_seconds <= 0:
        return GateResult(False, f"non-positive ttl_seconds: {entry.ttl_seconds}")
    if entry.scope is Scope.ORG and not entry.reviewed:
        return GateResult(False, "org-scope writes require review before merge (reviewed=False)")
    return GateResult(True, "ok")
