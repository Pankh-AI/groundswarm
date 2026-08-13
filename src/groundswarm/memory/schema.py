from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid


class Scope(str, Enum):
    ORG = "org"
    TEAM = "team"
    PROJECT = "project"
    RUN = "run"

    @property
    def rank(self) -> int:
        return {"org": 0, "team": 1, "project": 2, "run": 3}[self.value]


@dataclass
class MemoryEntry:
    content: str
    scope: Scope
    scope_id: str
    provenance: str
    confidence: float = 1.0
    ttl_seconds: float | None = None
    supersedes: str | None = None
    reviewed: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    def is_expired(self, *, now: float | None = None) -> bool:
        if self.ttl_seconds is None:
            return False
        now = now if now is not None else time.time()
        return now >= (self.created_at + self.ttl_seconds)
