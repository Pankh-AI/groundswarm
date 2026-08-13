from __future__ import annotations
import sqlite3
import time

from .schema import MemoryEntry, Scope
from .gate import validate, GateResult


class MemoryWriteRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class MemoryStore:
    """Scoped, gated, TTL-aware shared memory store.

    Every write goes through the deterministic gate in gate.py before it is
    persisted. Reads are tiered: ORG-scope entries are meant to be always
    loaded, everything else is retrieved on demand and budgeted by an
    explicit token estimate, so shared context is a deliberate cost, not an
    accident.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                provenance TEXT NOT NULL,
                confidence REAL NOT NULL,
                ttl_seconds REAL,
                supersedes TEXT,
                reviewed INTEGER NOT NULL,
                created_at REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        self._conn.commit()

    def write(self, entry: MemoryEntry) -> GateResult:
        result = validate(entry)
        if not result.accepted:
            raise MemoryWriteRejected(result.reason)
        if entry.supersedes:
            self._conn.execute(
                "UPDATE entries SET active = 0 WHERE id = ?", (entry.supersedes,)
            )
        self._conn.execute(
            """
            INSERT INTO entries
                (id, content, scope, scope_id, provenance, confidence,
                 ttl_seconds, supersedes, reviewed, created_at, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                entry.id, entry.content, entry.scope.value, entry.scope_id,
                entry.provenance, entry.confidence, entry.ttl_seconds,
                entry.supersedes, int(entry.reviewed), entry.created_at,
            ),
        )
        self._conn.commit()
        return result

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"], content=row["content"], scope=Scope(row["scope"]),
            scope_id=row["scope_id"], provenance=row["provenance"],
            confidence=row["confidence"], ttl_seconds=row["ttl_seconds"],
            supersedes=row["supersedes"], reviewed=bool(row["reviewed"]),
            created_at=row["created_at"],
        )

    def active_entries(self, *, scope: Scope | None = None, scope_id: str | None = None,
                        now: float | None = None) -> list[MemoryEntry]:
        now = now if now is not None else time.time()
        clauses = ["active = 1"]
        params: list = []
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope.value)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        rows = self._conn.execute(
            f"SELECT * FROM entries WHERE {' AND '.join(clauses)} ORDER BY created_at ASC",
            params,
        ).fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        return [e for e in entries if not e.is_expired(now=now)]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def load_context(self, *, project_scope_id: str | None = None,
                      run_scope_id: str | None = None,
                      max_project_tokens: int = 2000,
                      max_run_tokens: int = 1000) -> dict[str, list[MemoryEntry]]:
        """Tiered load: ORG entries always come back in full (small, global
        tier). PROJECT and RUN entries are retrieved on demand and capped by
        an explicit token budget instead of being force-fed in full.
        """
        org_entries = self.active_entries(scope=Scope.ORG)

        project_entries: list[MemoryEntry] = []
        if project_scope_id is not None:
            budget = max_project_tokens
            for e in self.active_entries(scope=Scope.PROJECT, scope_id=project_scope_id):
                cost = self._estimate_tokens(e.content)
                if cost > budget:
                    break
                project_entries.append(e)
                budget -= cost

        run_entries: list[MemoryEntry] = []
        if run_scope_id is not None:
            budget = max_run_tokens
            for e in self.active_entries(scope=Scope.RUN, scope_id=run_scope_id):
                cost = self._estimate_tokens(e.content)
                if cost > budget:
                    break
                run_entries.append(e)
                budget -= cost

        return {"org": org_entries, "project": project_entries, "run": run_entries}
