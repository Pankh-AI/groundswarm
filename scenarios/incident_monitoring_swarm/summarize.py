"""Aggregate-stats summary for a verification_log*.json file: real-claim
verdict counts, rejection rate, rejected-claims-by-method breakdown, and
fabrication catch rate. Built to make the Move-1 before/after and
worker-count-sweep comparisons mechanical instead of re-derived by hand
each time.

Usage: python summarize.py <verification_log.json> [label]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> dict:
    entries = json.loads(path.read_text(encoding="utf-8"))
    real = [e for e in entries if not e.get("is_injected_fabrication")]
    fabrications = [e for e in entries if e.get("is_injected_fabrication")]

    verdicts = Counter(e["verdict"] for e in real)
    rejected = [e for e in real if e["verdict"] == "rejected"]
    rejected_by_method = Counter(e["method"] for e in rejected)
    fab_caught = sum(1 for e in fabrications if e["verdict"] != "confirmed")

    total_tokens = None  # not stored per-entry in the log; comes from the observability report instead

    return {
        "path": str(path),
        "real_total": len(real),
        "confirmed": verdicts.get("confirmed", 0),
        "rejected": verdicts.get("rejected", 0),
        "unverified": verdicts.get("unverified", 0),
        "rejection_rate_pct": round(100 * verdicts.get("rejected", 0) / len(real), 1) if real else None,
        "rejected_by_method": dict(rejected_by_method),
        "fabrications_total": len(fabrications),
        "fabrications_caught": fab_caught,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python summarize.py <verification_log.json> [label]")
        return 2
    path = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else path.name
    result = summarize(path)
    print(f"=== {label} ===")
    for k, v in result.items():
        if k == "path":
            continue
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
