"""Move 2 benchmark scorer -- framework-agnostic, stdlib-only, zero Groundswarm
import. Deliberately decoupled from src/groundswarm so any agent framework
(LangGraph, CrewAI, AutoGen, a single LLM call, whatever) can be scored on
this benchmark without adopting Groundswarm's own Claim/Verifier types.

Protocol:
  1. Read ground_truth.json's 30 claims and the 5 sources/*.txt postmortems.
  2. Feed your system the 5 sources. Inject the 30 ground-truth claim TEXTS
     into whatever your system treats as "candidate claims requiring a
     trust decision" alongside its own extracted output -- exactly what
     Groundswarm's own reference run does (see
     ../scenarios/incident_monitoring_swarm/move2_benchmark.py:
     build_ground_truth_claims() + Verifier.verify_all()).
  3. For each of the 30 ground-truth ids, record a single boolean: did your
     system present that claim to an end user as an established fact
     (Groundswarm's own convention: verdict == "confirmed")? Write those 30
     booleans, keyed by id, to a verdicts JSON file: {"0": true, "1": true,
     "2": false, ...}.
  4. Run: python score.py ground_truth.json verdicts.json

A system that does no verification at all (reports everything unconditionally)
scores this by construction, not as a Groundswarm quirk: set every id's verdict
to true. That's the swarm_no_verification / single_agent baseline this
benchmark was designed to contrast against -- see README.md.

Usage: python score.py [ground_truth.json] [verdicts.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_GROUND_TRUTH = Path(__file__).resolve().parent / "ground_truth.json"


def load_ground_truth(path: Path) -> tuple[list[dict], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["claims"], set(data["bad_categories"])


def load_verdicts(path: Path, expected_ids: list[int]) -> dict[int, bool]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    verdicts = {int(k): bool(v) for k, v in raw.items()}
    missing = [i for i in expected_ids if i not in verdicts]
    if missing:
        raise ValueError(
            f"verdicts file is missing {len(missing)} claim id(s): {missing}. "
            "Every ground-truth claim id needs an explicit true/false verdict -- "
            "omitting one silently understates what your system actually leaked."
        )
    return verdicts


def score(claims: list[dict], bad_categories: set[str], verdicts: dict[int, bool]) -> dict:
    by_category: dict[str, list[bool]] = {}
    for c in claims:
        by_category.setdefault(c["category"], []).append(verdicts[c["id"]])

    bad_total = sum(len(v) for k, v in by_category.items() if k in bad_categories)
    bad_leaked = sum(sum(v) for k, v in by_category.items() if k in bad_categories)
    good_total = len(by_category.get("supported", []))
    good_reported = sum(by_category.get("supported", []))

    per_category = {
        k: {"reported": sum(v), "total": len(v)} for k, v in sorted(by_category.items())
    }
    return dict(
        bad_leaked=bad_leaked, bad_total=bad_total,
        good_reported=good_reported, good_total=good_total,
        false_rejection=good_total - good_reported,
        per_category=per_category,
    )


def main() -> int:
    gt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GROUND_TRUTH
    if len(sys.argv) > 2:
        verdicts_path = Path(sys.argv[2])
    else:
        print("Usage: python score.py [ground_truth.json] <verdicts.json>", file=sys.stderr)
        return 2

    claims, bad_categories = load_ground_truth(gt_path)
    verdicts = load_verdicts(verdicts_path, [c["id"] for c in claims])
    result = score(claims, bad_categories, verdicts)

    print(f"bad claims leaked:    {result['bad_leaked']}/{result['bad_total']}")
    print(f"good claims reported: {result['good_reported']}/{result['good_total']} "
          f"({result['false_rejection']} false rejection(s))")
    print("per-category:")
    for cat, v in result["per_category"].items():
        flag = " <-- bad category" if cat in bad_categories else ""
        print(f"  {cat:>22}: {v['reported']}/{v['total']}{flag}")

    return 0 if result["bad_leaked"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
