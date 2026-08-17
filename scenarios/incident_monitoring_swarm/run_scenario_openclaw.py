"""Live symbiosis demo (ADR-018/019): Groundswarm manages a real OpenClaw
Gateway's full lifecycle, dispatches the same 5-source swarm through it
instead of a bare Ollama call, verifies independently, then closes the loop
-- the worst-performing worker's next dispatch is re-run with a caution note
generated straight from its own verification tally, through the SAME worker
identity/session, and the before/after is measured, not asserted.

Requires OPENCLAW_REPO (path to a built OpenClaw checkout) and
OPENCLAW_GATEWAY_TOKEN (the dev profile's gateway.auth.token) in the
environment -- deliberately not hardcoded here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from groundswarm.ops.openclaw_manager import OpenClawGatewayManager, OpenClawGatewayError
from groundswarm.ops.feedback import CAUTION_THRESHOLD, compute_reliability, caution_note
from groundswarm.workers.base import WorkerTask
from groundswarm.workers.openclaw_worker import OpenClawWorker
from groundswarm.orchestrator.orchestrator import Orchestrator
from groundswarm.verifier.verifier import Verifier, VerificationResult

SCENARIO_DIR = Path(__file__).resolve().parent
SOURCES_DIR = SCENARIO_DIR / "sources"

WORKERS = ["meridian_cache", "anchor_cdn", "solace_queue", "ridgeline_auth", "cobalt_storage"]


def _summarize(label: str, verifications: list[VerificationResult]) -> None:
    confirmed = sum(1 for v in verifications if v.verdict == "confirmed")
    rejected = sum(1 for v in verifications if v.verdict == "rejected")
    unverified = sum(1 for v in verifications if v.verdict == "unverified")
    print(f"  {label}: {len(verifications)} claim(s) -> "
          f"{confirmed} confirmed, {rejected} rejected, {unverified} unverified")


def main() -> int:
    token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "").strip()
    if not token:
        print("OPENCLAW_GATEWAY_TOKEN is not set. Export the dev profile's gateway.auth.token first.")
        return 2

    tasks = [
        WorkerTask(
            worker_id=w,
            instruction="Extract the key findings from this incident postmortem as claims.",
            source_path=str(SOURCES_DIR / f"{w}_postmortem.txt"),
        )
        for w in WORKERS
    ]

    try:
        manager = OpenClawGatewayManager()
    except OpenClawGatewayError as exc:
        print(f"Could not construct the Gateway manager: {exc}")
        return 2

    print("Starting OpenClaw Gateway (managed by Groundswarm)...")
    try:
        with manager:
            print(f"Gateway ready at {manager.base_url}")
            worker = OpenClawWorker(base_url=manager.base_url, token=token)
            orchestrator = Orchestrator(worker=worker)
            verifier = Verifier()

            print(f"\n=== Pass 1: dispatching {len(tasks)} workers through the real OpenClaw agent ===")
            result = orchestrator.run(tasks)
            for out in result.worker_outputs:
                print(f"  [{out.worker_id}] extracted {len(out.claims)} claim(s)")
            print(f"\nSynthesized briefing ({len(result.briefing)} chars):\n{result.briefing}\n")

            all_claims = [c for out in result.worker_outputs for c in out.claims]
            if not all_claims:
                print("No claims extracted in pass 1 -- nothing to verify or feed back. Check the Gateway log.")
                return 1

            pass1_verifications = verifier.verify_all(all_claims)
            print(f"\nVerifying {len(all_claims)} claim(s) from pass 1...")
            _summarize("Pass 1 (all workers)", pass1_verifications)

            reliability = compute_reliability(pass1_verifications)
            for w in WORKERS:
                rel = reliability.get(w)
                if rel is None:
                    print(f"  [{w}] no claims extracted -- excluded from reliability ranking")
                    continue
                frac = rel.confirmed_fraction
                frac_str = f"{frac:.0%}" if frac is not None else "n/a"
                print(f"  [{w}] {rel.confirmed}/{rel.total} confirmed ({frac_str})")

            candidates = [r for r in reliability.values() if r.needs_caution]
            print(f"\n=== Feedback: {len(candidates)} of {len(reliability)} worker(s) fell below "
                  f"the {CAUTION_THRESHOLD:.0%} caution threshold ===")

            if not candidates:
                print("No worker needs caution this run -- the loop has nothing to act on. "
                      "This is a real, reportable outcome (small-sample run, not a validated null result), "
                      "not a failure of the mechanism.")
                worst = None
            else:
                worst = min(candidates, key=lambda r: (r.confirmed_fraction if r.confirmed_fraction is not None else 1.0))
                note = caution_note(worst)
                print(f"Worst performer: [{worst.worker_id}] {worst.confirmed}/{worst.total} confirmed. "
                      f"Injecting caution note into its next dispatch:\n  {note!r}")
                worker.source_notes[worst.worker_id] = note

                print(f"\n=== Pass 2: re-dispatching [{worst.worker_id}] through the SAME worker, "
                      f"now carrying the caution note ===")
                pass2_task = next(t for t in tasks if t.worker_id == worst.worker_id)
                pass2_output = worker.run(pass2_task)
                print(f"  [{pass2_output.worker_id}] extracted {len(pass2_output.claims)} claim(s) on re-dispatch")
                pass2_verifications = verifier.verify_all(pass2_output.claims)
                _summarize(f"Pass 2 ([{worst.worker_id}] only, cautioned)", pass2_verifications)

                pass1_for_worst = [v for v in pass1_verifications if v.claim.worker_id == worst.worker_id]
                print(f"\n=== Before/after for [{worst.worker_id}] ===")
                _summarize("  Before (pass 1, no caution)", pass1_for_worst)
                _summarize("  After  (pass 2, cautioned)  ", pass2_verifications)

            total_calls = len(worker.calls)
            total_tokens = sum((c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in worker.calls if c.success)
            total_duration = sum(c.duration_s or 0.0 for c in worker.calls)
            print(f"\nObservability: {total_calls} OpenClaw agent call(s), {total_tokens:,} total tokens, "
                  f"{total_duration:.1f}s in-call wall time.")

            return 0
    except OpenClawGatewayError as exc:
        print(f"Gateway lifecycle error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
