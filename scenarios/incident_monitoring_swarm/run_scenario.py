"""Swarm-scale proof case: five independent extraction workers, one synthesis
step, and one verifier checking every claim, run against five distinct
synthetic incident postmortems (not the two-worker toy case in
scenarios/incident_monitoring/). Three fabricated claims are injected across
three different sources by the script itself (not produced by any model) to
check the verifier still catches everything at swarm scale, not just in the
minimal case. Exists to produce real token/latency numbers at a size closer
to an actual multi-worker swarm than the original two-worker demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from groundswarm.llm.ollama_client import OllamaClient
from groundswarm.workers.base import Claim, WorkerTask
from groundswarm.workers.sim_worker import OllamaSimWorker
from groundswarm.orchestrator.orchestrator import Orchestrator
from groundswarm.verifier.verifier import Verifier
from groundswarm.memory.schema import MemoryEntry, Scope
from groundswarm.memory.store import MemoryStore
from groundswarm.observability import render_observability_html

SCENARIO_DIR = Path(__file__).resolve().parent
SOURCES_DIR = SCENARIO_DIR / "sources"
REPORT_PATH = SCENARIO_DIR.parents[1] / "html" / "ops" / "observability-swarm.html"

WORKERS = ["meridian_cache", "anchor_cdn", "solace_queue", "ridgeline_auth", "cobalt_storage"]

# Three fabrications injected by this script, spread across three different
# sources, mixing both check paths: two carry a fake quote (deterministic
# check should catch both), one carries no quote (only the LLM judge can
# catch it). None of these were produced by any worker or the synth step.
FABRICATIONS = [
    dict(
        text="Meridian Cache's stale-content incident exposed customer payment data.",
        quote="customer payment data was exposed during the stale-content incident",
        worker_id="meridian_cache",
    ),
    dict(
        text="Anchor CDN's certificate incident affected all customer hostnames, not just api.anchorcdn.net.",
        quote=None,
        worker_id="anchor_cdn",
    ),
    dict(
        text="Ridgeline Auth's connection pool exhaustion caused a full site-wide outage lasting six hours.",
        quote="the connection pool exhaustion caused a full site-wide outage lasting six hours",
        worker_id="ridgeline_auth",
    ),
]


def main() -> int:
    client = OllamaClient()
    if not client.is_available():
        print(f"Ollama is not reachable at {client.host}. Start it with `ollama serve` "
              f"(or the Ollama app) and make sure a model is pulled, then re-run this script.")
        return 2

    tasks = [
        WorkerTask(
            worker_id=w,
            instruction="Extract the key findings from this incident postmortem.",
            source_path=str(SOURCES_DIR / f"{w}_postmortem.txt"),
        )
        for w in WORKERS
    ]

    worker = OllamaSimWorker(client=client)
    orchestrator = Orchestrator(worker=worker, synth_client=client)

    print(f"Running {len(tasks)} extraction workers + synthesis...")
    result = orchestrator.run(tasks)

    for out in result.worker_outputs:
        print(f"  [{out.worker_id}] extracted {len(out.claims)} claim(s)")

    print(f"\nSynthesized briefing ({len(result.briefing)} chars):\n{result.briefing}\n")

    all_claims: list[Claim] = [c for out in result.worker_outputs for c in out.claims]

    injected: list[Claim] = []
    for f in FABRICATIONS:
        source_name = f["worker_id"]
        claim = Claim(
            text=f["text"],
            quote=f["quote"],
            source_path=str(SOURCES_DIR / f"{source_name}_postmortem.txt"),
            worker_id="injected_by_script",
        )
        all_claims.append(claim)
        injected.append(claim)
    # Claim is a plain (unfrozen) dataclass and so is unhashable; track the
    # injected claims by identity via id() instead of putting them in a set.
    fabrication_ids = {id(c) for c in injected}

    print(f"Verifying {len(all_claims)} claim(s) total, including {len(injected)} "
          f"deliberately fabricated claims injected across {len(FABRICATIONS)} different sources...")
    verifier = Verifier(judge_client=client)
    verifications = verifier.verify_all(all_claims)

    fabrication_results = [v for v in verifications if id(v.claim) in fabrication_ids]
    all_fabrications_caught = all(v.verdict != "confirmed" for v in fabrication_results)

    print()
    for v in fabrication_results:
        status = "caught" if v.verdict != "confirmed" else "MISSED"
        print(f"  {status}: (verdict={v.verdict!r}, method={v.method!r}) {v.claim.text!r}")

    real_claims = [v for v in verifications if id(v.claim) not in fabrication_ids]
    real_confirmed = sum(1 for v in real_claims if v.verdict == "confirmed")
    real_rejected = sum(1 for v in real_claims if v.verdict == "rejected")
    real_unverified = sum(1 for v in real_claims if v.verdict == "unverified")
    print(f"\nReal (non-injected) claims: {len(real_claims)} total, "
          f"{real_confirmed} confirmed, {real_rejected} rejected, {real_unverified} unverified.")

    if all_fabrications_caught:
        print(f"\nPASS: all {len(injected)} fabricated claims (across {len(FABRICATIONS)} sources) were NOT confirmed")
    else:
        missed = sum(1 for v in fabrication_results if v.verdict == "confirmed")
        print(f"\nFAIL: {missed} of {len(injected)} fabricated claim(s) were CONFIRMED. The verifier failed to catch them.")

    memory_db = SCENARIO_DIR / "memory.sqlite3"
    store = MemoryStore(db_path=str(memory_db))
    confirmed = [v for v in verifications if v.verdict == "confirmed" and id(v.claim) not in fabrication_ids]
    for v in confirmed:
        entry = MemoryEntry(
            content=v.claim.text,
            scope=Scope.RUN,
            scope_id="incident_monitoring_swarm_demo",
            provenance=f"{v.claim.worker_id}:{v.claim.source_path}",
            confidence=1.0,
        )
        store.write(entry)
    print(f"\nWrote {len(confirmed)} confirmed claim(s) to Memory ({memory_db}).")

    total_tokens = sum((c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in client.calls if c.success)
    total_duration = sum(c.duration_s or 0.0 for c in client.calls)
    print(f"\nObservability: {len(client.calls)} Ollama call(s), {total_tokens:,} total tokens, "
          f"{total_duration:.1f}s in-call wall time, across {len(tasks)} workers.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        render_observability_html(
            scenario_name="incident_monitoring_swarm (5 workers)",
            calls=client.calls,
            pass_fail=all_fabrications_caught,
            notes=(
                f"Swarm-scale run: {len(tasks)} independent workers over 5 distinct synthetic "
                f"incident postmortems, {len(all_claims)} total claims verified. Three fabricated "
                f"claims were injected by this script (not produced by any model) across three "
                f"different sources: two with a fake quote (deterministic check), one with no "
                f"quote (LLM-judge fallback). Compare against the two-worker toy case's "
                f"<a href='./observability.html'>observability report</a> to see how cost scales "
                f"with worker/claim count."
            ),
        ),
        encoding="utf-8",
    )
    print(f"Wrote observability report to {REPORT_PATH}")

    return 0 if all_fabrications_caught else 1


if __name__ == "__main__":
    raise SystemExit(main())
