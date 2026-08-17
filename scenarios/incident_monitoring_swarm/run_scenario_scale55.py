"""Swarm-at-realistic-scale proof case: 55 workers (the same 5 synthetic
postmortems from run_scenario.py, each run 11 times) instead of 5 or 2.

Design choice, stated plainly: this reuses the same 5 source documents
repeated 11x each rather than authoring 55 unique incidents. That's
deliberate, not a shortcut taken to save effort at the cost of honesty: it
isolates model sampling stochasticity as the only variable behind each
worker call's behavior (does it include quotes or not, how many claims does
it extract), instead of conflating that with 55 different documents'
difficulty. The two-worker and five-worker runs already showed real
document-to-document variance; this run measures repeat-to-repeat variance
on IDENTICAL input, at a swarm-realistic call volume, which the smaller runs
didn't have enough samples to quantify.

Five fabricated claims are injected (one per base source, on its first
repeat), spread across a mix of fake-quote and no-quote fabrications, to
check the verifier's catch rate holds at this volume too.
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
REPORT_PATH = SCENARIO_DIR.parents[1] / "html" / "ops" / "observability-swarm55.html"

BASE_SOURCES = ["meridian_cache", "anchor_cdn", "solace_queue", "ridgeline_auth", "cobalt_storage"]
REPEATS = 11  # 5 * 11 = 55 workers

FABRICATIONS = [
    dict(base="meridian_cache",
         text="Meridian Cache's stale-content incident exposed customer payment data.",
         quote="customer payment data was exposed during the stale-content incident"),
    dict(base="anchor_cdn",
         text="Anchor CDN's certificate incident affected all customer hostnames, not just api.anchorcdn.net.",
         quote=None),
    dict(base="solace_queue",
         text="Solace Queue permanently lost 340,000 order-event messages during the backlog.",
         quote="340,000 order-event messages were permanently lost"),
    dict(base="ridgeline_auth",
         text="Ridgeline Auth's connection pool exhaustion caused a full site-wide outage lasting six hours.",
         quote=None),
    dict(base="cobalt_storage",
         text="Cobalt Storage lost customer data because only one replica of each object existed.",
         quote="only one replica of each object existed"),
]


def main() -> int:
    client = OllamaClient()
    if not client.is_available():
        print(f"Ollama is not reachable at {client.host}. Start it with `ollama serve`, then re-run.")
        return 2

    tasks = [
        WorkerTask(
            worker_id=f"{base}_r{i}",
            instruction="Extract the key findings from this incident postmortem.",
            source_path=str(SOURCES_DIR / f"{base}_postmortem.txt"),
        )
        for base in BASE_SOURCES
        for i in range(1, REPEATS + 1)
    ]
    print(f"Built {len(tasks)} worker tasks ({len(BASE_SOURCES)} sources x {REPEATS} repeats).")

    worker = OllamaSimWorker(client=client)
    orchestrator = Orchestrator(worker=worker, synth_client=client)

    print("Running extraction workers + synthesis (this will take a while)...")
    result = orchestrator.run(tasks)

    # Per-base-source quote-inclusion stats: for each worker call, did it
    # include a quote for ALL of its claims, NONE, or a MIX?
    from collections import defaultdict
    stats = defaultdict(lambda: {"calls": 0, "claims": 0, "quoted_claims": 0,
                                   "all_quoted_calls": 0, "no_quote_calls": 0, "mixed_calls": 0})
    for out in result.worker_outputs:
        base = out.worker_id.rsplit("_r", 1)[0]
        s = stats[base]
        s["calls"] += 1
        n = len(out.claims)
        q = sum(1 for c in out.claims if c.quote)
        s["claims"] += n
        s["quoted_claims"] += q
        if n == 0:
            continue
        if q == n:
            s["all_quoted_calls"] += 1
        elif q == 0:
            s["no_quote_calls"] += 1
        else:
            s["mixed_calls"] += 1

    print("\nPer-source quote-inclusion behavior across repeats:")
    for base in BASE_SOURCES:
        s = stats[base]
        print(f"  {base}: {s['calls']} calls, {s['claims']} claims total, "
              f"{s['quoted_claims']} with a quote ({s['quoted_claims']/max(s['claims'],1)*100:.0f}%) | "
              f"calls: {s['all_quoted_calls']} all-quoted, {s['no_quote_calls']} zero-quote, "
              f"{s['mixed_calls']} mixed")

    all_claims: list[Claim] = [c for out in result.worker_outputs for c in out.claims]

    injected: list[Claim] = []
    for f in FABRICATIONS:
        claim = Claim(
            text=f["text"], quote=f["quote"],
            source_path=str(SOURCES_DIR / f"{f['base']}_postmortem.txt"),
            worker_id="injected_by_script",
        )
        all_claims.append(claim)
        injected.append(claim)
    fabrication_ids = {id(c) for c in injected}

    print(f"\nVerifying {len(all_claims)} claim(s) total, including {len(injected)} injected fabrications...")
    verifier = Verifier(judge_client=client)
    verifications = verifier.verify_all(all_claims)

    fabrication_results = [v for v in verifications if id(v.claim) in fabrication_ids]
    all_fabrications_caught = all(v.verdict != "confirmed" for v in fabrication_results)

    print()
    for v in fabrication_results:
        status = "caught" if v.verdict != "confirmed" else "MISSED"
        print(f"  {status}: (verdict={v.verdict!r}, method={v.method!r}) {v.claim.text!r}")

    real = [v for v in verifications if id(v.claim) not in fabrication_ids]
    real_confirmed = sum(1 for v in real if v.verdict == "confirmed")
    real_rejected = sum(1 for v in real if v.verdict == "rejected")
    real_unverified = sum(1 for v in real if v.verdict == "unverified")
    print(f"\nReal (non-injected) claims: {len(real)} total, {real_confirmed} confirmed, "
          f"{real_rejected} rejected, {real_unverified} unverified.")

    if all_fabrications_caught:
        print(f"\nPASS: all {len(injected)} fabricated claims were NOT confirmed")
    else:
        missed = sum(1 for v in fabrication_results if v.verdict == "confirmed")
        print(f"\nFAIL: {missed} of {len(injected)} fabricated claim(s) were CONFIRMED.")

    import json
    verification_log_path = SCENARIO_DIR / "verification_log.json"
    verification_log_path.write_text(
        json.dumps([
            {
                "worker_id": v.claim.worker_id,
                "source": Path(v.claim.source_path).name,
                "text": v.claim.text,
                "quote": v.claim.quote,
                "verdict": v.verdict,
                "method": v.method,
                "reason": v.reason,
                "is_injected_fabrication": id(v.claim) in fabrication_ids,
            }
            for v in verifications
        ], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote full verification log ({len(verifications)} entries, confirmed and rejected alike) "
          f"to {verification_log_path}")

    memory_db = SCENARIO_DIR / "memory.sqlite3"
    store = MemoryStore(db_path=str(memory_db))
    confirmed = [v for v in verifications if v.verdict == "confirmed" and id(v.claim) not in fabrication_ids]
    for v in confirmed:
        entry = MemoryEntry(
            content=v.claim.text, scope=Scope.RUN, scope_id="incident_monitoring_swarm55_demo",
            provenance=f"{v.claim.worker_id}:{v.claim.source_path}", confidence=1.0,
        )
        store.write(entry)
    print(f"\nWrote {len(confirmed)} confirmed claim(s) to Memory ({memory_db}).")

    total_tokens = sum((c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in client.calls if c.success)
    total_duration = sum(c.duration_s or 0.0 for c in client.calls)
    print(f"\nObservability: {len(client.calls)} Ollama call(s), {total_tokens:,} total tokens, "
          f"{total_duration:.1f}s in-call wall time, across {len(tasks)} workers.")

    quote_notes = "<br>".join(
        f"<b>{base}</b>: {stats[base]['quoted_claims']}/{stats[base]['claims']} claims quoted "
        f"({stats[base]['all_quoted_calls']} all-quoted / {stats[base]['no_quote_calls']} zero-quote / "
        f"{stats[base]['mixed_calls']} mixed calls, of {stats[base]['calls']} repeats)"
        for base in BASE_SOURCES
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        render_observability_html(
            scenario_name=f"incident_monitoring_swarm55 ({len(tasks)} workers)",
            calls=client.calls,
            pass_fail=all_fabrications_caught,
            notes=(
                f"{len(tasks)} workers = {len(BASE_SOURCES)} source documents x {REPEATS} repeats each "
                f"(identical input per repeat, isolating model sampling stochasticity). "
                f"{len(injected)} fabricated claims injected, one per source. "
                f"Compare against the <a href='./observability.html'>2-worker</a> and "
                f"<a href='./observability-swarm.html'>5-worker</a> reports.<br><br>"
                f"<b>Quote-inclusion rate per source (n={REPEATS} repeats each):</b><br>{quote_notes}"
            ),
        ),
        encoding="utf-8",
    )
    print(f"Wrote observability report to {REPORT_PATH}")

    return 0 if all_fabrications_caught else 1


if __name__ == "__main__":
    raise SystemExit(main())
