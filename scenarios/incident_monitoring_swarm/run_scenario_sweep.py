"""Parameterized worker-count point for the Move-1 sweep (ADR-010): same 5
synthetic postmortems and the same 5 fabricated claims as
run_scenario_scale55.py, at a worker count passed on the command line,
so "does the rejection rate / fabrication catch rate / latency profile
hold at every scale" is measured on one consistent instrument instead of
five different setups.

Usage: python run_scenario_sweep.py <N>
N must be <= 5 (uses the first N base sources once each) or a multiple of
5 (uses all 5 base sources, N/5 repeats each) -- kept simple rather than
generalized, since the sweep only needs a handful of fixed points.

Writes to size-suffixed files (verification_log_scaleN.json,
memory_scaleN.sqlite3, html/ops/observability-scaleN.html) so it never
touches the canonical 55-worker artifacts written by run_scenario_scale55.py.
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

BASE_SOURCES = ["meridian_cache", "anchor_cdn", "solace_queue", "ridgeline_auth", "cobalt_storage"]

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


def build_tasks(n: int) -> list[WorkerTask]:
    if n <= len(BASE_SOURCES):
        bases_and_repeats = [(base, 1) for base in BASE_SOURCES[:n]]
    elif n % len(BASE_SOURCES) == 0:
        repeats = n // len(BASE_SOURCES)
        bases_and_repeats = [(base, repeats) for base in BASE_SOURCES]
    else:
        raise SystemExit(f"N={n} must be <= {len(BASE_SOURCES)} or a multiple of {len(BASE_SOURCES)}")

    tasks = []
    for base, repeats in bases_and_repeats:
        for i in range(1, repeats + 1):
            tasks.append(WorkerTask(
                worker_id=f"{base}_r{i}",
                instruction="Extract the key findings from this incident postmortem.",
                source_path=str(SOURCES_DIR / f"{base}_postmortem.txt"),
            ))
    return tasks


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python run_scenario_sweep.py <N>")
        return 2
    n = int(sys.argv[1])
    label = f"scale{n}"

    client = OllamaClient()
    if not client.is_available():
        print(f"Ollama is not reachable at {client.host}. Start it with `ollama serve`, then re-run.")
        return 2

    tasks = build_tasks(n)
    bases_used = sorted({t.worker_id.rsplit("_r", 1)[0] for t in tasks})
    print(f"[{label}] Built {len(tasks)} worker task(s) over {len(bases_used)} source(s): {bases_used}")

    worker = OllamaSimWorker(client=client)
    orchestrator = Orchestrator(worker=worker, synth_client=client)

    print(f"[{label}] Running extraction workers + synthesis...")
    result = orchestrator.run(tasks)

    all_claims: list[Claim] = [c for out in result.worker_outputs for c in out.claims]

    # Persisted before verify_all() runs -- extraction is the expensive,
    # remote-LLM-call-heavy phase; verification is comparatively cheap. If
    # the verifier crashes on a claim shape it's never seen before (as it
    # did on a live 200-worker run), this file is what stops that extraction
    # compute from being silently lost with the crashed process.
    import json
    extracted_path = SCENARIO_DIR / f"extracted_claims_{label}.json"
    extracted_path.write_text(
        json.dumps([
            {
                "worker_id": c.worker_id, "source": Path(c.source_path).name,
                "text": c.text, "quote": c.quote, "computation": c.computation,
            }
            for c in all_claims
        ], indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[{label}] Wrote {len(all_claims)} extracted claim(s) to {extracted_path} before verification.")

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

    print(f"[{label}] Verifying {len(all_claims)} claim(s) total, including {len(injected)} injected fabrications...")
    verifier = Verifier(judge_client=client)
    verifications = verifier.verify_all(all_claims)

    fabrication_results = [v for v in verifications if id(v.claim) in fabrication_ids]
    all_fabrications_caught = all(v.verdict != "confirmed" for v in fabrication_results)

    real = [v for v in verifications if id(v.claim) not in fabrication_ids]
    real_confirmed = sum(1 for v in real if v.verdict == "confirmed")
    real_rejected = sum(1 for v in real if v.verdict == "rejected")
    real_unverified = sum(1 for v in real if v.verdict == "unverified")
    print(f"[{label}] Real (non-injected) claims: {len(real)} total, {real_confirmed} confirmed, "
          f"{real_rejected} rejected, {real_unverified} unverified.")
    print(f"[{label}] Fabrications: {sum(1 for v in fabrication_results if v.verdict != 'confirmed')}/"
          f"{len(injected)} caught.")

    import json
    verification_log_path = SCENARIO_DIR / f"verification_log_{label}.json"
    verification_log_path.write_text(
        json.dumps([
            {
                "worker_id": v.claim.worker_id,
                "source": Path(v.claim.source_path).name,
                "text": v.claim.text,
                "quote": v.claim.quote,
                "computation": v.claim.computation,
                "verdict": v.verdict,
                "method": v.method,
                "reason": v.reason,
                "is_injected_fabrication": id(v.claim) in fabrication_ids,
            }
            for v in verifications
        ], indent=2),
        encoding="utf-8",
    )
    print(f"[{label}] Wrote verification log to {verification_log_path}")

    memory_db = SCENARIO_DIR / f"memory_{label}.sqlite3"
    store = MemoryStore(db_path=str(memory_db))
    confirmed = [v for v in verifications if v.verdict == "confirmed" and id(v.claim) not in fabrication_ids]
    for v in confirmed:
        entry = MemoryEntry(
            content=v.claim.text, scope=Scope.RUN, scope_id=f"incident_monitoring_swarm_{label}_demo",
            provenance=f"{v.claim.worker_id}:{v.claim.source_path}", confidence=1.0,
        )
        store.write(entry)
    print(f"[{label}] Wrote {len(confirmed)} confirmed claim(s) to Memory ({memory_db}).")

    total_tokens = sum((c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in client.calls if c.success)
    total_duration = sum(c.duration_s or 0.0 for c in client.calls)
    print(f"[{label}] Observability: {len(client.calls)} Ollama call(s), {total_tokens:,} total tokens, "
          f"{total_duration:.1f}s in-call wall time, across {len(tasks)} workers.")

    report_path = SCENARIO_DIR.parents[1] / "html" / "ops" / f"observability-{label}.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_observability_html(
            scenario_name=f"incident_monitoring_swarm sweep point (N={n} workers)",
            calls=client.calls,
            pass_fail=all_fabrications_caught,
            notes=(
                f"Move-1 worker-count sweep point, {len(tasks)} worker(s) over "
                f"{len(bases_used)} source document(s). Same 5 fabricated claims injected "
                f"regardless of N. Post-quote-fix code (ADR-009/ADR-010)."
            ),
        ),
        encoding="utf-8",
    )
    print(f"[{label}] Wrote observability report to {report_path}")

    return 0 if all_fabrications_caught else 1


if __name__ == "__main__":
    raise SystemExit(main())
