"""End-to-end proof case: incident monitoring.

Two workers each extract findings from one vendor postmortem. A synthesis
worker writes a combined risk briefing. The Verifier independently checks
every claim, including one claim injected here by the script itself (not
produced by any model) that misstates one of the postmortems. If the
Verifier is doing its job, that claim gets rejected or marked unverified,
never confirmed. Only confirmed claims get written to the run-scoped
Memory store.
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
REPORT_PATH = SCENARIO_DIR.parents[1] / "html" / "ops" / "observability.html"

FABRICATED_CLAIM_TEXT = (
    "Vendor A's outage caused permanent data loss in us-east-2."
)
FABRICATED_CLAIM_QUOTE = (
    "permanent data loss occurred across all affected buckets in us-east-2"
)

# A second fabrication with no quote, so it can only be caught by the LLM
# judge fallback, not the deterministic check. The deterministic path
# already gets exercised by FABRICATED_CLAIM_TEXT above; this one exists
# specifically to prove the judge path also fails closed on a real model,
# not just in the mocked unit tests.
FABRICATED_CLAIM_2_TEXT = (
    "Vendor B's automatic failover completed in under 10 seconds, faster "
    "than its 30 second target."
)


def main() -> int:
    client = OllamaClient()
    if not client.is_available():
        print(f"Ollama is not reachable at {client.host}. Start it with `ollama serve` "
              f"(or the Ollama app) and make sure a model is pulled, then re-run this script.")
        return 2

    tasks = [
        WorkerTask(
            worker_id="worker_vendor_a",
            instruction="Extract the key findings from this incident postmortem.",
            source_path=str(SOURCES_DIR / "vendor_a_postmortem.txt"),
        ),
        WorkerTask(
            worker_id="worker_vendor_b",
            instruction="Extract the key findings from this incident postmortem.",
            source_path=str(SOURCES_DIR / "vendor_b_postmortem.txt"),
        ),
    ]

    worker = OllamaSimWorker(client=client)
    orchestrator = Orchestrator(worker=worker, synth_client=client)

    print("Running extraction workers + synthesis...")
    result = orchestrator.run(tasks)

    for out in result.worker_outputs:
        print(f"\n[{out.worker_id}] extracted {len(out.claims)} claim(s):")
        for c in out.claims:
            print(f"  - {c.text!r} (quote={c.quote!r})")

    print(f"\nSynthesized briefing:\n{result.briefing}\n")

    all_claims: list[Claim] = [c for out in result.worker_outputs for c in out.claims]

    fabricated = Claim(
        text=FABRICATED_CLAIM_TEXT,
        quote=FABRICATED_CLAIM_QUOTE,
        source_path=str(SOURCES_DIR / "vendor_a_postmortem.txt"),
        worker_id="injected_by_script",
    )
    fabricated_2 = Claim(
        text=FABRICATED_CLAIM_2_TEXT,
        quote=None,
        source_path=str(SOURCES_DIR / "vendor_b_postmortem.txt"),
        worker_id="injected_by_script",
    )
    all_claims.append(fabricated)
    all_claims.append(fabricated_2)
    # Claim is a plain (unfrozen) dataclass and so is unhashable; track the
    # two injected claims by identity via id() instead of putting them in a
    # set directly.
    fabrication_ids = {id(fabricated), id(fabricated_2)}

    print("Verifying all claims (including two deliberately fabricated claims "
          "injected by this script, not produced by any model: one with a "
          "fake quote to test the deterministic check, one with no quote to "
          "test the LLM-judge fallback)...")
    verifier = Verifier(judge_client=client)
    verifications = verifier.verify_all(all_claims)

    for v in verifications:
        flag = " <-- INJECTED FABRICATION" if id(v.claim) in fabrication_ids else ""
        print(f"  [{v.verdict:10s}] ({v.method:12s}) {v.claim.text!r}{flag}")
        print(f"               reason: {v.reason}")

    fabrication_results = [v for v in verifications if id(v.claim) in fabrication_ids]
    all_fabrications_caught = all(v.verdict != "confirmed" for v in fabrication_results)

    print()
    for v in fabrication_results:
        status = "caught" if v.verdict != "confirmed" else "MISSED"
        print(f"  {status}: (verdict={v.verdict!r}, method={v.method!r}) {v.claim.text!r}")

    if all_fabrications_caught:
        print("\nPASS: both fabricated claims were NOT confirmed")
    else:
        print("\nFAIL: at least one fabricated claim was CONFIRMED. The verifier failed to catch it.")

    memory_db = SCENARIO_DIR / "memory.sqlite3"
    store = MemoryStore(db_path=str(memory_db))
    confirmed = [v for v in verifications if v.verdict == "confirmed" and id(v.claim) not in fabrication_ids]
    for v in confirmed:
        entry = MemoryEntry(
            content=v.claim.text,
            scope=Scope.RUN,
            scope_id="incident_monitoring_demo",
            provenance=f"{v.claim.worker_id}:{v.claim.source_path}",
            confidence=1.0,
        )
        store.write(entry)
    print(f"\nWrote {len(confirmed)} confirmed claim(s) to Memory ({memory_db}).")

    total_tokens = sum((c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in client.calls if c.success)
    total_duration = sum(c.duration_s or 0.0 for c in client.calls)
    print(f"\nObservability: {len(client.calls)} Ollama call(s), {total_tokens:,} total tokens, "
          f"{total_duration:.1f}s in-call wall time.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        render_observability_html(
            scenario_name="incident_monitoring",
            calls=client.calls,
            pass_fail=all_fabrications_caught,
            notes=(
                "Two fabricated claims were injected by this script (not produced by any model): "
                "one with a fake quote (caught by the deterministic check, zero extra Ollama calls), "
                "one with no quote (caught, or not, by the <code>judge:injected_by_script</code> "
                "call(s) in the table above)."
            ),
        ),
        encoding="utf-8",
    )
    print(f"Wrote observability report to {REPORT_PATH}")

    return 0 if all_fabrications_caught else 1


if __name__ == "__main__":
    raise SystemExit(main())
