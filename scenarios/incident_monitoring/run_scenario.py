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

SCENARIO_DIR = Path(__file__).resolve().parent
SOURCES_DIR = SCENARIO_DIR / "sources"

FABRICATED_CLAIM_TEXT = (
    "Vendor A's outage caused permanent data loss in us-east-2."
)
FABRICATED_CLAIM_QUOTE = (
    "permanent data loss occurred across all affected buckets in us-east-2"
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
    all_claims.append(fabricated)

    print("Verifying all claims (including one deliberately fabricated claim "
          "injected by this script, not produced by any model)...")
    verifier = Verifier(judge_client=client)
    verifications = verifier.verify_all(all_claims)

    for v in verifications:
        flag = " <-- INJECTED FABRICATION" if v.claim is fabricated else ""
        print(f"  [{v.verdict:10s}] ({v.method:12s}) {v.claim.text!r}{flag}")
        print(f"               reason: {v.reason}")

    fabricated_result = next(v for v in verifications if v.claim is fabricated)
    fabrication_caught = fabricated_result.verdict != "confirmed"

    print()
    if fabrication_caught:
        print(f"PASS: fabricated claim was NOT confirmed "
              f"(verdict={fabricated_result.verdict!r}, method={fabricated_result.method!r})")
    else:
        print("FAIL: fabricated claim was CONFIRMED. The verifier failed to catch it.")

    memory_db = SCENARIO_DIR / "memory.sqlite3"
    store = MemoryStore(db_path=str(memory_db))
    confirmed = [v for v in verifications if v.verdict == "confirmed" and v.claim is not fabricated]
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

    return 0 if fabrication_caught else 1


if __name__ == "__main__":
    raise SystemExit(main())
