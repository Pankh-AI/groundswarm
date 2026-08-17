"""Move 2 benchmark (ADR-010 / ADR-012): "Verified Beats Plausible".

Compares three conditions over the SAME 5 incident postmortems already used by
the rest of this scenario:

  A. groundswarm           -- swarm extraction (1 worker per source) + Verifier;
                               only claims with verdict == "confirmed" are
                               counted as "reported" (shown to an end user as
                               fact).
  B. swarm_no_verification -- the IDENTICAL extraction output as (A) -- same
                               worker outputs, same LLM calls, zero re-sampling
                               -- but every claim is reported unconditionally,
                               with the Verifier never invoked. Isolates
                               verification as the only variable between A/B.
  C. single_agent           -- one LLM call across all 5 sources at once (no
                               worker decomposition, no verification); every
                               claim reported unconditionally.

Ground truth is NOT assigned by grading the model's own free-form claims with
a second LLM call -- that is an oracle scoring its own model family's output,
which the project's own methodology treats as a known inflation trap (see
[[oracle-inflation-split-half]] in the operator's research-methodology
memory, and the "no exogenous trace" style caution against trusting a second
model pass as ground truth). Instead, a FIXED, hand-authored corpus of 30
claims (5 sources x 6 claims/source: 2 supported, 1 unsupported, 1
unverifiable, 1 wrong_attribution, 1 incorrect_computation) is folded into
every condition's claim pool before scoring. Every quote and computation
operand below was checked character-for-character against the live source
files at authoring time -- ground truth here is true by construction, not by
a second model's opinion.

Each condition's REAL (non-injected) extraction output is separately checked
against a fixed per-source checklist (the 2 "supported" ground-truth claims)
via keyword coverage, to estimate an omission rate -- orthogonal to
verification, since the Verifier can only ever judge claims that were
actually extracted in the first place.

Usage: python move2_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from groundswarm.llm.ollama_client import OllamaClient, OllamaUnavailable
from groundswarm.workers.base import Claim, WorkerTask
from groundswarm.workers.sim_worker import OllamaSimWorker, EXTRACTION_SYSTEM_PROMPT
from groundswarm.orchestrator.orchestrator import Orchestrator
from groundswarm.verifier.verifier import Verifier
from groundswarm.observability import render_observability_html

SCENARIO_DIR = Path(__file__).resolve().parent
SOURCES_DIR = SCENARIO_DIR / "sources"

BASES = ["anchor_cdn", "cobalt_storage", "meridian_cache", "ridgeline_auth", "solace_queue"]


def _src(base: str) -> str:
    return str(SOURCES_DIR / f"{base}_postmortem.txt")


# ---------------------------------------------------------------------------
# Ground-truth corpus. category is never passed to the Verifier or to any
# LLM call -- it exists only for post-hoc scoring in this script.
# ---------------------------------------------------------------------------
GROUND_TRUTH = [
    # --- anchor_cdn -----------------------------------------------------
    dict(base="anchor_cdn", category="supported",
         text="Anchor CDN's certificate incident affected only the api.anchorcdn.net hostname; other Anchor CDN hostnames were unaffected.",
         quote="traffic to other Anchor CDN hostnames was unaffected", computation=None,
         checklist_keywords=["other anchor cdn hostnames", "unaffected"]),
    dict(base="anchor_cdn", category="supported",
         text="15 minutes elapsed between Anchor CDN's on-call page and full edge rollout of the new certificate.",
         quote=None, computation={"operation": "duration_minutes",
             "operands": [{"value": "06:19", "quote": "paged by an external uptime monitor at 06:19 UTC"},
                          {"value": "06:34", "quote": "completed rollout to all edge nodes by 06:34 UTC"}],
             "result": 15.0},
         checklist_keywords=["06:19", "06:34", "rollout"]),
    dict(base="anchor_cdn", category="unsupported",
         text="Anchor CDN lost customer credentials stored in the affected certificate.",
         quote=None, computation=None, checklist_keywords=None),
    dict(base="anchor_cdn", category="unverifiable",
         text="The expired API token belonged to a contractor account rather than an internal service account.",
         quote=None, computation=None, checklist_keywords=None),
    dict(base="anchor_cdn", category="wrong_attribution",
         text="Anchor CDN's incident was detected by an automated staleness monitor.",  # true fact, belongs to meridian_cache
         quote="detected by an automated staleness monitor", computation=None, checklist_keywords=None),
    dict(base="anchor_cdn", category="incorrect_computation",
         text="Anchor CDN's certificate failure lasted 44 minutes, from expiry to full edge rollout.",
         quote=None, computation={"operation": "duration_minutes",
             "operands": [{"value": "06:00", "quote": "At 06:00 UTC on 2026-06-22"},
                          {"value": "06:34", "quote": "completed rollout to all edge nodes by 06:34 UTC"}],
             "result": 44.0},  # correct answer is 34
         checklist_keywords=None),

    # --- cobalt_storage ---------------------------------------------------
    dict(base="cobalt_storage", category="supported",
         text="No write requests were affected by Cobalt Storage's degraded-node incident, and no data was lost.",
         quote="No write requests were affected, and no data was lost", computation=None,
         checklist_keywords=["write requests", "no data was lost"]),
    dict(base="cobalt_storage", category="supported",
         text="10 minutes elapsed between Cobalt Storage detecting the degraded node and draining it from the read path.",
         quote=None, computation={"operation": "duration_minutes",
             "operands": [{"value": "03:31", "quote": "identified through per-node latency percentile alerting at 03:31 UTC"},
                          {"value": "03:41", "quote": "drained from the read path at 03:41 UTC"}],
             "result": 10.0},
         checklist_keywords=["03:31", "03:41", "drained"]),
    dict(base="cobalt_storage", category="unsupported",
         text="Cobalt Storage's eu-central-1 incident triggered a full regional failover to a backup datacenter.",
         quote=None, computation=None, checklist_keywords=None),
    dict(base="cobalt_storage", category="unverifiable",
         text="The failing disk had been in service for over four years before it began degrading.",
         quote=None, computation=None, checklist_keywords=None),
    dict(base="cobalt_storage", category="wrong_attribution",
         text="Cobalt Storage's incident was caused by a database connection pool exhausted by a misconfigured batch job.",  # true fact, belongs to ridgeline_auth
         quote="a database connection pool was exhausted by a misconfigured batch job", computation=None, checklist_keywords=None),
    dict(base="cobalt_storage", category="incorrect_computation",
         text="Cobalt Storage's degraded-node incident lasted 61 minutes end to end.",
         quote=None, computation={"operation": "duration_minutes",
             "operands": [{"value": "02:50", "quote": "Starting at 02:50 UTC on 2026-07-29"},
                          {"value": "03:41", "quote": "drained from the read path at 03:41 UTC"}],
             "result": 61.0},  # correct answer is 51
         checklist_keywords=None),

    # --- meridian_cache -----------------------------------------------------
    dict(base="meridian_cache", category="supported",
         text="No customer data was exposed or corrupted during Meridian Cache's stale-content incident.",
         quote="No customer data was exposed or corrupted", computation=None,
         checklist_keywords=["customer data", "exposed or corrupted"]),
    dict(base="meridian_cache", category="supported",
         text="27 minutes elapsed from the start of Meridian Cache's incident to full cache freshness being restored.",
         quote=None, computation={"operation": "duration_minutes",
             "operands": [{"value": "03:14", "quote": "Starting at 03:14 UTC on 2026-05-11"},
                          {"value": "03:41", "quote": "Full cache freshness was restored by 03:41 UTC"}],
             "result": 27.0},
         checklist_keywords=["03:14", "03:41", "restored"]),
    dict(base="meridian_cache", category="unsupported",
         text="Meridian Cache's stale-content incident caused several customer purchase transactions to be double-charged.",
         quote=None, computation=None, checklist_keywords=None),
    dict(base="meridian_cache", category="unverifiable",
         text="The message broker involved in the partition rebalance was running a version several minor releases behind current.",
         quote=None, computation=None, checklist_keywords=None),
    dict(base="meridian_cache", category="wrong_attribution",
         text="Meridian Cache's consumer lag grew to approximately 340,000 unacknowledged messages at peak.",  # true fact, belongs to solace_queue
         quote="Consumer lag grew to approximately 340,000 unacknowledged messages at peak", computation=None, checklist_keywords=None),
    dict(base="meridian_cache", category="incorrect_computation",
         text="15 minutes elapsed between Meridian Cache's incident being detected and the broker partition being rebalanced.",
         quote=None, computation={"operation": "duration_minutes",
             "operands": [{"value": "03:22", "quote": "detected by an automated staleness monitor at 03:22 UTC"},
                          {"value": "03:27", "quote": "the broker partition was manually rebalanced at 03:27 UTC"}],
             "result": 15.0},  # correct answer is 5
         checklist_keywords=None),

    # --- ridgeline_auth -----------------------------------------------------
    dict(base="ridgeline_auth", category="supported",
         text="Users with still-valid access tokens were not logged out during Ridgeline Auth's incident.",
         quote="users with still-valid access tokens were not logged out", computation=None,
         checklist_keywords=["still-valid access tokens", "not logged out"]),
    dict(base="ridgeline_auth", category="supported",
         text="3 minutes elapsed between Ridgeline Auth's pool-saturation alert and the offending batch job being killed.",
         quote=None, computation={"operation": "duration_minutes",
             "operands": [{"value": "20:11", "quote": "identified via connection-pool saturation alerts at 20:11 UTC"},
                          {"value": "20:14", "quote": "killed at 20:14 UTC"}],
             "result": 3.0},
         checklist_keywords=["20:11", "20:14", "killed"]),
    dict(base="ridgeline_auth", category="unsupported",
         text="Ridgeline Auth's connection pool exhaustion exposed session tokens for affected users in server logs.",
         quote=None, computation=None, checklist_keywords=None),
    dict(base="ridgeline_auth", category="unverifiable",
         text="The misconfigured batch job had been deployed earlier that same day.",
         quote=None, computation=None, checklist_keywords=None),
    dict(base="ridgeline_auth", category="wrong_attribution",
         text="Ridgeline Auth's renewal failure was traced to an expired API token used by the certificate-renewal automation.",  # true fact, belongs to anchor_cdn
         quote="traced to an expired API token used by the certificate-renewal automation", computation=None, checklist_keywords=None),
    dict(base="ridgeline_auth", category="incorrect_computation",
         text="Ridgeline Auth's session-refresh incident lasted 22 minutes from onset to the batch job being killed.",
         quote=None, computation={"operation": "duration_minutes",
             "operands": [{"value": "20:02", "quote": "At 20:02 UTC on 2026-07-15"},
                          {"value": "20:14", "quote": "killed at 20:14 UTC"}],
             "result": 22.0},  # correct answer is 12
         checklist_keywords=None),

    # --- solace_queue -----------------------------------------------------
    dict(base="solace_queue", category="supported",
         text="No messages were dropped during Solace Queue's consumer-lag incident, and the retention window was never approached.",
         quote="No messages were dropped; the queue's retention window of 72 hours was never approached", computation=None,
         checklist_keywords=["no messages were dropped", "retention window"]),
    dict(base="solace_queue", category="supported",
         text="8 minutes elapsed between Solace Queue detecting the slow receiver and throttling the partner's webhook concurrency.",
         quote=None, computation={"operation": "duration_minutes",
             "operands": [{"value": "12:03", "quote": "identified the slow receiver via consumer-lag alerting at 12:03 UTC"},
                          {"value": "12:11", "quote": "throttled the specific partner's webhook concurrency at 12:11 UTC"}],
             "result": 8.0},
         checklist_keywords=["12:03", "12:11", "throttled"]),
    dict(base="solace_queue", category="unsupported",
         text="Solace Queue's order-events backlog caused permanent loss of approximately 12,000 messages.",
         quote=None, computation=None, checklist_keywords=None),
    dict(base="solace_queue", category="unverifiable",
         text="The partner integration responsible for the slow webhook receiver has since been suspended from the platform.",
         quote=None, computation=None, checklist_keywords=None),
    dict(base="solace_queue", category="wrong_attribution",
         text="Solace Queue's degraded disk was replaced during a maintenance window the same day.",  # true fact, belongs to cobalt_storage
         quote="the physical disk was replaced during a maintenance window the same day", computation=None, checklist_keywords=None),
    dict(base="solace_queue", category="incorrect_computation",
         text="Solace Queue's backlog took 57 minutes to drain from onset to full completion.",
         quote=None, computation={"operation": "duration_minutes",
             "operands": [{"value": "11:45", "quote": "Beginning at 11:45 UTC on 2026-07-03"},
                          {"value": "12:52", "quote": "Full backlog drain completed at 12:52 UTC"}],
             "result": 57.0},  # correct answer is 67
         checklist_keywords=None),
]

BAD_CATEGORIES = {"unsupported", "unverifiable", "wrong_attribution", "incorrect_computation"}


def build_ground_truth_claims() -> list[Claim]:
    claims = []
    for i, g in enumerate(GROUND_TRUTH):
        claims.append(Claim(
            text=g["text"], quote=g["quote"], source_path=_src(g["base"]),
            worker_id=f"ground_truth_{i}", computation=g["computation"],
        ))
    return claims


SINGLE_AGENT_SYSTEM_PROMPT = (
    "You extract findings from a batch of incident postmortems, given together "
    "below. Read ONLY the source text given to you. Respond with strict JSON: "
    '{"claims": [{"source": "<postmortem key>", "text": "...", "quote": <QUOTE_OR_NULL>, "computation": <COMPUTATION_OR_NULL>}]}. '
    "source: the exact key of the postmortem this claim is about, exactly as "
    "given in the '=== SOURCE: <key> ===' headers below.\n"
    "quote: copy a supporting span exactly from THAT postmortem's text, "
    "character for character, with no leading or trailing ellipsis and no "
    "text from outside that one span or from any other postmortem. If no "
    "single span supports the claim, set quote to null.\n"
    "computation: only set this if the claim's number is something YOU "
    "computed from two or more source values (a duration, sum, or "
    "difference) rather than a number stated outright in the source. "
    'Format: {"operation": "duration_minutes" | "sum" | "difference", '
    '"operands": [{"value": "...", "quote": "verbatim source span this '
    'operand came from"}, ...], "result": <the number your claim asserts>}. '
    "For duration_minutes, operands are [start_time, end_time]. If the "
    "claim's number appears directly in the source, leave computation null "
    "and use quote instead."
)


def run_single_agent(client: OllamaClient) -> tuple[list[Claim], int]:
    """One LLM call across all 5 postmortems at once -- no worker
    decomposition, no per-source context boundary. Returns (claims,
    dropped_count) where dropped_count is claims whose 'source' field didn't
    match a known postmortem key (logged, not silently discarded).
    """
    blocks = []
    for base in BASES:
        text = Path(_src(base)).read_text(encoding="utf-8")
        blocks.append(f"=== SOURCE: {base} ===\n{text}")
    prompt = "Extract the key findings from each of these incident postmortems.\n\n" + "\n\n".join(blocks)

    try:
        raw = client.generate(prompt, system=SINGLE_AGENT_SYSTEM_PROMPT, json_mode=True, label="single_agent")
    except OllamaUnavailable as exc:
        print(f"[move2] single_agent call failed: {exc}")
        return [], 0

    claims: list[Claim] = []
    dropped = 0
    try:
        parsed = json.loads(raw)
        for c in parsed.get("claims", []):
            source_key = c.get("source")
            if source_key not in BASES:
                dropped += 1
                continue
            claims.append(Claim(
                text=str(c.get("text", "")).strip(),
                quote=(c.get("quote") or None),
                source_path=_src(source_key),
                worker_id="single_agent",
                computation=(c.get("computation") or None),
            ))
    except (json.JSONDecodeError, AttributeError):
        pass
    return claims, dropped


def omission_rate(real_claims: list[Claim]) -> tuple[int, int, list[str]]:
    """Checklist coverage: for each source's 2 hand-authored 'supported'
    ground-truth facts, does ANY real claim's text+quote for that source
    contain the fact's distinguishing keywords? A heuristic (substring, not
    semantic) approximation, stated as such -- not a claim of perfect recall
    measurement.
    """
    by_source: dict[str, list[str]] = {}
    for c in real_claims:
        key = Path(c.source_path).stem.replace("_postmortem", "")
        by_source.setdefault(key, []).append(f"{c.text} {c.quote or ''}".lower())

    missed = []
    total = 0
    for g in GROUND_TRUTH:
        if g["category"] != "supported":
            continue
        total += 1
        haystacks = by_source.get(g["base"], [])
        covered = any(all(kw in h for kw in g["checklist_keywords"]) for h in haystacks)
        if not covered:
            missed.append(f"{g['base']}: {g['text']}")
    return total - len(missed), total, missed


def score_reported(condition: str, reported_ids: set[int], ground_truth_claims: list[Claim]) -> dict:
    by_category: dict[str, list[bool]] = {}
    for g, claim in zip(GROUND_TRUTH, ground_truth_claims):
        by_category.setdefault(g["category"], []).append(id(claim) in reported_ids)

    bad_total = sum(len(v) for k, v in by_category.items() if k in BAD_CATEGORIES)
    bad_leaked = sum(sum(v) for k, v in by_category.items() if k in BAD_CATEGORIES)
    good_total = len(by_category.get("supported", []))
    good_reported = sum(by_category.get("supported", []))

    per_category_leak = {
        k: f"{sum(v)}/{len(v)}" for k, v in by_category.items() if k in BAD_CATEGORIES
    }
    return dict(
        condition=condition,
        bad_leaked=bad_leaked, bad_total=bad_total,
        good_reported=good_reported, good_total=good_total,
        false_rejection=good_total - good_reported,
        per_category_leak=per_category_leak,
    )


def call_stats(client: OllamaClient) -> dict:
    calls = client.calls
    tokens = sum((c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in calls if c.success)
    duration = sum(c.duration_s or 0.0 for c in calls)
    return dict(n_calls=len(calls), tokens=tokens, duration_s=duration)


def main() -> int:
    swarm_client = OllamaClient()
    if not swarm_client.is_available():
        print(f"Ollama is not reachable at {swarm_client.host}. Start it, then re-run.")
        return 2

    # --- Run the swarm extraction ONCE; conditions A and B both read from
    # this same output, so verification is the only variable between them. --
    tasks = [
        WorkerTask(worker_id=base, instruction="Extract the key findings from this incident postmortem.",
                   source_path=_src(base))
        for base in BASES
    ]
    worker = OllamaSimWorker(client=swarm_client)
    orchestrator = Orchestrator(worker=worker, synth_client=swarm_client)
    print("[move2] Running swarm extraction (5 workers, 1 per source)...")
    orch_result = orchestrator.run(tasks)
    swarm_real_claims: list[Claim] = [c for out in orch_result.worker_outputs for c in out.claims]
    print(f"[move2] Swarm extracted {len(swarm_real_claims)} real claim(s).")

    ground_truth_claims = build_ground_truth_claims()

    # --- Condition A: groundswarm (swarm + verifier) ------------------------
    verify_client = OllamaClient()
    verifier = Verifier(judge_client=verify_client)
    print("[move2] Verifying condition A (groundswarm): real + ground-truth claims...")
    verifications_a = verifier.verify_all(swarm_real_claims + ground_truth_claims)
    reported_a = {id(v.claim) for v in verifications_a if v.verdict == "confirmed"}

    # --- Condition B: same swarm, verification disabled --------------------
    # Same extraction output as A, reused with zero re-sampling; every claim
    # reported unconditionally, Verifier never invoked -- zero extra calls.
    reported_b = {id(c) for c in swarm_real_claims + ground_truth_claims}

    # --- Condition C: single agent, no decomposition, no verification -----
    single_agent_client = OllamaClient()
    print("[move2] Running single_agent extraction (1 call across all 5 sources)...")
    single_agent_claims, dropped = run_single_agent(single_agent_client)
    print(f"[move2] single_agent extracted {len(single_agent_claims)} real claim(s)"
          f"{f', dropped {dropped} with an unrecognized source field' if dropped else ''}.")
    reported_c = {id(c) for c in single_agent_claims + ground_truth_claims}

    # --- Score all three conditions against the fixed ground truth ---------
    scores = [
        score_reported("groundswarm", reported_a, ground_truth_claims),
        score_reported("swarm_no_verification", reported_b, ground_truth_claims),
        score_reported("single_agent", reported_c, ground_truth_claims),
    ]

    # --- Omission: checklist coverage of each condition's REAL claims ------
    omit_a = omission_rate(swarm_real_claims)
    omit_b = omit_a  # same real claims as A
    omit_c = omission_rate(single_agent_claims)

    # --- Compute cost per condition -----------------------------------------
    swarm_cost = call_stats(swarm_client)          # extraction + synthesis, shared by A and B
    verify_cost = call_stats(verify_client)         # judge calls, A only
    single_agent_cost = call_stats(single_agent_client)

    cost_a = dict(n_calls=swarm_cost["n_calls"] + verify_cost["n_calls"],
                  tokens=swarm_cost["tokens"] + verify_cost["tokens"],
                  duration_s=swarm_cost["duration_s"] + verify_cost["duration_s"])
    cost_b = swarm_cost
    cost_c = single_agent_cost

    result = dict(
        scores=scores,
        omission=dict(groundswarm=omit_a, swarm_no_verification=omit_b, single_agent=omit_c),
        cost=dict(groundswarm=cost_a, swarm_no_verification=cost_b, single_agent=cost_c),
        real_claim_counts=dict(groundswarm=len(swarm_real_claims), swarm_no_verification=len(swarm_real_claims),
                                single_agent=len(single_agent_claims)),
        single_agent_dropped=dropped,
    )

    out_path = SCENARIO_DIR / "move2_benchmark_result.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[move2] Wrote result to {out_path}")

    # Full per-claim log for post-hoc reading (which specific claim leaked, what the
    # real extraction actually said) -- the summary JSON above deliberately doesn't
    # carry this, and eyeballing individual claims against source is how this project
    # catches heuristic/measurement artifacts before publishing a headline number
    # (see html/ops/rejection-analysis.html for the established precedent).
    verdict_by_id = {id(v.claim): v for v in verifications_a}

    def _claim_row(c: Claim, gt_category: str | None):
        v = verdict_by_id.get(id(c))
        return {
            "worker_id": c.worker_id, "source": Path(c.source_path).name, "text": c.text,
            "quote": c.quote, "computation": c.computation, "ground_truth_category": gt_category,
            "groundswarm_verdict": v.verdict if v else None,
            "groundswarm_method": v.method if v else None,
            "groundswarm_reason": v.reason if v else None,
        }

    claims_log = {
        "swarm_real_claims": [_claim_row(c, None) for c in swarm_real_claims],
        "single_agent_real_claims": [_claim_row(c, None) for c in single_agent_claims],
        "ground_truth_claims": [_claim_row(c, g["category"]) for g, c in zip(GROUND_TRUTH, ground_truth_claims)],
    }
    claims_log_path = SCENARIO_DIR / "move2_benchmark_claims.json"
    claims_log_path.write_text(json.dumps(claims_log, indent=2, default=str), encoding="utf-8")
    print(f"[move2] Wrote per-claim log to {claims_log_path}")

    for s in scores:
        print(f"[move2] {s['condition']}: bad claims leaked {s['bad_leaked']}/{s['bad_total']}, "
              f"false-rejected {s['false_rejection']}/{s['good_total']} good claims, "
              f"per-category leak {s['per_category_leak']}")
    print(f"[move2] omission (checklist coverage): groundswarm/B missed {len(omit_a[2])}/{omit_a[1]}, "
          f"single_agent missed {len(omit_c[2])}/{omit_c[1]}")
    print(f"[move2] compute: groundswarm {cost_a['tokens']:,} tok / {cost_a['duration_s']:.1f}s "
          f"({cost_a['n_calls']} calls); swarm_no_verification {cost_b['tokens']:,} tok / "
          f"{cost_b['duration_s']:.1f}s ({cost_b['n_calls']} calls); single_agent "
          f"{cost_c['tokens']:,} tok / {cost_c['duration_s']:.1f}s ({cost_c['n_calls']} calls)")

    all_calls = swarm_client.calls + verify_client.calls + single_agent_client.calls
    report_path = SCENARIO_DIR.parents[1] / "html" / "ops" / "move2-benchmark.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    all_bad_caught = scores[0]["bad_leaked"] == 0
    report_path.write_text(
        render_observability_html(
            scenario_name="Move 2 benchmark: groundswarm vs swarm_no_verification vs single_agent",
            calls=all_calls,
            pass_fail=all_bad_caught,
            notes=(
                f"30-claim fixed ground-truth corpus (5 sources x 6 categories) folded into all "
                f"3 conditions. groundswarm leaked {scores[0]['bad_leaked']}/{scores[0]['bad_total']} "
                f"bad claims, false-rejected {scores[0]['false_rejection']}/{scores[0]['good_total']} "
                f"good claims. See move2_benchmark_result.json for full per-category breakdown."
            ),
        ),
        encoding="utf-8",
    )
    print(f"[move2] Wrote observability report to {report_path}")

    return 0 if all_bad_caught else 1


if __name__ == "__main__":
    raise SystemExit(main())
