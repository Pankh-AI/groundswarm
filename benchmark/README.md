# Move 2: "Verified Beats Plausible" — a benchmark for agent swarms

A fixed, hand-authored ground-truth corpus for measuring one specific thing:
**when a multi-agent system extracts claims from source documents, how many
false or unsupported claims survive to the point where they'd reach a user as
fact — and at what cost in correctly-supported claims wrongly withheld?**

This is not a benchmark of extraction quality in general. It's a benchmark of
the trust decision that happens (or doesn't happen) after extraction. Every
orchestration framework we checked (LangGraph, CrewAI, AutoGen, swarms,
camel-ai, OpenHands, and 12 verification/eval products alongside them —
see [`../html/research/market-survey.html`](../html/research/market-survey.html))
either has no answer to this question or answers it with a second LLM call
grading the first one's output, which is not independent verification.

## The corpus

[`ground_truth.json`](./ground_truth.json): 30 claims, 5 sources ×
6 categories, about the 5 incident postmortems in [`sources/`](./sources/):

| Category | Count | What it tests |
|---|---|---|
| `supported` | 10 (2/source) | A true claim, quote-grounded or independently derivable by arithmetic from the source. A well-behaved system should report all 10. |
| `unsupported` | 5 | A plausible-sounding claim with **no support anywhere** in that source. |
| `unverifiable` | 5 | A claim the source neither confirms nor denies. |
| `wrong_attribution` | 5 | A **real, true fact** — but quoted from a *different* postmortem than the one it's attributed to. Tests whether a system checks *which* source a claim actually came from, not just whether the words exist somewhere. |
| `incorrect_computation` | 5 | A plausible arithmetic result (a duration between two real, quoted timestamps) that is simply **wrong** — the correct answer is a different number. Tests whether a system re-derives numbers or trusts them at face value. |

Every quote is checked character-for-character against `sources/*.txt`;
every computation's correct answer was independently recomputed at authoring
time. Ground truth here is true by construction, not by a second model's
opinion — see the rationale in
[`../scenarios/incident_monitoring_swarm/move2_benchmark.py`](../scenarios/incident_monitoring_swarm/move2_benchmark.py)'s
module docstring for why an LLM-graded oracle was deliberately rejected.

## Running it against your own system

1. Feed your system the 5 files in `sources/`.
2. Inject the 30 claim **texts** from `ground_truth.json` into whatever your
   system treats as "claims requiring a trust decision," alongside its own
   extraction output.
3. For each of the 30 ids, record one boolean: did your system present that
   claim to an end user as established fact? Write it to a verdicts file:
   `{"0": true, "1": true, "2": false, ...}` (see
   [`example_verdicts_groundswarm.json`](./example_verdicts_groundswarm.json)
   and [`example_verdicts_no_verification.json`](./example_verdicts_no_verification.json)
   for two worked examples).
4. `python score.py ground_truth.json your_verdicts.json`

[`score.py`](./score.py) is stdlib-only and has zero dependency on
Groundswarm's own code — it only reads the two JSON files. A system that does
no verification at all scores this by construction (every verdict `true`):
that's exactly what the `example_verdicts_no_verification.json` baseline
shows, reproducing 20/20 bad claims leaked.

## Groundswarm's own measured result

Run twice live against a remote model (not simulated). Full writeup:
[`../html/ops/move2-results.html`](../html/ops/move2-results.html).

| Condition | Bad claims leaked | False rejections of good claims |
|---|---|---|
| **groundswarm** (swarm + independent verifier) | **39 / 40** caught (both runs combined) | **0 / 20** |
| swarm_no_verification (identical extraction, no verifier) | 0 / 40 caught (40/40 leaked) | 0 / 20 |
| single_agent (one call, no decomposition, no verifier) | 0 / 40 caught (40/40 leaked) | 0 / 20 |

The gap is attributable entirely to the verification step — condition B reuses
*exactly* the same extraction output as condition A with zero re-sampling, so
verification is the only variable between them. The one miss (an
`unverifiable` claim resolved through the non-deterministic LLM-judge
fallback, correctly rejected on an identical re-run) and two follow-on
findings this run surfaced — see the full writeup, which reports them rather
than folding them into the headline number.

## Honest limits of this benchmark

- **30 claims is small.** It's enough to make the verification-vs-no-verification
  gap unambiguous (0/40 vs. 39/40 is not sampling noise), not enough to
  precision-rank two systems that are both already good at this.
- **One domain** (infrastructure incident postmortems). Whether the same gap
  holds in other domains (legal, medical, financial documents) is untested.
- **`wrong_attribution` is checked by "is the quote verbatim in the *declared*
  source," not by a system genuinely reasoning about multi-source attribution**
  — see the caveat in `move2-results.html`'s "Methodological caveats" section.
- This corpus and its scoring protocol are Apache-2.0, same as the rest of
  this repository — reuse it, extend it, or score a different framework
  against it. If you do, we'd like to know: it's exactly the kind of
  independent check this project's own literature review says the field is
  missing.
