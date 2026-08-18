# Groundswarm

A swarm that shows its work — or shuts up.

**Technical invariant:** Groundswarm will not present a factual claim as
verified unless it can attach independently revalidated evidence or a
deterministic derivation. (Not "proof" — an LLM judge's agreement is not
proof, and claiming otherwise would be exactly the kind of unfalsifiable
marketing this project exists to be the opposite of.)

## The problem

Multi-agent AI frameworks (AutoGen, LangGraph, CrewAI, and others) solve
orchestration: how to spawn agents, hand off tasks, and coordinate work.
None of them solve trust. A swarm of 100 agents doesn't make hallucination
less likely, it makes it faster and harder to audit, because errors compound
across every hop before a human ever sees the output.

OpenClaw solves distribution: a personal assistant that already lives in
WhatsApp, Slack, Discord, Telegram, and iMessage through its Gateway and
channel connectors. But it's built for one operator, one assistant, and it
has no formal grounding or verification layer, only basic anti-loop
heuristics.

Groundswarm is the missing piece between the two: a swarm of OpenClaw-based
workers, coordinated by an orchestration layer, whose combined output is
independently re-checked against primary sources before anything is
reported as fact.

## Measured, not implied

Run twice, live, against a real model — not simulated: with verification on,
Groundswarm caught **39 of 40** injected bad claims (unsupported,
unverifiable, wrong-source, wrong arithmetic) at **zero cost** in false
rejections of true claims. The identical swarm with verification switched off,
and a single-agent baseline with no decomposition at all, both leaked
**40 of 40**. The gap is attributable entirely to the verification step:
condition B reuses the exact same extraction output as condition A, so
verification is the only variable between them.

This is a real, runnable, Apache-2.0 benchmark, not a cited internal number —
see [`benchmark/`](./benchmark/) to run it against your own framework, and
[`html/ops/move2-results.html`](./html/ops/move2-results.html) for the full
result including what it doesn't yet establish.

## How it works

**Workers.** Each worker is an OpenClaw instance, configured with a bounded
scope, a specific skill set, and (optionally) its own channel presence. This
is the distribution and execution layer, reused rather than reinvented.

**Orchestrator.** Decomposes a task into bounded per-worker scopes, tracks
dependencies between workers (a synthesis step cannot run until the workers
it depends on have reported), and reconciles worker output into a single
combined result. Downstream work depends on upstream results; nothing runs
on stale or partial state.

**Verifier.** The trust layer, and the actual differentiator. Runs
independently of the swarm that produced the output:

- Re-fetches primary sources live at check time, never trusts a stored
  answer key or the swarm's own memory of what a source said
- Rejects padding: shape without substance (headings, sliced quotes,
  vocabulary-free paraphrase) earns no credit
- Rejects invented quotes: every claimed quote must round-trip into the
  live-refetched source text under normalization
- Rejects fabricated classifications: structured fields are checked against
  their own evidence, not just their own internal consistency
- Fails closed: if the verifier itself can't independently confirm an
  output, it says so, it does not silently pass a low score off as a
  confirmed result

This pattern (deterministic grounding checks plus a fail-closed independent
judge) was proven out end to end in a working prototype before this project
started: a 7-instrument, 63-claim research task, graded by a from-scratch
verifier with 41 independent checks, tested against both real and
deliberately empty submissions to confirm it can't be gamed by an empty or
padded answer.

**Memory.** The swarm's shared knowledge store, and the second place the
verifier's trust primitive gets applied, this time to the swarm's own
knowledge instead of its task output:

- Scoped by validity, not by origin: every entry is tagged by blast radius
  (org-wide, team, project, single-run), not by which worker wrote it. This
  is what stops one worker's local, run-specific conclusion from silently
  contaminating another worker's context.
- Structurally legible: scope, TTL, confidence, and provenance are schema
  fields a worker can read cold, not conventions it has to have been
  onboarded onto, because in a swarm most workers never will be.
- Gated by a deterministic validator, not by trust in the last writer's
  judgment: structural invariants are checked before a write merges into
  the canonical store; org-wide or high-stakes writes require a review
  step, everything else can auto-merge.
- Tiered, not force-fed: a small always-injected global tier plus on-demand
  retrieval for project/team tier, so shared-context token cost is a
  deliberate budget line every worker pays on every call, not an
  afterthought.
- Staleness is explicit: TTL and supersedes/superseded-by links, since
  facts (pricing, org structure, policy) actually expire, they don't just
  quietly decay in confidence.

Unlike the verifier, this design isn't validated by a graded benchmark. It's
generalized from operating a real single-user, single-store instance of the
same pattern across many long-lived projects, where a hand-curated index,
folder-path-as-scope, and a lint script run by hand were good enough at that
scale. Each of those shortcuts is a specifically named thing that breaks
once writes come from many concurrent workers instead of one operator, and
Memory is designed to the version that has to hold up once that's true, not
the version that only had to work for one person.

## Status

Early. This repository currently holds the architecture and licensing
decisions; implementation is starting from here.

**First proof case (decided, ADR-005):** vendor infrastructure incident
monitoring, the same domain the grounding pattern was first proven on: a
swarm of workers each reads a vendor's public incident postmortems, extracts
and classifies findings, and the verifier confirms every claim against the
live source before it reaches a risk briefing a real team could act on. This
is also the domain every measured result in this repo (Move 2, the N-worker
sweep, the OpenClaw feedback loop, the CLI harness calibration) was run
against.

## License

Groundswarm's own code (the orchestrator and the verifier) is licensed
under **Apache License 2.0**, see [LICENSE](LICENSE). Free to use, modify,
and self-host, for individuals and organizations of any size.

A separate **Groundswarm Enterprise** edition is planned for teams that need
SSO, compliance/audit reporting, managed hosting, or support SLAs, under a
commercial license. The core engine will never be gated behind it.

OpenClaw is a separate project with its own license and its own governance
(the OpenClaw Foundation). Groundswarm builds on top of OpenClaw as a
worker runtime; it does not modify or relicense OpenClaw itself. See
[github.com/openclaw/openclaw](https://github.com/openclaw/openclaw) for
OpenClaw's own terms.
