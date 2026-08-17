"""Renders an HTML observability report from a run's OllamaClient call log.

Kept separate from the html/ output location: this module only builds the
string, the caller (a scenario script) decides where to write it.
"""
from __future__ import annotations

import math
from html import escape

from .llm.ollama_client import CallLog


def _weighted_perplexity(calls: list[CallLog]) -> float | None:
    """Aggregate perplexity across calls, done correctly in log-space
    (exp(-total_logprob_sum / total_tokens)), not a naive mean of
    per-call perplexities, which would misweight short/long completions.
    """
    total_logprob = 0.0
    total_tokens = 0
    for c in calls:
        if c.perplexity is None or not c.completion_tokens:
            continue
        total_logprob += -math.log(c.perplexity) * c.completion_tokens
        total_tokens += c.completion_tokens
    if total_tokens == 0:
        return None
    return math.exp(-total_logprob / total_tokens)

_STYLE = """
  :root{--bg:#0f1115;--fg:#e6e9ef;--dim:#98a2b3;--card:#171a21;--line:#252a34;
        --good:#3fb950;--bad:#f85149;--warn:#d29922;--acc:#58a6ff}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:980px;margin:0 auto;padding:28px 20px 80px}
  h1{font-size:24px;margin:0 0 4px} h2{font-size:17px;margin:30px 0 8px;color:var(--acc)}
  .dim{color:var(--dim)} .small{font-size:13px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}
  table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}
  .scroll{overflow-x:auto}
  th,td{border-bottom:1px solid var(--line);padding:8px 9px;text-align:left;vertical-align:top}
  th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
  code{background:#0b0d11;padding:1px 5px;border-radius:4px;font-size:13px}
  a{color:var(--acc)}
  .back{font-size:13px;margin-bottom:18px;display:inline-block}
  .lead{font-size:15.5px;border-left:3px solid var(--acc);padding-left:14px;margin:16px 0}
  .tag{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11.5px;font-weight:700}
  .t-done{background:#123018;color:var(--good)} .t-fail{background:#3a1418;color:var(--bad)}
  .num{text-align:right;font-variant-numeric:tabular-nums}
  .stat{display:inline-block;min-width:150px;margin:6px 22px 6px 0}
  .stat b{display:block;font-size:20px} .stat span{color:var(--dim);font-size:12px}
"""


def _phase(label: str) -> str:
    return label.split(":", 1)[0] if ":" in label else label


def render_observability_html(scenario_name: str, calls: list[CallLog],
                                 pass_fail: bool | None, notes: str = "") -> str:
    total_prompt = sum(c.prompt_tokens or 0 for c in calls if c.success)
    total_completion = sum(c.completion_tokens or 0 for c in calls if c.success)
    total_duration = sum(c.duration_s or 0.0 for c in calls)
    failures = [c for c in calls if not c.success]
    overall_perplexity = _weighted_perplexity(calls)

    by_phase: dict[str, dict] = {}
    for c in calls:
        p = _phase(c.label)
        agg = by_phase.setdefault(p, {"calls": 0, "prompt": 0, "completion": 0, "duration": 0.0,
                                        "failed": 0, "members": []})
        agg["calls"] += 1
        agg["prompt"] += c.prompt_tokens or 0
        agg["completion"] += c.completion_tokens or 0
        agg["duration"] += c.duration_s or 0.0
        agg["failed"] += 0 if c.success else 1
        agg["members"].append(c)

    def _fmt_ppl(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "&mdash;"

    phase_rows = "\n".join(
        f"<tr><td>{escape(p)}</td><td class='num'>{int(a['calls'])}</td>"
        f"<td class='num'>{int(a['prompt']):,}</td><td class='num'>{int(a['completion']):,}</td>"
        f"<td class='num'>{int(a['prompt'] + a['completion']):,}</td>"
        f"<td class='num'>{a['duration']:.1f}s</td>"
        f"<td class='num'>{_fmt_ppl(_weighted_perplexity(a['members']))}</td>"
        f"<td class='num'>{int(a['failed']) or ''}</td></tr>"
        for p, a in sorted(by_phase.items())
    )

    def _status_cell(c: CallLog) -> str:
        if c.success:
            return '<span class="tag t-done">ok</span>'
        return f'<span class="tag t-fail">failed</span> {escape(c.error or "")}'

    call_rows = "\n".join(
        f"<tr><td>{i}</td><td><code>{escape(c.label)}</code></td><td>{escape(c.model)}</td>"
        f"<td class='num'>{c.prompt_tokens if c.success else '&mdash;'}</td>"
        f"<td class='num'>{c.completion_tokens if c.success else '&mdash;'}</td>"
        f"<td class='num'>{_fmt_ppl(c.perplexity)}</td>"
        f"<td class='num'>{(c.duration_s or 0):.2f}s</td>"
        f"<td>{_status_cell(c)}</td></tr>"
        for i, c in enumerate(calls, 1)
    )

    verdict_tag = (
        '<span class="tag t-done">PASS</span>' if pass_fail is True
        else '<span class="tag t-fail">FAIL</span>' if pass_fail is False
        else '<span class="tag">n/a</span>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Groundswarm &mdash; Observability: {escape(scenario_name)}</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="wrap">

<a class="back" href="../index.html">&larr; Groundswarm hub</a>
<h1>Observability &mdash; {escape(scenario_name)}</h1>
<div class="dim">Real per-call token counts and latency from the local Ollama server, captured by <code>OllamaClient</code> itself (not estimated). Scenario outcome: {verdict_tag}</div>

<div class="lead">
Every number on this page comes straight from Ollama's own OpenAI-compatible <code>/v1/chat/completions</code>
response (<code>usage.*</code> for tokens, <code>choices[0].logprobs</code> for perplexity), recorded on
<code>OllamaClient.calls</code> at the moment each request returns. This is local inference &mdash; no
Claude API tokens are spent running this scenario; this page exists so token/latency/confidence cost of
the swarm itself is visible, not guessed at. Perplexity is <code>exp(-mean(token logprob))</code> over
each call's generated tokens &mdash; lower means the model was more confident in the exact tokens it
produced; it says nothing about whether those tokens are factually correct, that's what the Verifier is
for.
</div>

<h2>Totals</h2>
<div class="card">
<div class="stat"><b>{len(calls)}</b><span>LLM calls</span></div>
<div class="stat"><b>{total_prompt:,}</b><span>prompt tokens</span></div>
<div class="stat"><b>{total_completion:,}</b><span>completion tokens</span></div>
<div class="stat"><b>{total_prompt + total_completion:,}</b><span>total tokens</span></div>
<div class="stat"><b>{total_duration:.1f}s</b><span>wall time in Ollama calls</span></div>
<div class="stat"><b>{_fmt_ppl(overall_perplexity)}</b><span>weighted avg perplexity</span></div>
<div class="stat"><b>{len(failures)}</b><span>failed calls</span></div>
</div>

<h2>By phase</h2>
<div class="scroll"><table><thead>
<tr><th>Phase</th><th>Calls</th><th>Prompt tok</th><th>Completion tok</th><th>Total tok</th><th>Duration</th><th>Perplexity</th><th>Failed</th></tr>
</thead><tbody>
{phase_rows}
</tbody></table></div>

<h2>Every call, in order</h2>
<div class="scroll"><table><thead>
<tr><th>#</th><th>Label</th><th>Model</th><th>Prompt tok</th><th>Completion tok</th><th>Perplexity</th><th>Duration</th><th>Status</th></tr>
</thead><tbody>
{call_rows}
</tbody></table></div>

{f'<h2>Notes</h2><div class="card">{notes}</div>' if notes else ''}

</div>
</body>
</html>
"""
