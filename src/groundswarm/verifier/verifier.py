from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

from ..llm.ollama_client import OllamaClient, OllamaUnavailable
from ..workers.base import Claim
from .checks import quote_round_trips

JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checker. You will be given a SOURCE text and a "
    "CLAIM. Decide only whether the source text actually supports the "
    "claim. Respond with strict JSON: "
    '{"supported": true or false, "reason": "one sentence"}. '
    "If you are not confident, respond supported: false. Do not guess."
)


@dataclass
class VerificationResult:
    claim: Claim
    verdict: str          # "confirmed" | "rejected" | "unverified"
    method: str            # "deterministic" | "llm_judge" | "fail_closed"
    reason: str


class Verifier:
    """Runs independently of whatever produced the claims. Deterministic
    checks run first and are trusted; the LLM judge is a fallback, and if
    it can't be reached or returns something unparseable, the claim is
    rejected rather than silently passed. Fails closed, not open.
    """

    def __init__(self, judge_client: OllamaClient | None = None):
        self.judge_client = judge_client or OllamaClient()

    def verify_claim(self, claim: Claim) -> VerificationResult:
        deterministic = quote_round_trips(claim)
        if deterministic is not None:
            verdict = "confirmed" if deterministic.passed else "rejected"
            return VerificationResult(claim, verdict, "deterministic", deterministic.reason)
        return self._judge(claim)

    def _judge(self, claim: Claim) -> VerificationResult:
        try:
            source_text = Path(claim.source_path).read_text(encoding="utf-8")
        except OSError as exc:
            return VerificationResult(claim, "unverified", "fail_closed", f"could not read source: {exc}")

        prompt = f"SOURCE:\n{source_text}\n\nCLAIM:\n{claim.text}"
        try:
            raw = self.judge_client.generate(prompt, system=JUDGE_SYSTEM_PROMPT, json_mode=True)
        except OllamaUnavailable as exc:
            return VerificationResult(claim, "unverified", "fail_closed", f"judge unreachable: {exc}")

        try:
            parsed = json.loads(raw)
            supported = parsed["supported"]
            reason = str(parsed.get("reason", ""))
        except (json.JSONDecodeError, KeyError, TypeError):
            return VerificationResult(claim, "unverified", "fail_closed", f"judge response unparseable: {raw!r}")

        if supported is True:
            return VerificationResult(claim, "confirmed", "llm_judge", reason)
        return VerificationResult(claim, "rejected", "llm_judge", reason)

    def verify_all(self, claims: list[Claim]) -> list[VerificationResult]:
        return [self.verify_claim(c) for c in claims]
