# orchestrator.py
#
# Triage agent — decides whether a query can be auto-resolved or must be
# escalated to a human. Deliberately rule-based rather than LLM-based: the
# decision needs to be fast, deterministic, and auditable, so it triages on
# retrieval confidence scores rather than asking the LLM to judge itself.
# Contains:
#   - OrchestratorDecision : should_resolve + escalation_reasons
#   - decide()             : applies triage rules to a RoutingResult

from __future__ import annotations

from dataclasses import dataclass, field

from routing_agent.config import RESOLUTION_CONFIDENCE_THRESHOLD
from routing_agent.router import RoutingResult


@dataclass
class OrchestratorDecision:
    should_resolve: bool
    escalation_reasons: list[str] = field(default_factory=list)


def decide(routing: RoutingResult) -> OrchestratorDecision:
    # Escalates whenever the knowledge base can't back a resolution with
    # sufficient confidence, rather than letting the LLM guess at an answer.
    reasons = []

    if not routing.context_docs:
        reasons.append("No relevant matches found in the knowledge base")
    else:
        top_score = routing.context_docs[0]["score"]
        if top_score < RESOLUTION_CONFIDENCE_THRESHOLD:
            reasons.append(
                f"Top retrieval confidence ({top_score:.2f}) is below the "
                f"resolution threshold ({RESOLUTION_CONFIDENCE_THRESHOLD:.2f})"
            )

    return OrchestratorDecision(should_resolve=not reasons, escalation_reasons=reasons)
