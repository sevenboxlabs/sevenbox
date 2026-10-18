"""
sevenbox_demo.security_incident_scenario -- a third scenario, in a domain
distinct from refund/kyc, wired into the same server/visualization layer.
Same public-API-only posture as the other two: only sevenbox_sdk and
warrant_core's public surface, nothing private, nothing special-cased for
the Operations Center.

Exists specifically to exercise a fact the Operations Center had never
observed before this scenario: a genuine, unresolved frontier conflict --
two independent, live correctors of the same target, neither superseding
the other. `compute_frontier()` has modeled this since Phase 1
(warrant_core.frontier.Conflict); nothing in sevenbox_demo had ever produced
one until now. An on-call engineer and a threat-intel feed disagree about
the same severity assessment, and Sevenbox does not arbitrate between them --
see frontier.py's own design note on why that's a runtime-layer decision,
not warrant_core's to make.
"""
from __future__ import annotations

from typing import Dict

from sevenbox_sdk import SevenboxSession, derive, irreversible, oracle, tool
from warrant_core import (
    EffectRecord,
    EpistemicClass,
    ReversibilityClass,
    SourceKind,
    WarrantStore,
    supersede,
)

_call_counts: Dict[str, int] = {"triage_alert": 0, "query_telemetry": 0}


@oracle
def triage_alert(raw_alert):
    _call_counts["triage_alert"] += 1
    return {"category": "anomalous_network_activity", "confidence": "medium"}


@tool
def query_telemetry(triage):
    _call_counts["query_telemetry"] += 1
    return {"host": "web-03", "outbound_bytes": 4_800_000_000}


@derive
def assess_severity(telemetry):
    return {
        "severity": "high",
        "reason": "outbound transfer volume exceeds baseline by 40x",
    }


@irreversible
@derive
def isolate_host(severity):
    if severity["severity"] != "high":
        return None
    return {"action": "network_isolate", "host": "web-03"}


def _run(session: SevenboxSession) -> None:
    raw_alert = session.axiom({"host": "web-03", "signal": "outbound_spike"})
    triage = triage_alert(raw_alert)
    telemetry = query_telemetry(triage)
    severity = assess_severity(telemetry)
    isolate_host(severity)

    # Two independent, competing corrections to the same severity
    # assessment -- an on-call engineer and a threat-intel feed disagree,
    # and neither supersedes the other. Both correctors omit source_name
    # (same pattern as refund/kyc's own out-of-band corrections), so both
    # are degraded-replay in addition to being an unresolved conflict.
    supersede(
        session.store, target_id=severity.warrant_id,
        content={"claim": "false positive -- scheduled nightly backup job, not exfiltration"},
        source_kind=SourceKind.HUMAN, epistemic=EpistemicClass.ASSUMED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(
            request="on-call review", response="assessed as false positive"
        ),
    )
    supersede(
        session.store, target_id=severity.warrant_id,
        content={"claim": "confirmed by threat-intel feed: matches known exfiltration signature"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(
            request="threat-intel lookup", response="signature match: simulated-APT-041"
        ),
    )


def run_live() -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["triage_alert"] = 0
    _call_counts["query_telemetry"] = 0
    session = SevenboxSession()
    with session:
        _run(session)
    return session, dict(_call_counts)


def run_replay(reference_store: WarrantStore) -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["triage_alert"] = 0
    _call_counts["query_telemetry"] = 0
    session = SevenboxSession(replay_from=reference_store)
    with session:
        _run(session)
    return session, dict(_call_counts)
