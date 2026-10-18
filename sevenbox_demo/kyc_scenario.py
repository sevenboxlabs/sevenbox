"""
sevenbox_demo.kyc_scenario -- a second scenario, in a different domain, wired
into the same server/visualization layer as sevenbox_demo.scenario. This exists
specifically to prove sevenbox_demo.visualize is genuinely decoupled from any one
scenario, not to duplicate examples/kyc_scenario.py's full walkthrough (which
also demonstrates redaction in detail -- see that file for the fuller story).

Same public-API-only posture as sevenbox_demo.scenario: only sevenbox_sdk and
warrant_core's public surface, nothing private.
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

_call_counts: Dict[str, int] = {"extract_identity": 0, "check_watchlist": 0}


@oracle
def extract_identity(application):
    _call_counts["extract_identity"] += 1
    return {"full_name": application["full_name"], "ssn": application["ssn"]}


@tool
def check_watchlist(identity):
    _call_counts["check_watchlist"] += 1
    return {"on_watchlist": False}


@tool
def verify_document(identity):
    return {"document_valid": True}


@derive
def assess_risk(watchlist_result, document_result):
    if watchlist_result["on_watchlist"] or not document_result["document_valid"]:
        return {"risk": "high"}
    return {"risk": "low"}


@irreversible
@derive
def approve_account(risk_assessment):
    if risk_assessment["risk"] != "low":
        return None
    return {"account_status": "approved", "account_id": "acc_9911"}


def _run(session: SevenboxSession) -> None:
    application = session.axiom({"full_name": "Jordan Rivera", "ssn": "123-45-6789"})
    identity = extract_identity(application)
    watchlist_result = check_watchlist(identity)
    document_result = verify_document(identity)
    risk_assessment = assess_risk(watchlist_result, document_result)
    approve_account(risk_assessment)

    supersede(
        session.store, target_id=watchlist_result.warrant_id,
        content={"claim": "applicant flagged after manual watchlist review"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="watchlist re-check", response="flagged"),
    )


def run_live() -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["extract_identity"] = 0
    _call_counts["check_watchlist"] = 0
    session = SevenboxSession()
    with session:
        _run(session)
    return session, dict(_call_counts)


def run_replay(reference_store: WarrantStore) -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["extract_identity"] = 0
    _call_counts["check_watchlist"] = 0
    session = SevenboxSession(replay_from=reference_store)
    with session:
        _run(session)
    return session, dict(_call_counts)
