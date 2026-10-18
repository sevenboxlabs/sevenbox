"""
sevenbox_demo.procurement_scenario -- a fourth scenario, in a domain distinct
from refund/kyc/security-incident, wired into the same server/
visualization layer. Same public-API-only posture: only sevenbox_sdk and
warrant_core's public surface.

Exercises two facts the Operations Center had never observed from a real
scenario before this one:

  - Genuine redaction. sevenbox_demo.kyc_scenario (unlike
    examples/kyc_scenario.py) never actually applies a RedactionPolicy --
    every "Redacted" count in the Operations Center's Decision Browser had
    read 0 for every System reachable from the UI. This scenario redacts
    real sensitive fields (bank account, routing number) the same way
    examples/kyc_scenario.py demonstrates, so the Decision Browser's
    "Redacted" preset has real data to show for the first time.
  - Branching. Every prior scenario's causal graph was a single chain.
    Here, one risk assessment forks into two independently-computed
    sibling evaluations (auto-approval eligibility, manual-escalation
    requirement), which the purchase-order decision then merges back
    into one action -- a real fork/merge shape in the Execution Graph,
    not a straight line.
"""
from __future__ import annotations

from typing import Dict

from sevenbox_sdk import SevenboxSession, derive, irreversible, oracle, tool
from warrant_core import (
    EffectRecord,
    EpistemicClass,
    RedactionAction,
    RedactionPolicy,
    ReversibilityClass,
    SourceKind,
    WarrantStore,
    supersede,
)

_call_counts: Dict[str, int] = {"extract_vendor_info": 0, "check_vendor_risk": 0}

_BANKING_POLICY = RedactionPolicy(
    fields={"bank_account": RedactionAction.HASH, "routing_number": RedactionAction.REDACT}
)


@oracle(redaction=_BANKING_POLICY)
def extract_vendor_info(request):
    _call_counts["extract_vendor_info"] += 1
    return {
        "vendor_name": request["vendor_name"],
        "bank_account": request["bank_account"],
        "routing_number": request["routing_number"],
    }


@tool(redaction=_BANKING_POLICY)
def check_vendor_risk(vendor_info):
    _call_counts["check_vendor_risk"] += 1
    return {"risk_score": "low", "sanctioned": False}


@derive
def assess_risk(risk_check):
    if risk_check["sanctioned"] or risk_check["risk_score"] != "low":
        return {"risk": "high"}
    return {"risk": "low"}


# The fork: both branches are always computed and always produce a real
# Warrant -- the point is that the system considered both paths, not that
# only the taken one exists. issue_purchase_order() below is the merge.
@derive
def evaluate_auto_approval(risk_assessment):
    return {"path": "auto_approve", "eligible": risk_assessment["risk"] == "low"}


@derive
def evaluate_manual_escalation(risk_assessment):
    return {"path": "manual_escalation", "required": risk_assessment["risk"] != "low"}


@irreversible
@derive
def issue_purchase_order(auto_approval, escalation):
    if not auto_approval["eligible"]:
        return None
    return {"action": "po_issued", "po_number": "PO-88213"}


def _run(session: SevenboxSession) -> None:
    request = session.axiom({
        "vendor_name": "Northwind Supplies",
        "bank_account": "000123456789",
        "routing_number": "021000021",
    })
    vendor_info = extract_vendor_info(request)
    risk_check = check_vendor_risk(vendor_info)
    risk_assessment = assess_risk(risk_check)

    auto_approval = evaluate_auto_approval(risk_assessment)
    escalation = evaluate_manual_escalation(risk_assessment)
    issue_purchase_order(auto_approval, escalation)

    # Correction arrives after the PO was already issued: a later audit
    # finds the vendor risk check itself was stale (fraud discovered).
    # This targets an ancestor of the irreversible action, not the action
    # itself -- the same transitive-compensation shape refund/kyc already
    # use, found here via find_transitive_obligations(), not
    # compensation_for() directly.
    supersede(
        session.store, target_id=risk_check.warrant_id,
        content={"claim": "vendor flagged for invoice fraud after audit review"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(
            request="fraud audit re-check", response="flagged: invoice fraud pattern"
        ),
    )


def run_live() -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["extract_vendor_info"] = 0
    _call_counts["check_vendor_risk"] = 0
    session = SevenboxSession()
    with session:
        _run(session)
    return session, dict(_call_counts)


def run_replay(reference_store: WarrantStore) -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["extract_vendor_info"] = 0
    _call_counts["check_vendor_risk"] = 0
    session = SevenboxSession(replay_from=reference_store)
    with session:
        _run(session)
    return session, dict(_call_counts)
