"""
sevenbox_demo.document_review_scenario -- a seventh scenario, in a domain
distinct from every scenario before it, wired into the same server/
visualization layer. Same public-API-only posture: only sevenbox_sdk and
warrant_core's public surface, nothing private, nothing special-cased.

A contract gets its key clauses extracted, checked against a compliance
database, and approved for signature (irreversible -- a signed contract
can't be unsigned). Later, a correction: the compliance database result
was stale, a clause actually violates an updated regulation, triggering
a compensation obligation on the already-approved contract.
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

_call_counts: Dict[str, int] = {"extract_clauses": 0, "check_compliance": 0}


@oracle
def extract_clauses(document):
    _call_counts["extract_clauses"] += 1
    return {"clauses": ["termination_for_convenience", "data_residency_us"]}


@tool
def check_compliance(clauses):
    _call_counts["check_compliance"] += 1
    return {"compliant": True, "flagged_clauses": []}


@derive
def assess_review_outcome(compliance_result):
    if not compliance_result["compliant"]:
        return {"outcome": "rejected"}
    return {"outcome": "approved"}


@irreversible
@derive
def approve_for_signature(review_outcome):
    if review_outcome["outcome"] != "approved":
        return None
    return {"action": "signature_approved", "contract_id": "K-2291"}


def _run(session: SevenboxSession) -> None:
    document = session.axiom({"contract_id": "K-2291", "counterparty": "Meridian Logistics"})
    clauses = extract_clauses(document)
    compliance_result = check_compliance(clauses)
    review_outcome = assess_review_outcome(compliance_result)
    approve_for_signature(review_outcome)

    # The compliance check turns out to have run against a stale
    # regulation snapshot -- the data-residency clause actually violates
    # an update that took effect the week before. Targets an ancestor of
    # the already-approved (irreversible) signature decision.
    supersede(
        session.store, target_id=compliance_result.warrant_id,
        content={"claim": "data_residency_us clause violates updated cross-border transfer rule"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(
            request="compliance re-check against current regulation set",
            response="violation found: data_residency_us",
        ),
    )


def run_live() -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["extract_clauses"] = 0
    _call_counts["check_compliance"] = 0
    session = SevenboxSession()
    with session:
        _run(session)
    return session, dict(_call_counts)


def run_replay(reference_store: WarrantStore) -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["extract_clauses"] = 0
    _call_counts["check_compliance"] = 0
    session = SevenboxSession(replay_from=reference_store)
    with session:
        _run(session)
    return session, dict(_call_counts)
