"""
sevenbox_demo.customer_support_scenario -- a sixth scenario, in a domain
distinct from every scenario before it, wired into the same server/
visualization layer. Same public-API-only posture: only sevenbox_sdk and
warrant_core's public surface, nothing private, nothing special-cased.

A support ticket gets classified, the customer's account looked up, a
resolution email sent (irreversible -- an email, once sent, can't be
unsent), and later a correction: the account tier lookup was wrong,
triggering a compensation obligation on the already-sent resolution.
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

_call_counts: Dict[str, int] = {"classify_ticket": 0, "lookup_account": 0}


@oracle
def classify_ticket(message):
    _call_counts["classify_ticket"] += 1
    return {"intent": "billing_dispute", "sentiment": "frustrated"}


@tool
def lookup_account(intent):
    _call_counts["lookup_account"] += 1
    return {"tier": "standard", "account_age_days": 214}


@derive
def determine_resolution(account):
    if account["tier"] == "premium":
        return {"resolution": "full_refund"}
    return {"resolution": "partial_credit"}


@irreversible
@tool
def send_resolution_email(resolution):
    return {"action": "email_sent", "template": resolution["resolution"]}


def _run(session: SevenboxSession) -> None:
    message = session.axiom({"text": "I was charged twice for my subscription"})
    intent = classify_ticket(message)
    account = lookup_account(intent)
    resolution = determine_resolution(account)
    send_resolution_email(resolution)

    # The account tier lookup turns out to have been stale -- the
    # customer had actually upgraded to premium the day before. Targets
    # an ancestor of the already-sent (irreversible) email, the same
    # transitive-correction shape as every other scenario here.
    supersede(
        session.store, target_id=account.warrant_id,
        content={"claim": "customer upgraded to premium tier prior to this ticket"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(
            request="billing system re-check", response="tier=premium as of yesterday"
        ),
    )


def run_live() -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["classify_ticket"] = 0
    _call_counts["lookup_account"] = 0
    session = SevenboxSession()
    with session:
        _run(session)
    return session, dict(_call_counts)


def run_replay(reference_store: WarrantStore) -> "tuple[SevenboxSession, Dict[str, int]]":
    _call_counts["classify_ticket"] = 0
    _call_counts["lookup_account"] = 0
    session = SevenboxSession(replay_from=reference_store)
    with session:
        _run(session)
    return session, dict(_call_counts)
