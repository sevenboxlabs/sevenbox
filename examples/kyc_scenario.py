"""
A second complete scenario, deliberately not e-commerce/refunds: KYC (Know
Your Customer) identity verification -- the domain where "approved something
irreversible based on a claim that later turned out false" is the worst day
of a compliance officer's year, not just an operational annoyance.

Demonstrates, through the same public sevenbox_sdk surface used throughout this
project (no new API introduced for this scenario):
  - an oracle call (parsing submitted identity documents)
  - two tool calls (a watchlist check and a document-authenticity check)
  - a derivation combining both tool results into a risk assessment
  - an irreversible action (account approval) gated on that assessment
  - redaction of genuinely sensitive fields (SSN, date of birth) in the
    captured EffectRecord -- content itself stays intact, as designed
  - a later correction (the watchlist provider updates its data) arriving
    after the account was already approved
  - the resulting compensation obligation, found the same way as every
    other scenario in this project: find_transitive_obligations()
  - replay: the exact same rule-driven chain reproduced without touching a
    real handler again

Run: PYTHONPATH=. python3 examples/kyc_scenario.py
"""
from sevenbox_sdk import SevenboxSession, derive, irreversible, oracle, tool
from warrant_core import (
    EffectRecord,
    EpistemicClass,
    RedactionAction,
    RedactionPolicy,
    ReversibilityClass,
    SourceKind,
    find_transitive_obligations,
    supersede,
)

_PII_POLICY = RedactionPolicy(
    fields={"ssn": RedactionAction.HASH, "date_of_birth": RedactionAction.REDACT}
)


@oracle(redaction=_PII_POLICY)
def extract_identity(application):
    """Simulates an LLM/OCR pipeline parsing a submitted ID document."""
    return {
        "full_name": application["full_name"],
        "ssn": application["ssn"],
        "date_of_birth": application["date_of_birth"],
    }


@tool(redaction=_PII_POLICY)
def check_watchlist(identity):
    """Simulates a sanctions/watchlist screening API call."""
    return {"on_watchlist": False, "screened_name": identity["full_name"]}


@tool
def verify_document(identity):
    """Simulates a third-party document-authenticity check. No PII in this
    particular request/response shape, so no policy needed here -- redaction
    is opt-in per call, not a blanket rule."""
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


def run_live():
    with SevenboxSession() as session:
        application = session.axiom({
            "full_name": "Jordan Rivera",
            "ssn": "123-45-6789",
            "date_of_birth": "1990-04-12",
        })
        identity = extract_identity(application)
        watchlist_result = check_watchlist(identity)
        document_result = verify_document(identity)
        risk_assessment = assess_risk(watchlist_result, document_result)
        account = approve_account(risk_assessment)

        print("Frontier after initial approval:")
        for w in session.frontier().live:
            print(f"  - [{w.source_kind.value:7s}] {w.content}")

        # Show that PII never leaked into the captured EffectRecord, even
        # though it's fully present in `content` (the typed claim used
        # downstream) -- this is the scope boundary stated in
        # warrant_core.redaction, made concrete here.
        identity_warrant = session.store.get(identity.warrant_id)
        print(f"\nCaptured EffectRecord for the identity-extraction call:")
        print(f"  request : {identity_warrant.effect_record.request}")
        print(f"  response: {identity_warrant.effect_record.response}")
        print(f"  (content, used downstream, is untouched: {identity_warrant.content})")

        # New evidence arrives after the account was already approved: the
        # watchlist provider issues a correction following manual review.
        supersede(
            session.store, target_id=watchlist_result.warrant_id,
            content={"claim": "applicant flagged after manual watchlist review"},
            source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
            reversibility=ReversibilityClass.PURE,
            effect_record=EffectRecord(
                request="watchlist re-check", response="flagged after manual review"
            ),
        )

        obligations = find_transitive_obligations(session.store)
        print(f"\nCompensation obligations after the correction: {len(obligations)}")
        for o in obligations:
            print(f"  - {o.irreversible.content} needs compensation")

    return session, account


def run_replay(reference_store):
    with SevenboxSession(replay_from=reference_store) as session:
        application = session.axiom({
            "full_name": "Jordan Rivera",
            "ssn": "123-45-6789",
            "date_of_birth": "1990-04-12",
        })
        identity = extract_identity(application)
        watchlist_result = check_watchlist(identity)
        document_result = verify_document(identity)
        risk_assessment = assess_risk(watchlist_result, document_result)
        account = approve_account(risk_assessment)
    return session, account


def main():
    live_session, live_account = run_live()
    replay_session, replay_account = run_replay(live_session.store)

    assert replay_account.value == live_account.value
    print("\nReplay reproduced the same approval decision without a real "
          "watchlist/document/model call being made again.")


if __name__ == "__main__":
    main()
