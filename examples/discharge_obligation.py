"""
discharge_obligation.py -- resolving compensation obligations, not just reading them.

The other refund examples (refund_scenario*.py) show Sevenbox finding a pending
obligation and printing it. They stop there: nothing in this repo demonstrated
actually resolving one, so a developer with three pending obligations had no named
API to reach for next. session.discharge() is that API.

Two scenarios here:
  1. A refund action whose eligibility claim gets corrected later. Sevenbox finds
     the (transitive) obligation this creates. We actually reverse the refund, then
     discharge the obligation as COMPENSATED, citing that reversal as evidence.
  2. A write action that later turns out to have been pointless -- its target was
     already gone by the time anyone got around to compensating it. We discharge
     that one as NOT_NEEDED, with a reason and no evidence, since no compensating
     action was actually taken.

Run: PYTHONPATH=. python3 examples/discharge_obligation.py
"""
from warrant_core import (
    DischargeOutcome,
    EffectRecord,
    EpistemicClass,
    ReversibilityClass,
    SourceKind,
    compensation_for,
    supersede,
)
from sevenbox_sdk import SevenboxSession, derive, irreversible, tool


@derive
def eligibility(order):
    return {"eligible": order["status"] == "shipped"}


@irreversible
@derive
def refund_action(elig):
    if not elig["eligible"]:
        return None
    return {"action": "refund_action(4471)"}


@tool
def reverse_refund(action):
    return {"action": "reversed_refund(4471)"}


@irreversible
@tool
def write_record(claim):
    return {"action": f"wrote record for {claim['target']}"}


@tool
def check_target_still_exists(write):
    return {"target_exists": False}


def scenario_1_refund_gets_compensated(session):
    print("Scenario 1: a refund whose eligibility claim gets corrected later\n")

    order = session.axiom({"status": "shipped"})
    elig = eligibility(order)
    action = refund_action(elig)

    # Time passes. A later check finds the order had already been refunded once --
    # the eligibility claim the refund depended on was wrong. Corrections go
    # through supersede() directly (same as every other Sevenbox example that
    # needs one): there's no @oracle/@tool-style decorator for "this replaces an
    # earlier claim," because a correction isn't produced by calling an external
    # system, it's an explicit relationship the caller is asserting between two
    # claims.
    supersede(
        session.store, target_id=elig.warrant_id,
        content={"eligible": False, "reason": "already refunded once"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="re-check order", response="already refunded"),
    )

    obligations = session.obligations()
    assert len(obligations) == 1
    obligation = obligations[0]
    print(f"Pending: {obligation.irreversible.content} needs compensation, "
          f"its ancestor was corrected.")

    # Actually reverse the refund -- a real compensating action, captured as its
    # own Warrant like anything else.
    reversal = reverse_refund(action)

    discharge = session.discharge(
        obligation, DischargeOutcome.COMPENSATED,
        reason="reversed the duplicate refund",
        evidence=reversal,
    )
    print(f"Discharged: {discharge.value}")
    print(f"Remaining obligations: {session.obligations()}\n")


def scenario_2_write_became_moot(session):
    print("Scenario 2: a write whose target was already gone\n")

    claim = session.axiom({"target": "record_88"})
    write = write_record(claim)

    # Something else corrects the write action directly -- a later check finds
    # its target doesn't exist anymore.
    check = check_target_still_exists(write)
    # Build the correction by hand here only because this example wants to show
    # the *direct* case (compensation_for) rather than the transitive one above --
    # an application would normally reach this through its own correction logic.
    supersede(
        session.store, target_id=write.warrant_id,
        content={"finding": "target record_88 no longer exists"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="check", response=check.value),
    )

    obligation = compensation_for(session.store, write.warrant_id)
    print(f"Pending: {obligation.target.content} needs compensation.")

    discharge = session.discharge(
        obligation, DischargeOutcome.NOT_NEEDED,
        reason="target record was already deleted before compensation was attempted",
    )
    print(f"Discharged: {discharge.value}")
    print(f"Still pending: {compensation_for(session.store, write.warrant_id)}\n")


def main():
    with SevenboxSession() as session:
        scenario_1_refund_gets_compensated(session)
        scenario_2_write_became_moot(session)


if __name__ == "__main__":
    main()
