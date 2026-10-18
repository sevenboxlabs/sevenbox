"""
The same refund scenario as examples/refund_scenario.py (Phase 1, hand-built
Warrants) and examples/refund_scenario_runtime.py (Phase 2, registered Rules) --
now written the way an application developer actually would with the SDK.

Compare this file's line count and shape to the other two. That comparison IS
the Phase 3 deliverable, not just this file in isolation.

Run: PYTHONPATH=. python3 examples/refund_scenario_sdk.py
"""
from sevenbox_sdk import SevenboxSession, derive, irreversible, oracle, tool


@oracle
def infer_intent(ticket):
    return {"intent": "refund_request", "order_id": ticket["order_id"]}


@tool
def lookup_order(order_id):
    return {"status": "shipped"}


@derive
def eligibility(order):
    return {"eligible": order["status"] == "shipped"}


@irreversible
@derive
def refund_action(elig):
    if not elig["eligible"]:
        return None  # ordinary Python control flow -- no conditional-edge wiring
    return {"action": "refund_action(4471)"}


def main():
    with SevenboxSession() as session:
        ticket = session.axiom({"order_id": "4471"})
        intent = infer_intent(ticket)
        order = lookup_order(intent["order_id"])
        elig = eligibility(order)
        action = refund_action(elig)

        print("Frontier after execution:")
        for w in session.frontier().live:
            print(f"  - {w.content}")

        print(f"\n{len(session.store)} Warrants were created. Application code "
              f"never constructed one directly, never wrote a parent_ids list, "
              f"and never named a source_kind by hand except via the decorators "
              f"at the top of this file.")

        # Replay, with zero code changes to the scenario itself:
        print("\nReplaying from the captured store (no real calls happen):")

    with SevenboxSession(replay_from=session.store) as replay_session:
        ticket2 = replay_session.axiom({"order_id": "4471"})
        intent2 = infer_intent(ticket2)
        order2 = lookup_order(intent2["order_id"])
        elig2 = eligibility(order2)
        action2 = refund_action(elig2)

    print("Replayed frontier:")
    for w in replay_session.frontier().live:
        print(f"  - {w.content}")

    assert action2.value == action.value
    print("\nReplay reproduced the same result without invoking a real handler.")


if __name__ == "__main__":
    main()
