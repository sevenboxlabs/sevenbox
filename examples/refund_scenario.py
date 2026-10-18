"""
End-to-end example: the refund scenario from the Fase 0.5 adversarial review, Task 9,
run against real warrant-core code instead of a hand-traced table.

Run: python3 examples/refund_scenario.py
"""
from warrant_core import (
    EffectRecord,
    EpistemicClass,
    ReversibilityClass,
    SourceKind,
    WarrantStore,
    compensation_for,
    compute_frontier,
    create_warrant,
    find_transitive_obligations,
    fold,
    supersede,
)


def main() -> None:
    store = WarrantStore()

    # 1. axiom: the agent exists, handling a ticket
    ticket = store.add(create_warrant(
        content={"claim": "agent handling support ticket #91"},
        source_kind=SourceKind.AXIOM,
        epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
    ))

    # 2. oracle: LLM infers customer intent
    intent = store.add(create_warrant(
        content={"claim": "customer wants a refund for order #4471"},
        source_kind=SourceKind.ORACLE,
        epistemic=EpistemicClass.INFERRED,
        reversibility=ReversibilityClass.PURE,
        parent_ids=(ticket.id,),
        effect_record=EffectRecord(
            request="What does the customer want?",
            response="A refund for order #4471.",
        ),
    ))

    # 3. tool: look up the order -- an effectful, non-oracle external call.
    # Note this is source_kind=TOOL, not ORACLE -- and it is STILL treated as
    # effectful (see types.py: PURE_SOURCE_KINDS only exempts DERIVED and AXIOM).
    # Tool calls are non-deterministic too, and are captured exactly like
    # oracle calls.
    order_status = store.add(create_warrant(
        content={"claim": "order #4471 status=shipped, refund-eligible"},
        source_kind=SourceKind.TOOL,
        epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        parent_ids=(intent.id,),
        effect_record=EffectRecord(request="GET /orders/4471", response="shipped"),
    ))

    # 4. derived: eligibility conclusion (pure function of order_status)
    eligible = store.add(create_warrant(
        content={"claim": "order #4471 is refund-eligible"},
        source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        parent_ids=(order_status.id,),
    ))

    # 5. derived: the irreversible action itself
    refund_action = store.add(create_warrant(
        content={"claim": "refund_action(4471)"},
        source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.IRREVERSIBLE,
        parent_ids=(eligible.id,),
    ))

    print("Frontier before any new evidence:")
    for w in compute_frontier(store).live:
        print(f"  - {w.content}")

    # 6. New evidence arrives AFTER the refund already ran: a second tool call
    # reveals it was already refunded earlier. This supersedes the eligibility
    # claim (#4), not the action itself -- the action is a *consequence*, and
    # gets handled via compensation, not direct supersession.
    supersede(
        store,
        target_id=eligible.id,
        content={"claim": "order #4471 was already refunded on 2026-07-20"},
        source_kind=SourceKind.TOOL,
        epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(
            request="GET /orders/4471/refund-history",
            response="already refunded 2026-07-20",
        ),
    )

    print("\nFrontier after the correction:")
    for w in compute_frontier(store).live:
        print(f"  - {w.content}")

    # 7. Does refund_action owe a compensation? compensation_for() only checks
    # *direct* superseders, and nothing directly supersedes refund_action -- the
    # correction targeted `eligible`, its parent. Direct lookup alone misses it:
    obligation_on_action = compensation_for(store, refund_action.id)
    print(f"\ncompensation_for(refund_action) directly: {obligation_on_action}")
    print("(None -- nothing directly supersedes refund_action.)")

    # find_transitive_obligations() walks the whole store for irreversible
    # warrants whose *ancestors* were superseded. This is the piece that used to
    # be manual-only; it's now a real function, not just documented as a gap.
    obligations = find_transitive_obligations(store)
    print(f"\nfind_transitive_obligations(store): {len(obligations)} found")
    for o in obligations:
        print(f"  - {o.irreversible.content} owes compensation")
        print(f"    (ancestor {o.superseded_ancestor_id[:8]}... was corrected by "
              f"{o.corrector_id[:8]}...)")

    def combine(acc, warrant):
        acc.append(warrant.content.get("claim"))
        return acc

    state = fold(compute_frontier(store), combine, [])
    print("\nFolded state (canonical id order):")
    for claim in state:
        print(f"  - {claim}")


if __name__ == "__main__":
    main()
