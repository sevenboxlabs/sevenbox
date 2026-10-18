"""
The refund scenario one more time -- this time as an actual LangGraph app
(built against the real `langgraph` package), wrapped with the Phase 4 adapter
instead of rewritten in Sevenbox's own vocabulary.

This is the concrete test of "install in an afternoon": compare the diff between
a plain LangGraph app and this file. It's two kinds of changes: wrap each
add_node call, wrap the invoke() call site. Nothing inside infer_intent,
lookup_order, eligibility, or refund_action changed at all.

Run: PYTHONPATH=. python3 examples/refund_scenario_langgraph.py
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from warrant_core import EpistemicClass, ReversibilityClass, SourceKind, find_transitive_obligations, supersede
from warrant_core import EffectRecord
from sevenbox_adapters.langgraph_adapter import sevenbox_node, sevenbox_run


class RefundState(TypedDict, total=False):
    ticket: dict
    intent: dict
    order: dict
    eligible: bool
    action: str


# --- these four functions are completely ordinary LangGraph node functions ---
# --- nothing about them changes to work with Sevenbox -----------------------------

def infer_intent(state):
    return {"intent": {"intent": "refund_request", "order_id": state["ticket"]["order_id"]}}


def lookup_order(state):
    return {"order": {"status": "shipped"}}


def eligibility(state):
    return {"eligible": state["order"]["status"] == "shipped"}


def refund_action(state):
    if not state.get("eligible"):
        return None
    return {"action": "refund_action(4471)"}


def build_graph():
    g = StateGraph(RefundState)
    # --- change 1: wrap each add_node call ---
    g.add_node("infer_intent", sevenbox_node(
        infer_intent, source_kind=SourceKind.ORACLE, epistemic=EpistemicClass.INFERRED,
    ))
    g.add_node("lookup_order", sevenbox_node(lookup_order, source_kind=SourceKind.TOOL))
    g.add_node("eligibility", sevenbox_node(eligibility, source_kind=SourceKind.DERIVED))
    g.add_node("refund_action", sevenbox_node(
        refund_action, source_kind=SourceKind.DERIVED, reversibility=ReversibilityClass.IRREVERSIBLE,
    ))
    g.add_edge(START, "infer_intent")
    g.add_edge("infer_intent", "lookup_order")
    g.add_edge("lookup_order", "eligibility")
    g.add_edge("eligibility", "refund_action")
    g.add_edge("refund_action", END)
    return g.compile()


def main():
    compiled = build_graph()

    # --- change 2: wrap the invoke() call site ---
    with sevenbox_run() as krun:
        result = compiled.invoke({"ticket": {"order_id": "4471"}})

    print("LangGraph's own result (unchanged):", result)

    print("\nWhat Sevenbox captured, that plain LangGraph would not give you:")
    for w in krun.session.frontier().live:
        print(f"  - [{w.source_kind.value:7s}] {w.content}")

    # New evidence arrives after the fact -- same pattern as every prior example.
    eligible_warrant = next(w for w in krun.session.store.all() if "eligible" in w.content)
    supersede(
        krun.session.store, target_id=eligible_warrant.id,
        content={"claim": "order #4471 was already refunded earlier"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="refund-history check", response="already refunded"),
    )

    obligations = find_transitive_obligations(krun.session.store)
    print(f"\nCompensation obligations found: {len(obligations)}")
    for o in obligations:
        print(f"  - {o.irreversible.content} needs compensation")

    # Replay: rerun the exact same graph, same invoke() call, zero real work done.
    print("\nReplaying the same graph from the captured store:")
    with sevenbox_run(replay_from=krun.session.store) as replay_run:
        replay_result = compiled.invoke({"ticket": {"order_id": "4471"}})

    assert replay_result == result
    print("Replay reproduced LangGraph's own result exactly, with no real node "
          "bodies executed for the effectful nodes.")


if __name__ == "__main__":
    main()
