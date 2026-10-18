from typing import TypedDict

import pytest

langgraph = pytest.importorskip("langgraph")
from langgraph.graph import END, START, StateGraph  # noqa: E402

from warrant_core import EpistemicClass, ReversibilityClass, SourceKind, compute_frontier  # noqa: E402
from sevenbox_adapters.langgraph_adapter import sevenbox_node, sevenbox_run  # noqa: E402


class RefundState(TypedDict, total=False):
    ticket: dict
    intent: dict
    order: dict
    eligible: bool
    action: str


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
    g.add_node("infer_intent", sevenbox_node(infer_intent, source_kind=SourceKind.ORACLE, epistemic=EpistemicClass.INFERRED))
    g.add_node("lookup_order", sevenbox_node(lookup_order, source_kind=SourceKind.TOOL))
    g.add_node("eligibility", sevenbox_node(eligibility, source_kind=SourceKind.DERIVED))
    g.add_node("refund_action", sevenbox_node(
        refund_action, source_kind=SourceKind.DERIVED, reversibility=ReversibilityClass.IRREVERSIBLE
    ))
    g.add_edge(START, "infer_intent")
    g.add_edge("infer_intent", "lookup_order")
    g.add_edge("lookup_order", "eligibility")
    g.add_edge("eligibility", "refund_action")
    g.add_edge("refund_action", END)
    return g.compile()


def test_node_called_outside_sevenbox_run_raises_clear_error():
    compiled = build_graph()
    with pytest.raises(RuntimeError, match="outside a `with sevenbox_run"):
        compiled.invoke({"ticket": {"order_id": "4471"}})


def test_wrapped_graph_populates_a_warrant_store():
    compiled = build_graph()
    with sevenbox_run() as krun:
        result = compiled.invoke({"ticket": {"order_id": "4471"}})

    assert result["action"] == "refund_action(4471)"
    # axiom (initial state) + 4 nodes = 5 warrants
    assert len(krun.session.store) == 5

    live = compute_frontier(krun.session.store).live
    kinds = {w.source_kind for w in live}
    assert SourceKind.ORACLE in kinds
    assert SourceKind.TOOL in kinds
    assert SourceKind.DERIVED in kinds
    assert SourceKind.AXIOM in kinds


def test_live_then_replay_never_calls_real_node_bodies_again():
    call_log = []

    def tracked_infer_intent(state):
        call_log.append(("oracle", state["ticket"]["order_id"]))
        return infer_intent(state)

    def tracked_lookup_order(state):
        call_log.append(("tool", state["intent"]["order_id"]))
        return lookup_order(state)

    g = StateGraph(RefundState)
    g.add_node("infer_intent", sevenbox_node(tracked_infer_intent, source_kind=SourceKind.ORACLE))
    g.add_node("lookup_order", sevenbox_node(tracked_lookup_order, source_kind=SourceKind.TOOL))
    g.add_node("eligibility", sevenbox_node(eligibility, source_kind=SourceKind.DERIVED))
    g.add_node("refund_action", sevenbox_node(
        refund_action, source_kind=SourceKind.DERIVED, reversibility=ReversibilityClass.IRREVERSIBLE
    ))
    g.add_edge(START, "infer_intent")
    g.add_edge("infer_intent", "lookup_order")
    g.add_edge("lookup_order", "eligibility")
    g.add_edge("eligibility", "refund_action")
    g.add_edge("refund_action", END)
    compiled = g.compile()

    with sevenbox_run() as live_run:
        live_result = compiled.invoke({"ticket": {"order_id": "4471"}})

    assert len(call_log) == 2  # one oracle call, one tool call

    with sevenbox_run(replay_from=live_run.session.store) as replay_run:
        replay_result = compiled.invoke({"ticket": {"order_id": "4471"}})

    # no NEW real calls happened
    assert len(call_log) == 2
    assert replay_result == live_result

    live_ids = {w.id for w in compute_frontier(live_run.session.store).live}
    replay_ids = {w.id for w in compute_frontier(replay_run.session.store).live}
    assert live_ids == replay_ids


def test_node_returning_none_stops_the_chain_cleanly():
    """refund_action returns None when not eligible -- confirm the wrapper
    propagates that correctly through LangGraph's own state-merge semantics."""
    def not_shipped(state):
        return {"order": {"status": "pending"}}

    g = StateGraph(RefundState)
    g.add_node("infer_intent", sevenbox_node(infer_intent, source_kind=SourceKind.ORACLE))
    g.add_node("lookup_order", sevenbox_node(not_shipped, source_kind=SourceKind.TOOL))
    g.add_node("eligibility", sevenbox_node(eligibility, source_kind=SourceKind.DERIVED))
    g.add_node("refund_action", sevenbox_node(
        refund_action, source_kind=SourceKind.DERIVED, reversibility=ReversibilityClass.IRREVERSIBLE
    ))
    g.add_edge(START, "infer_intent")
    g.add_edge("infer_intent", "lookup_order")
    g.add_edge("lookup_order", "eligibility")
    g.add_edge("eligibility", "refund_action")
    g.add_edge("refund_action", END)
    compiled = g.compile()

    with sevenbox_run() as krun:
        result = compiled.invoke({"ticket": {"order_id": "4471"}})

    assert "action" not in result
    action_warrants = [w for w in krun.session.store.all() if "action" in w.content]
    assert action_warrants == []


def test_an_explicitly_passed_empty_store_is_not_silently_discarded():
    """Same regression as sevenbox_sdk's SevenboxSession -- GraphSession had the
    identical `store or WarrantStore()` truthiness bug."""
    from warrant_core import WarrantStore
    from sevenbox_adapters.langgraph_adapter import GraphSession

    my_store = WarrantStore()
    gs = GraphSession(store=my_store)
    assert gs.store is my_store
