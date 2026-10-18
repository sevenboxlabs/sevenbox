"""
The scenario requested for LangGraph testing specifically:

    Agent -> Tool -> Observation -> Decision -> Branch -> Tool -> Decision -> Final effect

Built as a real LangGraph app (StateGraph + add_conditional_edges, the
actual branching primitive, not a simulated fork) in a domain neither
existing example touches (credit/loan review), wrapped with the same
two changes as every other sevenbox_node example: wrap add_node, wrap the
invoke() call site.

This goes further than the shipped refund_scenario_langgraph.py in two
ways the linear 5-node refund chain never exercises:

  1. Real conditional routing (add_conditional_edges): depending on the
     first Decision node's output, execution takes ONE of two different
     paths (auto-verify vs. human-review) through the SAME compiled
     graph. Run twice below, once per path, to confirm Sevenbox correctly
     captures whichever path actually ran -- not a static assumption
     baked in at graph-build time.

  2. Genuine multi-parent Compose across the branch: the second Decision
     node explicitly declares an ADDITIONAL_PARENTS_KEY parent reaching
     back to the Observation node's Warrant, alongside the implicit
     frontier parent (the branch-specific Tool's Warrant) -- i.e. its
     claim is causally justified by BOTH the branch-specific evidence
     AND the earlier observation, exactly the "two independent
     evidentiary lines" pattern the paper's own KYC worked example
     (Sec. 13.2) describes, now exercised for real against LangGraph
     rather than only stated as Warrant-tuple notation in the paper.

Run: PYTHONPATH=. python3 examples/loan_review_scenario_langgraph_branching.py
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from warrant_core import (
    EpistemicClass, ReversibilityClass, SourceKind,
    find_transitive_obligations, supersede,
)
from warrant_core import EffectRecord
from sevenbox_adapters.langgraph_adapter import (
    ADDITIONAL_PARENTS_KEY, _current_graph_session, sevenbox_node, sevenbox_run,
)


class LoanState(TypedDict, total=False):
    application: dict
    intent: dict
    credit_raw: dict
    credit_observation: dict
    risk: dict
    verification: dict
    decision: dict
    effect: str


# --- ordinary LangGraph node functions; nothing about them is Sevenbox-aware ---
# --- except final_decision, which uses ADDITIONAL_PARENTS_KEY on purpose ---

def agent_intent(state):                                        # Agent
    return {"intent": {"intent": "loan_application", "applicant_id": state["application"]["applicant_id"]}}


def credit_check(state):                                        # Tool
    return {"credit_raw": {"raw_score": state["application"]["raw_score"]}}


def observe_credit(state):                                       # Observation
    raw = state["credit_raw"]["raw_score"]
    return {"credit_observation": {"score": raw, "tier": "good" if raw >= 700 else "poor"}}


def risk_decision(state):                                        # Decision
    return {"risk": {"level": "low" if state["credit_observation"]["tier"] == "good" else "high"}}


def route_on_risk(state) -> str:                                 # Branch
    return "auto_verify" if state["risk"]["level"] == "low" else "human_review"


def auto_verify(state):                                          # Tool (branch A)
    return {"verification": {"method": "automated", "result": "passed"}}


def human_review(state):                                         # Tool (branch B)
    return {"verification": {"method": "human", "result": "approved_with_conditions"}}


def final_decision(state):                                       # Decision, Compose across the branch
    # Reach back for the Observation Warrant explicitly -- this is the
    # intended use of ADDITIONAL_PARENTS_KEY: a node that KNOWS it is
    # joining two evidentiary lines (the branch-specific verification,
    # implicit via the frontier, and the original credit observation,
    # explicit here) declares the second one by id.
    session = _current_graph_session.get()
    observation_w = next(w for w in session.store.all() if "credit_observation" in w.content)
    approved = state["verification"]["result"] in ("passed", "approved_with_conditions")
    return {
        "decision": {"approved": approved, "based_on_tier": state["credit_observation"]["tier"]},
        ADDITIONAL_PARENTS_KEY: [observation_w.id],
    }


def final_effect(state):                                         # Final effect
    if not state["decision"]["approved"]:
        return {"effect": "loan_denied"}
    return {"effect": f"loan_disbursed({state['application']['applicant_id']})"}


def build_graph():
    g = StateGraph(LoanState)
    g.add_node("agent_intent", sevenbox_node(agent_intent, source_kind=SourceKind.ORACLE, epistemic=EpistemicClass.INFERRED))
    g.add_node("credit_check", sevenbox_node(credit_check, source_kind=SourceKind.TOOL))
    g.add_node("observe_credit", sevenbox_node(observe_credit, source_kind=SourceKind.DERIVED))
    g.add_node("risk_decision", sevenbox_node(risk_decision, source_kind=SourceKind.DERIVED))
    g.add_node("auto_verify", sevenbox_node(auto_verify, source_kind=SourceKind.TOOL))
    g.add_node("human_review", sevenbox_node(human_review, source_kind=SourceKind.TOOL))
    g.add_node("final_decision", sevenbox_node(final_decision, source_kind=SourceKind.DERIVED))
    g.add_node("final_effect", sevenbox_node(
        final_effect, source_kind=SourceKind.DERIVED, reversibility=ReversibilityClass.IRREVERSIBLE,
    ))

    g.add_edge(START, "agent_intent")
    g.add_edge("agent_intent", "credit_check")
    g.add_edge("credit_check", "observe_credit")
    g.add_edge("observe_credit", "risk_decision")
    g.add_conditional_edges("risk_decision", route_on_risk, {
        "auto_verify": "auto_verify", "human_review": "human_review",
    })
    g.add_edge("auto_verify", "final_decision")
    g.add_edge("human_review", "final_decision")
    g.add_edge("final_decision", "final_effect")
    g.add_edge("final_effect", END)
    return g.compile()


def run_and_report(compiled, applicant_id: str, raw_score: int, label: str):
    print(f"\n{'=' * 60}\n{label}  (applicant={applicant_id}, raw_score={raw_score})\n{'=' * 60}")

    with sevenbox_run() as krun:
        result = compiled.invoke({"application": {"applicant_id": applicant_id, "raw_score": raw_score}})

    print("LangGraph's own result:", result)
    print(f"\nWarrant graph ({len(krun.session.store)} warrants):")
    id_prefix = {}
    for w in krun.session.store.all():
        id_prefix[w.id] = w.id[:8]
    for w in krun.session.store.all():
        parents = [id_prefix[p] for p in w.parent_ids] or ["(none)"]
        print(f"  id={w.id[:8]}  [{w.source_kind.value:7s}]  parents={parents}  content={w.content}")

    # --- verify the branch actually taken is the only one reflected ---
    fired = {w.effect_record.source_name for w in krun.session.store.all() if w.effect_record}
    assert ("auto_verify" in fired) != ("human_review" in fired), \
        "exactly one branch tool should have fired, never both, never neither"
    which = "auto_verify" if "auto_verify" in fired else "human_review"
    print(f"\nBranch actually taken: {which}")

    # --- verify genuine Compose: final_decision has TWO parents ---
    decision_w = next(w for w in krun.session.store.all() if "decision" in w.content and "approved" in w.content.get("decision", {}))
    print(f"final_decision parent count: {len(decision_w.parent_ids)} (expect 2: branch tool + observation)")
    assert len(decision_w.parent_ids) == 2

    # --- transitive compensation: correct the ORIGINAL credit check, ---
    # --- several hops upstream of the irreversible final_effect, and ---
    # --- confirm the obligation is still found across the branch     ---
    credit_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL and "credit_raw" in w.content)
    supersede(
        krun.session.store, target_id=credit_w.id,
        content={"raw_score": None, "correction": "bureau reports this score was for a different applicant"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="bureau re-check", response="score mismatch"),
    )
    obligations = find_transitive_obligations(krun.session.store)
    print(f"Compensation obligations after upstream correction: {len(obligations)}")
    for o in obligations:
        print(f"  - {o.irreversible.content} needs compensation "
              f"(traced back through {which} branch, {len(decision_w.parent_ids)}-parent Compose node)")
    assert len(obligations) == 1

    # --- replay: same path, zero real node execution ---
    with sevenbox_run(replay_from=krun.session.store) as replay_run:
        replay_result = compiled.invoke({"application": {"applicant_id": applicant_id, "raw_score": raw_score}})
    assert replay_result == result
    print("Replay reproduced the exact same result, with no real node bodies executed.")

    return krun


def main():
    compiled = build_graph()
    run_and_report(compiled, "APP-1001", 750, "PATH A: good credit -> auto_verify branch")
    run_and_report(compiled, "APP-2002", 580, "PATH B: poor credit -> human_review branch")
    print("\nBoth branches of the SAME compiled graph correctly captured as "
          "distinct, correct Warrant graphs -- no cross-contamination between runs.")


if __name__ == "__main__":
    main()
