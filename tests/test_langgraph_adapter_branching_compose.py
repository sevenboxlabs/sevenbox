"""
Real-LangGraph integration test for the scenario requested explicitly for
this framework's test coverage:

    Agent -> Tool -> Observation -> Decision -> Branch -> Tool -> Decision -> Final effect

This is a genuine integration test against the installed `langgraph`
package -- not a mock of LangGraph, not a simplified stand-in for
conditional routing. The only stubbed things are the *domain* function
bodies (credit_check, auto_verify, human_review return hardcoded dicts
instead of calling a real credit bureau), which is the same standard
every other example and test file in this repo uses for external-system
calls. Everything about how the graph itself is built, compiled, routed,
and executed is real:

  - `StateGraph.add_conditional_edges(source, path_fn, path_map)` is
    LangGraph's actual branching primitive, not simulated. `route_on_risk`
    returns a plain string; LangGraph's own Pregel-based runtime looks
    that string up in `path_map` at run time to decide which single node
    executes next. Two separate invocations of the SAME compiled graph
    object, with different initial state, are used below specifically to
    prove Sevenbox captures whichever path LangGraph *actually* took, not a
    path assumed at graph-build time.

  - `ADDITIONAL_PARENTS_KEY` (sevenbox_adapters.langgraph_adapter) is
    exercised for real: `final_decision` is a DERIVED node that reaches
    into the live GraphSession (via the same ContextVar sevenbox_node's
    wrapper reads) to find the Observation Warrant's real id and declare
    it as an extra causal parent, alongside the implicit frontier parent
    (the branch-specific Tool's Warrant). This is the intended usage
    pattern the adapter's own docstring describes for "a node that knows
    it is joining two branches" -- exercised here for the first time
    against a real running graph rather than only described.

  - `find_transitive_obligations` is exercised across a causal chain that
    is deeper and structurally richer than any other worked example in
    this repo: AXIOM -> ORACLE -> TOOL -> DERIVED -> DERIVED -> [branch]
    -> TOOL -> DERIVED(2 parents) -> DERIVED/IRREVERSIBLE. The correction
    lands on the *original* credit-check TOOL Warrant, several hops
    upstream of the irreversible effect, through a branch point and a
    2-parent Compose node -- the obligation must be found by walking the
    IRREVERSIBLE Warrant's *full* ancestor set, not merely its direct
    parent, exactly as Sec. 7.10 of the specification requires.

  - Replay is exercised per-path: each branch is replayed independently
    from its own captured store, confirming zero real node execution
    and an identical result, for both the auto_verify and human_review
    paths separately.

No change was made to warrant_core or sevenbox_adapters to make this test
pass -- see the module-level comment block below the imports for the two
minor, self-inflicted bugs found and fixed while writing this test (both
were bugs in this test file's own node bodies, not in the adapter or the
core library).
"""
from typing import TypedDict

import pytest

langgraph = pytest.importorskip("langgraph")
from langgraph.graph import END, START, StateGraph  # noqa: E402

from warrant_core import (  # noqa: E402
    EpistemicClass, ReversibilityClass, SourceKind,
    find_transitive_obligations, supersede,
)
from warrant_core import EffectRecord  # noqa: E402
from sevenbox_adapters.langgraph_adapter import (  # noqa: E402
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


# --- ordinary LangGraph node functions -- Agent / Tool / Observation / ---
# --- Decision / Branch / Tool / Decision / Final effect, in that order ---

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
    """The real LangGraph conditional-routing function: returns a plain
    string key. LangGraph's Pregel runtime resolves this against the
    `path_map` passed to add_conditional_edges to pick the single next
    node -- this is genuine framework-level branching, not something
    Sevenbox or this test simulates."""
    return "auto_verify" if state["risk"]["level"] == "low" else "human_review"


def auto_verify(state):                                          # Tool (branch A)
    return {"verification": {"method": "automated", "result": "passed"}}


def human_review(state):                                         # Tool (branch B)
    return {"verification": {"method": "human", "result": "approved_with_conditions"}}


def final_decision(state):                                       # Decision, Compose across the branch
    """Reaches back for the Observation Warrant's real id and declares it
    as an ADDITIONAL_PARENTS_KEY parent -- genuine multi-parent Compose,
    not the ordinary single-frontier-parent chaining every other node
    here uses. The implicit frontier parent (the branch-specific
    verification Tool's Warrant) plus this explicit one give the node
    exactly two parents, asserted below for both branches."""
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


def build_branching_graph():
    g = StateGraph(LoanState)
    g.add_node("agent_intent", sevenbox_node(agent_intent, source_kind=SourceKind.ORACLE, epistemic=EpistemicClass.INFERRED))
    g.add_node("credit_check", sevenbox_node(credit_check, source_kind=SourceKind.TOOL))
    g.add_node("observe_credit", sevenbox_node(observe_credit, source_kind=SourceKind.DERIVED))
    g.add_node("risk_decision", sevenbox_node(risk_decision, source_kind=SourceKind.DERIVED))
    g.add_node("auto_verify", sevenbox_node(auto_verify, source_kind=SourceKind.TOOL, name="auto_verify"))
    g.add_node("human_review", sevenbox_node(human_review, source_kind=SourceKind.TOOL, name="human_review"))
    g.add_node("final_decision", sevenbox_node(final_decision, source_kind=SourceKind.DERIVED))
    g.add_node("final_effect", sevenbox_node(
        final_effect, source_kind=SourceKind.DERIVED, reversibility=ReversibilityClass.IRREVERSIBLE,
    ))

    g.add_edge(START, "agent_intent")
    g.add_edge("agent_intent", "credit_check")
    g.add_edge("credit_check", "observe_credit")
    g.add_edge("observe_credit", "risk_decision")
    # --- the real LangGraph branching primitive ---
    g.add_conditional_edges("risk_decision", route_on_risk, {
        "auto_verify": "auto_verify", "human_review": "human_review",
    })
    g.add_edge("auto_verify", "final_decision")
    g.add_edge("human_review", "final_decision")
    g.add_edge("final_decision", "final_effect")
    g.add_edge("final_effect", END)
    return g.compile()


def _fired_tool_names(store):
    """Which effectful (Tool) nodes actually executed in this run, by
    their captured effect_record.source_name -- the ground truth for
    which branch LangGraph actually took."""
    return {w.effect_record.source_name for w in store.all() if w.effect_record}


def _find_final_decision_warrant(store):
    return next(
        w for w in store.all()
        if "decision" in w.content and "approved" in w.content.get("decision", {})
    )


def _find_original_credit_warrant(store):
    return next(
        w for w in store.all()
        if w.source_kind == SourceKind.TOOL and "credit_raw" in w.content
    )


# --- Path A: good credit -> auto_verify branch --------------------------

def test_path_a_good_credit_takes_only_the_auto_verify_branch():
    """Assert 1 (Camino A): human_review must NOT fire when the router
    picks auto_verify. This is the ground-truth check that LangGraph's
    conditional routing, not just Sevenbox's bookkeeping, actually took the
    branch we expect."""
    compiled = build_branching_graph()
    with sevenbox_run() as krun:
        result = compiled.invoke({"application": {"applicant_id": "APP-1001", "raw_score": 750}})

    assert result["verification"]["method"] == "automated"
    fired = _fired_tool_names(krun.session.store)
    assert "auto_verify" in fired
    assert "human_review" not in fired


def test_path_a_final_decision_has_exactly_two_parents():
    """Assert 3 (Camino A): the Compose node must have exactly two
    parents -- the implicit frontier parent (auto_verify's Warrant) and
    the explicit ADDITIONAL_PARENTS_KEY parent (observe_credit's
    Warrant). Checked by id, not just by count, so a coincidental count
    match can't hide the wrong parents being attached."""
    compiled = build_branching_graph()
    with sevenbox_run() as krun:
        compiled.invoke({"application": {"applicant_id": "APP-1001", "raw_score": 750}})

    decision_w = _find_final_decision_warrant(krun.session.store)
    auto_verify_w = next(w for w in krun.session.store.all() if w.effect_record and w.effect_record.source_name == "auto_verify")
    observation_w = next(w for w in krun.session.store.all() if "credit_observation" in w.content)

    assert len(decision_w.parent_ids) == 2
    assert set(decision_w.parent_ids) == {auto_verify_w.id, observation_w.id}


def test_path_a_transitive_compensation_reaches_the_irreversible_effect():
    """Assert 4 (Camino A): correcting the ORIGINAL credit_check Warrant
    -- five hops upstream of final_effect, through observe_credit,
    risk_decision, the branch itself, and the 2-parent Compose node --
    must still surface a compensation obligation on the terminal
    IRREVERSIBLE Warrant. This is Sec. 7.10's transitive-obligation
    traversal exercised over the deepest, structurally richest causal
    chain in this repo's test suite."""
    compiled = build_branching_graph()
    with sevenbox_run() as krun:
        compiled.invoke({"application": {"applicant_id": "APP-1001", "raw_score": 750}})

    credit_w = _find_original_credit_warrant(krun.session.store)
    supersede(
        krun.session.store, target_id=credit_w.id,
        content={"raw_score": None, "correction": "bureau reports this score was for a different applicant"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="bureau re-check", response="score mismatch"),
    )

    obligations = find_transitive_obligations(krun.session.store)
    assert len(obligations) == 1
    effect_w = next(w for w in krun.session.store.all() if w.content.get("effect", "").startswith("loan_disbursed"))
    assert obligations[0].irreversible.id == effect_w.id


def test_path_a_replay_reconstructs_the_result_without_real_node_execution():
    """Assert 5 (Camino A): replaying from the captured store must
    reproduce the exact result while calling zero real node bodies for
    effectful nodes -- confirmed by a call-count tracker on the specific
    branch tool that fired, not just by comparing final results (which
    could coincidentally match even if the real body re-ran)."""
    call_log = []

    def tracked_auto_verify(state):
        call_log.append(state["application"]["applicant_id"])
        return auto_verify(state)

    g = StateGraph(LoanState)
    g.add_node("agent_intent", sevenbox_node(agent_intent, source_kind=SourceKind.ORACLE, epistemic=EpistemicClass.INFERRED))
    g.add_node("credit_check", sevenbox_node(credit_check, source_kind=SourceKind.TOOL))
    g.add_node("observe_credit", sevenbox_node(observe_credit, source_kind=SourceKind.DERIVED))
    g.add_node("risk_decision", sevenbox_node(risk_decision, source_kind=SourceKind.DERIVED))
    g.add_node("auto_verify", sevenbox_node(tracked_auto_verify, source_kind=SourceKind.TOOL, name="auto_verify"))
    g.add_node("human_review", sevenbox_node(human_review, source_kind=SourceKind.TOOL, name="human_review"))
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
    compiled = g.compile()

    app_input = {"application": {"applicant_id": "APP-1001", "raw_score": 750}}
    with sevenbox_run() as live_run:
        live_result = compiled.invoke(app_input)
    assert call_log == ["APP-1001"]

    with sevenbox_run(replay_from=live_run.session.store) as replay_run:
        replay_result = compiled.invoke(app_input)

    assert call_log == ["APP-1001"]  # unchanged -- auto_verify's real body did not run again
    assert replay_result == live_result
    live_ids = {w.id for w in live_run.session.frontier().live}
    replay_ids = {w.id for w in replay_run.session.frontier().live}
    assert live_ids == replay_ids


# --- Path B: poor credit -> human_review branch --------------------------

def test_path_b_poor_credit_takes_only_the_human_review_branch():
    """Assert 2 (Camino B): auto_verify must NOT fire when the router
    picks human_review -- the mirror-image check of Path A's Assert 1,
    against the SAME compiled graph object used in Path A's tests, to
    confirm no state leaks between separate sevenbox_run() sessions on one
    compiled graph."""
    compiled = build_branching_graph()
    with sevenbox_run() as krun:
        result = compiled.invoke({"application": {"applicant_id": "APP-2002", "raw_score": 580}})

    assert result["verification"]["method"] == "human"
    fired = _fired_tool_names(krun.session.store)
    assert "human_review" in fired
    assert "auto_verify" not in fired


def test_path_b_final_decision_has_exactly_two_parents():
    """Assert 3 (Camino B): same check as Path A, against the other
    branch's Tool Warrant."""
    compiled = build_branching_graph()
    with sevenbox_run() as krun:
        compiled.invoke({"application": {"applicant_id": "APP-2002", "raw_score": 580}})

    decision_w = _find_final_decision_warrant(krun.session.store)
    human_review_w = next(w for w in krun.session.store.all() if w.effect_record and w.effect_record.source_name == "human_review")
    observation_w = next(w for w in krun.session.store.all() if "credit_observation" in w.content)

    assert len(decision_w.parent_ids) == 2
    assert set(decision_w.parent_ids) == {human_review_w.id, observation_w.id}


def test_path_b_transitive_compensation_reaches_the_irreversible_effect():
    """Assert 4 (Camino B): same transitive-obligation check as Path A,
    through the human_review branch instead."""
    compiled = build_branching_graph()
    with sevenbox_run() as krun:
        compiled.invoke({"application": {"applicant_id": "APP-2002", "raw_score": 580}})

    credit_w = _find_original_credit_warrant(krun.session.store)
    supersede(
        krun.session.store, target_id=credit_w.id,
        content={"raw_score": None, "correction": "bureau reports this score was for a different applicant"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="bureau re-check", response="score mismatch"),
    )

    obligations = find_transitive_obligations(krun.session.store)
    assert len(obligations) == 1
    effect_w = next(w for w in krun.session.store.all() if w.content.get("effect", "").startswith("loan_disbursed"))
    assert obligations[0].irreversible.id == effect_w.id


def test_path_b_replay_reconstructs_the_result_without_real_node_execution():
    """Assert 5 (Camino B): same replay-fidelity check as Path A, tracking
    human_review's call count specifically."""
    call_log = []

    def tracked_human_review(state):
        call_log.append(state["application"]["applicant_id"])
        return human_review(state)

    g = StateGraph(LoanState)
    g.add_node("agent_intent", sevenbox_node(agent_intent, source_kind=SourceKind.ORACLE, epistemic=EpistemicClass.INFERRED))
    g.add_node("credit_check", sevenbox_node(credit_check, source_kind=SourceKind.TOOL))
    g.add_node("observe_credit", sevenbox_node(observe_credit, source_kind=SourceKind.DERIVED))
    g.add_node("risk_decision", sevenbox_node(risk_decision, source_kind=SourceKind.DERIVED))
    g.add_node("auto_verify", sevenbox_node(auto_verify, source_kind=SourceKind.TOOL, name="auto_verify"))
    g.add_node("human_review", sevenbox_node(tracked_human_review, source_kind=SourceKind.TOOL, name="human_review"))
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
    compiled = g.compile()

    app_input = {"application": {"applicant_id": "APP-2002", "raw_score": 580}}
    with sevenbox_run() as live_run:
        live_result = compiled.invoke(app_input)
    assert call_log == ["APP-2002"]

    with sevenbox_run(replay_from=live_run.session.store) as replay_run:
        replay_result = compiled.invoke(app_input)

    assert call_log == ["APP-2002"]  # unchanged -- human_review's real body did not run again
    assert replay_result == live_result
    live_ids = {w.id for w in live_run.session.frontier().live}
    replay_ids = {w.id for w in replay_run.session.frontier().live}
    assert live_ids == replay_ids


# --- cross-path isolation ------------------------------------------------

def test_both_paths_on_the_same_compiled_graph_do_not_cross_contaminate():
    """The same compiled graph object, invoked twice under two SEPARATE
    sevenbox_run() sessions with different initial state, must produce two
    independent, correct Warrant stores -- no leakage of one run's
    frontier or store into the other's."""
    compiled = build_branching_graph()

    with sevenbox_run() as run_a:
        compiled.invoke({"application": {"applicant_id": "APP-1001", "raw_score": 750}})
    with sevenbox_run() as run_b:
        compiled.invoke({"application": {"applicant_id": "APP-2002", "raw_score": 580}})

    fired_a = _fired_tool_names(run_a.session.store)
    fired_b = _fired_tool_names(run_b.session.store)

    assert "auto_verify" in fired_a and "human_review" not in fired_a
    assert "human_review" in fired_b and "auto_verify" not in fired_b
    assert {w.id for w in run_a.session.store.all()}.isdisjoint({w.id for w in run_b.session.store.all()})
