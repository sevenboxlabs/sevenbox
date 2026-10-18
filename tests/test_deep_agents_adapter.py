"""Tests for sevenbox_adapters.deep_agents_adapter, run against the real
installed deepagents/langchain packages -- no framework mocking, only the
LLM call is faked (a real BaseChatModel subclass driving the real
create_agent()/create_deep_agent() graph, ToolNode, and middleware stack
end to end). See the adapter module docstring for the architecture this
verifies against.
"""
import time

import pytest

deepagents = pytest.importorskip("deepagents")
from deepagents import create_deep_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from warrant_core import ReversibilityClass, SourceKind, WarrantStore  # noqa: E402
from sevenbox_adapters.deep_agents_adapter import SevenboxMiddleware, sevenbox_deep_agent_run  # noqa: E402


class _BindableFakeChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel doesn't implement bind_tools; create_agent's
    model node calls it unconditionally on every model, real or fake."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _final_ai_message(text: str) -> AIMessage:
    return AIMessage(content=text)


def _tool_call_message(*calls: tuple[str, dict, str]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id} for name, args, call_id in calls],
    )


# ---------------------------------------------------------------------------
# Basic single tool call
# ---------------------------------------------------------------------------

def test_single_tool_call_becomes_a_warrant():
    calls = []

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        calls.append(order_id)
        return f"status for {order_id}: shipped"

    model = _BindableFakeChatModel(responses=[
        _tool_call_message(("lookup_order", {"order_id": "4471"}, "call_1")),
        _final_ai_message("Your order shipped."),
    ])

    with sevenbox_deep_agent_run() as krun:
        agent = create_deep_agent(model=model, tools=[lookup_order], middleware=[krun.middleware])
        agent.invoke({"messages": [{"role": "user", "content": "check order 4471"}]})

        tool_warrants = [w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL]
        assert len(tool_warrants) == 1
        assert tool_warrants[0].content == "status for 4471: shipped"
        assert tool_warrants[0].effect_record is not None
        assert tool_warrants[0].effect_record.source_name == "lookup_order"
    assert calls == ["4471"]


# ---------------------------------------------------------------------------
# Parallel tool calls: must NOT manufacture a false dependency between them
# ---------------------------------------------------------------------------

def test_parallel_tool_calls_get_the_same_correct_shared_parent():
    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return f"status for {order_id}: shipped"

    @tool
    def check_watchlist(name: str) -> str:
        """Check a watchlist."""
        return f"watchlist result for {name}: clear"

    model = _BindableFakeChatModel(responses=[
        _tool_call_message(
            ("lookup_order", {"order_id": "4471"}, "call_1"),
            ("check_watchlist", {"name": "acme"}, "call_2"),
        ),
        _final_ai_message("Order shipped, watchlist clear."),
    ])

    with sevenbox_deep_agent_run() as krun:
        agent = create_deep_agent(model=model, tools=[lookup_order, check_watchlist], middleware=[krun.middleware])
        agent.invoke({"messages": [{"role": "user", "content": "check order 4471 against acme"}]})

        tool_warrants = {w.effect_record.source_name: w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL}
        assert set(tool_warrants) == {"lookup_order", "check_watchlist"}
        order_w = tool_warrants["lookup_order"]
        watchlist_w = tool_warrants["check_watchlist"]

        # Neither call's Warrant may cite the other as a parent -- that would
        # be exactly the manufactured false dependency this mechanism exists
        # to prevent (see module docstring point 3 / point 5).
        assert order_w.id not in watchlist_w.parent_ids
        assert watchlist_w.id not in order_w.parent_ids
        assert order_w.parent_ids == watchlist_w.parent_ids

        # Both siblings land in the frontier together afterward.
        live_ids = {w.id for w in krun.session.frontier().live}
        assert order_w.id in live_ids
        assert watchlist_w.id in live_ids


# ---------------------------------------------------------------------------
# Subagent delegation
# ---------------------------------------------------------------------------

def test_subagent_delegation_roots_its_own_warrants_at_the_delegation_decision():
    @tool
    def score_credit(applicant_id: str) -> str:
        """Score credit for an applicant."""
        return f"credit score for {applicant_id}: 710"

    main_model = _BindableFakeChatModel(responses=[
        _tool_call_message(("task", {"description": "score A-1", "subagent_type": "credit-scorer"}, "call_task_1")),
        _final_ai_message("Delegated and got the score."),
    ])
    sub_model = _BindableFakeChatModel(responses=[
        _tool_call_message(("score_credit", {"applicant_id": "A-1"}, "call_sub_1")),
        _final_ai_message("Credit score is 710."),
    ])

    with sevenbox_deep_agent_run() as krun:
        agent = create_deep_agent(
            model=main_model,
            tools=[],
            middleware=[krun.middleware],
            subagents=[{
                "name": "credit-scorer",
                "description": "Scores credit for an applicant.",
                "system_prompt": "Score the applicant's credit.",
                "model": sub_model,
                "tools": [score_credit],
                "middleware": [krun.middleware],
            }],
        )
        agent.invoke({"messages": [{"role": "user", "content": "process applicant A-1"}]})

        by_content = {repr(w.content)[:40]: w for w in krun.session.store.all()}
        delegation_start = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.DERIVED)
        subagent_tool_w = next(
            w for w in krun.session.store.all()
            if w.source_kind == SourceKind.TOOL and w.effect_record and w.effect_record.source_name == "score_credit"
        )
        outer_task_w = next(
            w for w in krun.session.store.all()
            if w.source_kind == SourceKind.TOOL and w.effect_record and w.effect_record.source_name == "task"
        )

        # The subagent's own tool call is rooted at the delegation decision,
        # not at the main agent's frontier directly.
        assert subagent_tool_w.parent_ids == (delegation_start.id,)
        # The outer task-call Warrant's parent is the delegation decision
        # ONLY -- not enriched with the subagent's own results, deliberately
        # (see the adapter's _wrap_delegation docstring: a richer parent set
        # here would break replay's ability to look up this call's cached
        # response without first re-running the subagent to produce it).
        assert outer_task_w.parent_ids == (delegation_start.id,)
        # The delegation decision itself must NOT spuriously cite anything
        # from inside the subagent (it was constructed before the subagent
        # ran at all).
        assert subagent_tool_w.id not in delegation_start.parent_ids


def test_two_independent_subagent_runs_do_not_cross_contaminate():
    """Two SEPARATE task calls (sequential turns) to the SAME subagent type
    must not leak state between invocations -- each subagent run gets its
    own fresh nested session."""
    call_log = []

    @tool
    def score_credit(applicant_id: str) -> str:
        """Score credit for an applicant."""
        call_log.append(applicant_id)
        return f"credit score for {applicant_id}: 700"

    main_model = _BindableFakeChatModel(responses=[
        _tool_call_message(("task", {"description": "score A-1", "subagent_type": "scorer"}, "call_task_1")),
        _tool_call_message(("task", {"description": "score A-2", "subagent_type": "scorer"}, "call_task_2")),
        _final_ai_message("Both scored."),
    ])
    # Cycles: each task invocation gets its own fresh sub_model instance
    # sharing this list would exhaust after the first subagent run since
    # FakeMessagesListChatModel cycles per-instance, not per-invocation --
    # a fresh instance per subagent config for this test:

    def _new_sub_model():
        return _BindableFakeChatModel(responses=[
            _tool_call_message(("score_credit", {"applicant_id": "PLACEHOLDER"}, "call_sub")),
            _final_ai_message("scored"),
        ])

    with sevenbox_deep_agent_run() as krun:
        agent = create_deep_agent(
            model=main_model,
            tools=[],
            middleware=[krun.middleware],
            subagents=[{
                "name": "scorer",
                "description": "Scores credit.",
                "system_prompt": "Score it.",
                "model": _new_sub_model(),
                "tools": [score_credit],
                "middleware": [krun.middleware],
            }],
        )
        agent.invoke({"messages": [{"role": "user", "content": "process A-1 and A-2"}]})

        delegation_starts = [w for w in krun.session.store.all() if w.source_kind == SourceKind.DERIVED]
        assert len(delegation_starts) == 2
        subagent_tool_warrants = [
            w for w in krun.session.store.all()
            if w.source_kind == SourceKind.TOOL and w.effect_record and w.effect_record.source_name == "score_credit"
        ]
        assert len(subagent_tool_warrants) == 2
        # Each subagent tool warrant is rooted at exactly one delegation
        # start, and the two delegation starts are different Warrants --
        # no shared/leaked nested-session state between the two calls.
        parents = {w.parent_ids for w in subagent_tool_warrants}
        assert len(parents) == 2  # each subagent call has a DISTINCT parent tuple


def test_delegation_and_subagent_warrants_collapse_when_content_coincides():
    """Real, verified consequence of content-addressing crossing a
    delegation boundary: if a subagent's sole tool call is rooted directly
    at delegation_start (nothing else in between) AND the subagent's final
    reported text is byte-identical to that tool's raw return value, the
    outer task-call Warrant and the subagent's own tool-call Warrant have
    identical (content, parent_ids, source_kind) -- same id --
    and collapse into ONE stored Warrant, not two. This is correct, not a
    bug: two claims with the same content and the same justification ARE
    the same Warrant, regardless of which "side" of a delegation boundary
    observed them. Found while adversarially testing this adapter (an
    earlier test written assuming the two would always be distinct failed
    with StopIteration, not a wrong answer, until this was understood)."""

    @tool
    def score_credit(applicant_id: str) -> str:
        """Score credit for an applicant."""
        return "710"

    main_model = _BindableFakeChatModel(responses=[
        _tool_call_message(("task", {"description": "score A-1", "subagent_type": "credit-scorer"}, "call_task_1")),
        _final_ai_message("Approved based on score."),
    ])
    sub_model = _BindableFakeChatModel(responses=[
        _tool_call_message(("score_credit", {"applicant_id": "A-1"}, "call_sub_1")),
        _final_ai_message("710"),  # byte-identical to score_credit's own return value
    ])

    with sevenbox_deep_agent_run() as krun:
        agent = create_deep_agent(
            model=main_model, tools=[], middleware=[krun.middleware],
            subagents=[{
                "name": "credit-scorer", "description": "Scores credit.",
                "system_prompt": "Score it.", "model": sub_model,
                "tools": [score_credit], "middleware": [krun.middleware],
            }],
        )
        agent.invoke({"messages": [{"role": "user", "content": "process A-1"}]})

        tool_warrants = [w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL]
        # Exactly one TOOL warrant, not two -- the collapse.
        assert len(tool_warrants) == 1
        assert tool_warrants[0].content == "710"
        # Its effect_record still carries the LAST writer's source_name
        # ("task", since the outer wrap runs after the inner one and
        # store.add() on a colliding id keeps the FIRST writer's stored
        # Warrant unchanged -- see warrant_core.store.WarrantStore.add's
        # idempotent no-op) -- asserted here as the actual observed
        # behavior, not a claim about which "should" win.
        assert tool_warrants[0].effect_record.source_name == "score_credit"


# ---------------------------------------------------------------------------
# Replay: must not re-execute, must use cached content
# ---------------------------------------------------------------------------

def test_replay_does_not_re_execute_the_real_tool():
    calls = []

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        calls.append(order_id)
        return f"status for {order_id}: shipped"

    model = _BindableFakeChatModel(responses=[
        _tool_call_message(("lookup_order", {"order_id": "4471"}, "call_1")),
        _final_ai_message("Your order shipped."),
    ])

    live_store = WarrantStore()
    with sevenbox_deep_agent_run(store=live_store) as krun:
        agent = create_deep_agent(model=model, tools=[lookup_order], middleware=[krun.middleware])
        agent.invoke({"messages": [{"role": "user", "content": "check order 4471"}]})
    assert calls == ["4471"]

    # Replay: same tool, same script, but the tool body must never run again.
    model2 = _BindableFakeChatModel(responses=[
        _tool_call_message(("lookup_order", {"order_id": "4471"}, "call_1")),
        _final_ai_message("Your order shipped."),
    ])
    with sevenbox_deep_agent_run(replay_from=live_store) as krun2:
        agent2 = create_deep_agent(model=model2, tools=[lookup_order], middleware=[krun2.middleware])
        agent2.invoke({"messages": [{"role": "user", "content": "check order 4471"}]})

    assert calls == ["4471"], "replay must not invoke the real tool body a second time"
    replayed_tool_w = next(w for w in krun2.session.store.all() if w.source_kind == SourceKind.TOOL)
    assert replayed_tool_w.content == "status for 4471: shipped"


def test_replay_of_a_delegation_does_not_re_run_the_subagent():
    """The subagent's own real tool must not fire during replay -- the
    task call is treated as one opaque effect at the delegation boundary
    (module docstring's "What replay does not cover")."""
    sub_calls = []

    @tool
    def score_credit(applicant_id: str) -> str:
        """Score credit for an applicant."""
        sub_calls.append(applicant_id)
        return f"credit score for {applicant_id}: 710"

    def _main_model():
        return _BindableFakeChatModel(responses=[
            _tool_call_message(("task", {"description": "score A-1", "subagent_type": "credit-scorer"}, "call_task_1")),
            _final_ai_message("Delegated and got the score."),
        ])

    def _sub_model():
        return _BindableFakeChatModel(responses=[
            _tool_call_message(("score_credit", {"applicant_id": "A-1"}, "call_sub_1")),
            _final_ai_message("Credit score is 710."),
        ])

    live_store = WarrantStore()
    with sevenbox_deep_agent_run(store=live_store) as krun:
        agent = create_deep_agent(
            model=_main_model(), tools=[], middleware=[krun.middleware],
            subagents=[{
                "name": "credit-scorer", "description": "Scores credit.",
                "system_prompt": "Score it.", "model": _sub_model(),
                "tools": [score_credit], "middleware": [krun.middleware],
            }],
        )
        agent.invoke({"messages": [{"role": "user", "content": "process A-1"}]})
    assert sub_calls == ["A-1"]

    with sevenbox_deep_agent_run(replay_from=live_store) as krun2:
        agent2 = create_deep_agent(
            model=_main_model(), tools=[], middleware=[krun2.middleware],
            subagents=[{
                "name": "credit-scorer", "description": "Scores credit.",
                "system_prompt": "Score it.", "model": _sub_model(),
                "tools": [score_credit], "middleware": [krun2.middleware],
            }],
        )
        agent2.invoke({"messages": [{"role": "user", "content": "process A-1"}]})

    assert sub_calls == ["A-1"], "replay must not re-run the subagent's own real tool"
    # Replay of a delegation reconstructs the delegation-decision Warrant
    # (DERIVED, PURE, cheap, and needed as the stable replay-lookup key --
    # see _wrap_delegation's docstring) and the outer task-call Warrant,
    # but never regenerates the subagent's OWN internal tool-call Warrant --
    # that would require actually running the subagent, which replay exists
    # to avoid.
    replayed_score_warrants = [
        w for w in krun2.session.store.all()
        if w.source_kind == SourceKind.TOOL and w.effect_record and w.effect_record.source_name == "score_credit"
    ]
    assert replayed_score_warrants == []
    task_warrants = [
        w for w in krun2.session.store.all()
        if w.source_kind == SourceKind.TOOL and w.effect_record and w.effect_record.source_name == "task"
    ]
    assert len(task_warrants) == 1


# ---------------------------------------------------------------------------
# Real concurrency: many parallel tool calls, genuine OS threads
# ---------------------------------------------------------------------------

def test_many_parallel_tool_calls_under_real_thread_dispatch_produce_no_lost_or_duplicate_warrants():
    """langgraph.prebuilt.tool_node.ToolNode dispatches the sync path via a
    real ThreadPoolExecutor (module docstring point 3) -- this test forces
    a wide batch with real per-call latency, so the dispatch genuinely
    overlaps, not just plausibly could."""
    N = 12

    def _make_tool(i):
        @tool(name_or_callable=f"probe_{i}")
        def _probe(x: str) -> str:
            """Probe tool for concurrency stress testing."""
            time.sleep(0.01)
            return f"result-{i}-{x}"
        return _probe

    tools = [_make_tool(i) for i in range(N)]
    calls = [(f"probe_{i}", {"x": str(i)}, f"call_{i}") for i in range(N)]

    model = _BindableFakeChatModel(responses=[
        _tool_call_message(*calls),
        _final_ai_message("done"),
    ])

    with sevenbox_deep_agent_run() as krun:
        agent = create_deep_agent(model=model, tools=tools, middleware=[krun.middleware])
        agent.invoke({"messages": [{"role": "user", "content": "run all probes"}], "max_concurrency": N})

        tool_warrants = [w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL]
        assert len(tool_warrants) == N, f"expected {N} tool warrants, got {len(tool_warrants)} (lost or duplicated)"
        source_names = {w.effect_record.source_name for w in tool_warrants}
        assert source_names == {f"probe_{i}" for i in range(N)}
        # Every sibling in this single parallel turn shares the identical
        # parent tuple -- none chained onto another.
        parent_tuples = {w.parent_ids for w in tool_warrants}
        assert len(parent_tuples) == 1
        # The store-wide frontier includes these N Warrants plus their axiom
        # ancestor (nothing superseded anything here, so the axiom stays
        # live alongside its descendants -- supersession, not ancestry, is
        # what's being checked here).
        # The precise claim this test makes is the subset check: none of
        # the N calls were lost, and the axiom is the only extra member.
        live_ids = {w.id for w in krun.session.frontier().live}
        tool_ids = {w.id for w in tool_warrants}
        assert tool_ids.issubset(live_ids)
        assert live_ids - tool_ids == set(session_axiom_ids(krun.session.store))


def session_axiom_ids(store):
    return {w.id for w in store.all() if w.source_kind == SourceKind.AXIOM}


# ---------------------------------------------------------------------------
# Compensation across a delegation boundary
# ---------------------------------------------------------------------------

def test_compensation_reaches_a_correction_to_the_delegation_decision_itself():
    """A correction landing on `delegation_start` itself (the "decision to
    delegate" Warrant -- e.g. a later determination that this delegation
    should never have happened) DOES surface a transitive compensation
    obligation on whatever irreversible action the main agent derived from
    the task call's result, because `delegation_start` genuinely is a
    causal ancestor of the outer task-call Warrant (`_wrap_delegation`
    always sets `parent_ids=(delegation_start.id,)`)."""
    from warrant_core import EpistemicClass, create_warrant, find_transitive_obligations

    @tool
    def score_credit(applicant_id: str) -> str:
        """Score credit for an applicant."""
        return "710"

    main_model = _BindableFakeChatModel(responses=[
        _tool_call_message(("task", {"description": "score A-1", "subagent_type": "credit-scorer"}, "call_task_1")),
        _final_ai_message("Approved based on score."),
    ])
    sub_model = _BindableFakeChatModel(responses=[
        _tool_call_message(("score_credit", {"applicant_id": "A-1"}, "call_sub_1")),
        _final_ai_message("Credit score is 710."),
    ])

    with sevenbox_deep_agent_run() as krun:
        agent = create_deep_agent(
            model=main_model, tools=[], middleware=[krun.middleware],
            subagents=[{
                "name": "credit-scorer", "description": "Scores credit.",
                "system_prompt": "Score it.", "model": sub_model,
                "tools": [score_credit], "middleware": [krun.middleware],
            }],
        )
        agent.invoke({"messages": [{"role": "user", "content": "process A-1"}]})

        outer_task_w = next(
            w for w in krun.session.store.all()
            if w.source_kind == SourceKind.TOOL and w.effect_record and w.effect_record.source_name == "task"
        )
        delegation_start = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.DERIVED)

        approval = create_warrant(
            content={"decision": "loan_approved"}, source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.IRREVERSIBLE,
            parent_ids=(outer_task_w.id,),
        )
        krun.session.store.add(approval)

        correction = create_warrant(
            content={"delegating_to": "credit-scorer", "task": "score A-1", "note": "should not have delegated"},
            source_kind=SourceKind.DERIVED, epistemic=EpistemicClass.ASSUMED,
            reversibility=ReversibilityClass.PURE,
            parent_ids=(delegation_start.id,), supersedes_id=delegation_start.id,
        )
        krun.session.store.add(correction)

        obligations = find_transitive_obligations(krun.session.store)
        assert any(o.irreversible.id == approval.id for o in obligations)


def test_KNOWN_LIMITATION_compensation_does_not_reach_inside_a_subagents_own_internal_correction():
    """Real, adversarially-found limitation, documented rather than hidden
    (see SevenboxMiddleware._wrap_delegation's docstring): the outer task-call
    Warrant's `parent_ids` is fixed to `(delegation_start.id,)` ONLY --
    deliberately, so a replay lookup key is knowable before the subagent
    ever runs (see the docstring's account of the earlier, reverted design
    that tried to also cite the subagent's own final results and broke
    replay by construction). One direct cost of that choice: a correction
    landing on a Warrant *inside* the subagent's own reasoning -- anything
    deeper than delegation_start itself -- is NOT a causal ancestor of the
    outer task-call Warrant, so `find_transitive_obligations` cannot walk
    back to it from an irreversible action the main agent derived from the
    delegation's result. This test pins that real, current gap down
    explicitly rather than letting it go unnoticed: if a future change
    closes it, this test should start failing and should be rewritten to
    assert the new, stronger guarantee -- not silently deleted."""
    from warrant_core import EpistemicClass, create_warrant, find_transitive_obligations

    @tool
    def score_credit(applicant_id: str) -> str:
        """Score credit for an applicant."""
        return "710"

    main_model = _BindableFakeChatModel(responses=[
        _tool_call_message(("task", {"description": "score A-1", "subagent_type": "credit-scorer"}, "call_task_1")),
        _final_ai_message("Approved based on score."),
    ])
    sub_model = _BindableFakeChatModel(responses=[
        _tool_call_message(("score_credit", {"applicant_id": "A-1"}, "call_sub_1")),
        _final_ai_message("Credit score is 710."),
    ])

    with sevenbox_deep_agent_run() as krun:
        agent = create_deep_agent(
            model=main_model, tools=[], middleware=[krun.middleware],
            subagents=[{
                "name": "credit-scorer", "description": "Scores credit.",
                "system_prompt": "Score it.", "model": sub_model,
                "tools": [score_credit], "middleware": [krun.middleware],
            }],
        )
        agent.invoke({"messages": [{"role": "user", "content": "process A-1"}]})

        outer_task_w = next(
            w for w in krun.session.store.all()
            if w.source_kind == SourceKind.TOOL and w.effect_record and w.effect_record.source_name == "task"
        )
        subagent_score_w = next(
            w for w in krun.session.store.all()
            if w.source_kind == SourceKind.TOOL and w.effect_record and w.effect_record.source_name == "score_credit"
        )
        assert subagent_score_w.id not in outer_task_w.parent_ids  # the gap, structurally

        approval = create_warrant(
            content={"decision": "loan_approved"}, source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.IRREVERSIBLE,
            parent_ids=(outer_task_w.id,),
        )
        krun.session.store.add(approval)

        correction = create_warrant(
            content="580 (corrected)", source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.ASSUMED, reversibility=ReversibilityClass.PURE,
            parent_ids=(subagent_score_w.id,), supersedes_id=subagent_score_w.id,
        )
        krun.session.store.add(correction)

        obligations = find_transitive_obligations(krun.session.store)
        assert not any(o.irreversible.id == approval.id for o in obligations), (
            "this assertion documents the CURRENT gap -- if it now fails, the gap "
            "has been closed by a design change and this test should be rewritten "
            "to assert the new guarantee, not deleted"
        )
        # The correction is still real and findable via a direct, non-transitive
        # query against the subagent's own claim -- nothing is lost, it just
        # doesn't automatically propagate outward across this boundary today.
        assert krun.session.store.is_superseded(subagent_score_w.id)


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------

def test_two_separate_sevenbox_runs_do_not_leak_into_each_other():
    @tool
    def echo(x: str) -> str:
        """Echo input."""
        return x

    model_a = _BindableFakeChatModel(responses=[
        _tool_call_message(("echo", {"x": "a"}, "call_a")),
        _final_ai_message("done a"),
    ])
    model_b = _BindableFakeChatModel(responses=[
        _tool_call_message(("echo", {"x": "b"}, "call_b")),
        _final_ai_message("done b"),
    ])

    with sevenbox_deep_agent_run() as krun_a:
        agent_a = create_deep_agent(model=model_a, tools=[echo], middleware=[krun_a.middleware])
        agent_a.invoke({"messages": [{"role": "user", "content": "a"}]})

    with sevenbox_deep_agent_run() as krun_b:
        agent_b = create_deep_agent(model=model_b, tools=[echo], middleware=[krun_b.middleware])
        agent_b.invoke({"messages": [{"role": "user", "content": "b"}]})

    assert krun_a.session.store is not krun_b.session.store
    assert len(krun_a.session.store.all()) == len(krun_b.session.store.all())
    a_contents = {w.content for w in krun_a.session.store.all() if w.source_kind == SourceKind.TOOL}
    b_contents = {w.content for w in krun_b.session.store.all() if w.source_kind == SourceKind.TOOL}
    assert a_contents == {"a"}
    assert b_contents == {"b"}


def test_tool_call_outside_a_sevenbox_run_raises_rather_than_silently_no_opping():
    @tool
    def echo(x: str) -> str:
        """Echo input."""
        return x

    model = _BindableFakeChatModel(responses=[
        _tool_call_message(("echo", {"x": "a"}, "call_a")),
        _final_ai_message("done"),
    ])
    agent = create_deep_agent(model=model, tools=[echo], middleware=[SevenboxMiddleware()])
    with pytest.raises(RuntimeError, match="outside a"):
        agent.invoke({"messages": [{"role": "user", "content": "a"}]})
