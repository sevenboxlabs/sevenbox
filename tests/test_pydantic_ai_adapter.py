"""
Real-pydantic-ai integration tests for sevenbox_adapters.pydantic_ai_adapter.

Uses pydantic_ai.models.function.FunctionModel -- a real, network-free
Model implementation the SDK itself provides for offline testing -- to
drive a real Agent through real @agent.tool-registered tool calls. This
is the same "verify against the actual installed SDK, not training-data
assumptions" standard the other two adapters' test files hold to.
"""
import asyncio
import contextlib
import sys
import threading

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")
from pydantic_ai import Agent, RunContext  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402

from warrant_core import EpistemicClass, SourceKind, compute_frontier  # noqa: E402
from sevenbox_adapters.pydantic_ai_adapter import sevenbox_agent_run, sevenbox_tool  # noqa: E402


# --- Assumption-pinning test: run this FIRST, in isolation from the ---
# --- adapter's own code, so a future pydantic-ai upgrade that changed ---
# --- run_step's behavior fails loudly here, not as a mysterious ---
# --- parent-attribution bug somewhere else. ---

def test_run_step_behavior_matches_the_documented_assumption():
    """Documents, empirically, the exact SDK behavior
    sevenbox_adapters.pydantic_ai_adapter's turn-aware fan-out design
    depends on: two tool calls the model requests in the SAME turn get
    an IDENTICAL run_step; sequential calls across different turns get
    increasing, distinct run_step values. No Sevenbox code involved here at
    all -- this is pinned directly against bare pydantic_ai."""
    observed = []

    def fake_model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:  # turn 1: two calls requested at once
            return ModelResponse(parts=[
                ToolCallPart(tool_name="tool_a", args={}),
                ToolCallPart(tool_name="tool_b", args={}),
            ])
        if len(messages) == 3:  # turn 2: one more, sequential
            return ModelResponse(parts=[ToolCallPart(tool_name="tool_c", args={})])
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(FunctionModel(fake_model_fn))

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        observed.append(("tool_a", ctx.run_step))
        return "a"

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        observed.append(("tool_b", ctx.run_step))
        return "b"

    @agent.tool
    def tool_c(ctx: RunContext) -> str:
        observed.append(("tool_c", ctx.run_step))
        return "c"

    agent.run_sync("go")

    steps = dict(observed)
    assert steps["tool_a"] == steps["tool_b"], "genuinely parallel calls must share run_step"
    assert steps["tool_c"] != steps["tool_a"], "a later, sequential turn must get a different run_step"
    assert steps["tool_c"] > steps["tool_a"], "run_step must increase across turns"


def test_run_id_behavior_matches_the_documented_assumption():
    """Companion to test_run_step_behavior_matches_the_documented_
    assumption, pinning the OTHER half of the (run_id, run_step) turn
    identity Fix (test_a_fresh_run_never_matches_a_stale_closed_turn_
    from_a_prior_run) depends on, in isolation from any Sevenbox code:
    run_id must be non-None, IDENTICAL across every call within one
    agent.run()/run_sync() call (including across genuinely parallel
    siblings), and DIFFERENT across two separate run_sync() calls made
    from the same Agent instance. Previously verified only indirectly,
    through the full adapter's cross-run regression test -- if a future
    pydantic-ai release changed run_id's behavior (e.g. defaulting it to
    None under some run mode), this test fails here, specifically,
    rather than degrading the cross-run fix back to its pre-fix bug
    silently."""
    observed_run1 = []
    observed_run2 = []

    def fake_model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name="tool_a", args={}),
                ToolCallPart(tool_name="tool_b", args={}),
            ])
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(FunctionModel(fake_model_fn))

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        observed_run1.append(ctx.run_id)
        return "a"

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        observed_run1.append(ctx.run_id)
        return "b"

    agent.run_sync("go")

    assert all(rid is not None for rid in observed_run1), "run_id must never be None"
    assert len(set(observed_run1)) == 1, (
        f"run_id must be identical across every call within one run, including "
        f"genuinely parallel siblings, got: {observed_run1}"
    )

    # A second, separate run_sync() call from the SAME Agent instance.
    def fake_model_fn_2(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="tool_c", args={})])
        return ModelResponse(parts=[TextPart(content="done")])

    agent2 = Agent(FunctionModel(fake_model_fn_2))

    @agent2.tool
    def tool_c(ctx: RunContext) -> str:
        observed_run2.append(ctx.run_id)
        return "c"

    agent2.run_sync("go again")

    assert observed_run2[0] is not None
    assert observed_run2[0] not in set(observed_run1), (
        "run_id must be unique per agent.run()/run_sync() call -- a second, "
        "separate run must never share the first run's run_id"
    )


# --- Sequential scenario (mirrors the other two adapters' basic test) ---

def _build_sequential_agent():
    def fake_model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="lookup_order", args={"order_id": "4471"})])
        return ModelResponse(parts=[TextPart(content="Your order has shipped.")])

    agent = Agent(FunctionModel(fake_model_fn))

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="lookup_order")
    def lookup_order(ctx: RunContext, order_id: str) -> dict:
        return {"status": "shipped", "order_id": order_id}

    return agent


def test_tool_called_outside_sevenbox_agent_run_raises_clear_error():
    agent = _build_sequential_agent()
    with pytest.raises(RuntimeError, match="outside a `with sevenbox_agent_run"):
        agent.run_sync("Where is order 4471?")


def test_wrapped_agent_populates_a_warrant_store():
    agent = _build_sequential_agent()

    with sevenbox_agent_run() as krun:
        krun.session.start("Where is order 4471?")
        result = agent.run_sync("Where is order 4471?")

    assert result.output == "Your order has shipped."
    # explicit axiom (the user's prompt) + 1 tool call = 2 warrants
    assert len(krun.session.store) == 2

    live = compute_frontier(krun.session.store).live
    kinds = {w.source_kind for w in live}
    assert SourceKind.TOOL in kinds
    assert SourceKind.AXIOM in kinds

    tool_warrant = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    assert tool_warrant.content == {"status": "shipped", "order_id": "4471"}
    assert tool_warrant.effect_record is not None
    assert tool_warrant.effect_record.source_name == "lookup_order"
    assert len(tool_warrant.parent_ids) == 1


def test_live_then_replay_never_calls_the_real_tool_again():
    call_log = []

    def fake_model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="lookup_order", args={"order_id": "4471"})])
        return ModelResponse(parts=[TextPart(content="Your order has shipped.")])

    agent = Agent(FunctionModel(fake_model_fn))

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="lookup_order")
    def lookup_order(ctx: RunContext, order_id: str) -> dict:
        call_log.append(order_id)
        return {"status": "shipped", "order_id": order_id}

    with sevenbox_agent_run() as live_run:
        live_run.session.start("Where is order 4471?")
        agent.run_sync("Where is order 4471?")
    assert call_log == ["4471"]

    with sevenbox_agent_run(replay_from=live_run.session.store) as replay_run:
        replay_run.session.start("Where is order 4471?")
        agent.run_sync("Where is order 4471?")

    assert call_log == ["4471"]  # unchanged -- the real tool body did not run again
    assert {w.id for w in replay_run.session.store.all()} == {w.id for w in live_run.session.store.all()}


# --- Genuine parallel fan-out: THE scenario this adapter's whole design ---
# --- exists for -- no hooks, no ADDITIONAL_PARENTS_KEY, just correct by ---
# --- default. ---

def _build_parallel_kyc_agent():
    def fake_model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:  # ONE turn, TWO tool calls -- genuine fan-out
            return ModelResponse(parts=[
                ToolCallPart(tool_name="check_watchlist", args={"applicant_id": "A-991"}),
                ToolCallPart(tool_name="verify_document", args={"applicant_id": "A-991"}),
            ])
        return ModelResponse(parts=[TextPart(content="KYC complete.")])

    agent = Agent(FunctionModel(fake_model_fn))

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="check_watchlist")
    def check_watchlist(ctx: RunContext, applicant_id: str) -> dict:
        return {"applicant_id": applicant_id, "watchlist_match": False}

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="verify_document")
    def verify_document(ctx: RunContext, applicant_id: str) -> dict:
        return {"applicant_id": applicant_id, "document_verified": True}

    return agent


def test_parallel_tool_calls_get_correct_independent_parents_with_zero_extra_wiring():
    """The headline claim of this adapter's design: no hooks=, no
    ADDITIONAL_PARENTS_KEY, no opt-in of any kind -- just
    @agent.tool + @sevenbox_tool, and genuinely parallel calls still get
    correct, independent parents."""
    agent = _build_parallel_kyc_agent()

    with sevenbox_agent_run() as krun:
        krun.session.start("Run KYC on A-991")
        agent.run_sync("Run KYC on A-991")

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    watchlist_w = next(w for w in warrants.values() if "watchlist_match" in w.content)
    document_w = next(w for w in warrants.values() if "document_verified" in w.content)

    assert watchlist_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids != (watchlist_w.id,), "no false dependency between siblings"


def test_parallel_turn_before_any_explicit_start_shares_one_freshly_created_axiom():
    """Previously untested (found by code review): every other parallel
    test calls krun.session.start(...) explicitly before agent.run_sync().
    This one does not -- the very first turn already fans out to two
    genuinely parallel tool calls, so _resolve_parent_ids's own lazy
    axiom creation (self._start_locked(fallback_axiom_content), guarded
    by the same lock Fix #1 added) is what has to ensure exactly one
    axiom is created and shared, under real thread-pool dispatch, not
    the cooperative-async case this same scenario exercises in the
    OpenAI Agents SDK adapter's tests."""
    agent = _build_parallel_kyc_agent()

    with sevenbox_agent_run() as krun:
        # Deliberately no krun.session.start(...) call.
        agent.run_sync("Run KYC on A-991")

    warrants = {w.id: w for w in krun.session.store.all()}
    axioms = [w for w in warrants.values() if w.source_kind == SourceKind.AXIOM]
    watchlist_w = next(w for w in warrants.values() if "watchlist_match" in w.content)
    document_w = next(w for w in warrants.values() if "document_verified" in w.content)

    assert len(axioms) == 1, f"expected exactly one lazily-created axiom, got {len(axioms)}"
    assert watchlist_w.parent_ids == (axioms[0].id,)
    assert document_w.parent_ids == (axioms[0].id,)


def test_sequential_calls_still_chain_correctly_when_not_parallel():
    """The turn-aware logic must not over-correct: a SECOND, later,
    sequential call (different run_step) should still chain onto the
    frontier the first turn left behind, exactly like the other two
    adapters' ordinary case."""
    def fake_model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="check_watchlist", args={"applicant_id": "A-991"})])
        if len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart(tool_name="verify_document", args={"applicant_id": "A-991"})])
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(FunctionModel(fake_model_fn))

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="check_watchlist")
    def check_watchlist(ctx: RunContext, applicant_id: str) -> dict:
        return {"applicant_id": applicant_id, "watchlist_match": False}

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="verify_document")
    def verify_document(ctx: RunContext, applicant_id: str) -> dict:
        return {"applicant_id": applicant_id, "document_verified": True}

    with sevenbox_agent_run() as krun:
        krun.session.start("Run KYC on A-991")
        agent.run_sync("Run KYC on A-991")

    warrants = {w.id: w for w in krun.session.store.all()}
    watchlist_w = next(w for w in warrants.values() if "watchlist_match" in w.content)
    document_w = next(w for w in warrants.values() if "document_verified" in w.content)
    # sequential this time: document_w SHOULD chain onto watchlist_w
    assert document_w.parent_ids == (watchlist_w.id,)


# --- tool_plain (no RunContext) must raise, not silently mis-attribute ---

def test_sevenbox_tool_on_a_context_free_call_raises_a_clear_error():
    """A function invoked without a RunContext-carrying ctx (e.g. hand-
    called directly rather than through @agent.tool machinery) must
    raise, not silently guess at a parent."""
    @sevenbox_tool(source_kind=SourceKind.TOOL)
    def lookup_order(ctx, order_id: str) -> dict:
        return {"status": "shipped", "order_id": order_id}

    class _NoRunStep:
        pass

    with sevenbox_agent_run():
        with pytest.raises(TypeError, match="RunContext"):
            lookup_order(_NoRunStep(), "4471")


# --- async tool support ---------------------------------------------------

def test_async_tool_bodies_are_awaited_and_captured_correctly():
    call_log = []

    def fake_model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="async_lookup", args={"order_id": "4471"})])
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(FunctionModel(fake_model_fn))

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="async_lookup")
    async def async_lookup(ctx: RunContext, order_id: str) -> dict:
        await asyncio.sleep(0)  # real await, not just an async def that never suspends
        call_log.append(order_id)
        return {"status": "shipped", "order_id": order_id}

    with sevenbox_agent_run() as krun:
        krun.session.start("Where is order 4471?")
        result = agent.run_sync("Where is order 4471?")

    assert call_log == ["4471"]
    assert result.output == "done"
    tool_warrant = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    assert tool_warrant.content == {"status": "shipped", "order_id": "4471"}
    assert tool_warrant.effect_record.source_name == "async_lookup"


def test_async_parallel_tool_calls_also_get_independent_parents():
    """The turn-aware fan-out fix must hold for async tools too, not
    only sync ones -- genuinely parallel async calls, same guarantee."""
    def fake_model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name="check_watchlist", args={"applicant_id": "A-991"}),
                ToolCallPart(tool_name="verify_document", args={"applicant_id": "A-991"}),
            ])
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(FunctionModel(fake_model_fn))

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="check_watchlist")
    async def check_watchlist(ctx: RunContext, applicant_id: str) -> dict:
        await asyncio.sleep(0)
        return {"applicant_id": applicant_id, "watchlist_match": False}

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="verify_document")
    async def verify_document(ctx: RunContext, applicant_id: str) -> dict:
        await asyncio.sleep(0)
        return {"applicant_id": applicant_id, "document_verified": True}

    with sevenbox_agent_run() as krun:
        krun.session.start("Run KYC on A-991")
        agent.run_sync("Run KYC on A-991")

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    watchlist_w = next(w for w in warrants.values() if "watchlist_match" in w.content)
    document_w = next(w for w in warrants.values() if "document_verified" in w.content)

    assert watchlist_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids == (axiom.id,)


def test_mixed_sync_and_async_tools_in_the_same_parallel_turn():
    """A turn where the model fans out to ONE sync tool and ONE async
    tool simultaneously -- confirms record_call and record_call_async
    share correct, consistent turn-state bookkeeping (both read/write
    the same _last_turn_step / _last_turn_parent_ids / _current_turn_new_ids)."""
    def fake_model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name="check_watchlist", args={"applicant_id": "A-991"}),
                ToolCallPart(tool_name="verify_document", args={"applicant_id": "A-991"}),
            ])
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(FunctionModel(fake_model_fn))

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="check_watchlist")
    def check_watchlist(ctx: RunContext, applicant_id: str) -> dict:  # sync
        return {"applicant_id": applicant_id, "watchlist_match": False}

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="verify_document")
    async def verify_document(ctx: RunContext, applicant_id: str) -> dict:  # async
        await asyncio.sleep(0)
        return {"applicant_id": applicant_id, "document_verified": True}

    with sevenbox_agent_run() as krun:
        krun.session.start("Run KYC on A-991")
        agent.run_sync("Run KYC on A-991")

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    watchlist_w = next(w for w in warrants.values() if "watchlist_match" in w.content)
    document_w = next(w for w in warrants.values() if "document_verified" in w.content)

    assert watchlist_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids != (watchlist_w.id,)


def test_an_explicitly_passed_empty_store_is_not_silently_discarded():
    """Same regression class as the other two adapters' identical test
    -- `store or WarrantStore()` truthiness bug."""
    from warrant_core import WarrantStore
    from sevenbox_adapters.pydantic_ai_adapter import PydanticSession

    my_store = WarrantStore()
    session = PydanticSession(store=my_store)
    assert session.store is my_store


# --- Real bug found via adversarial cross-framework testing, now fixed ---

def test_a_fresh_run_never_matches_a_stale_closed_turn_from_a_prior_run():
    """Regression test for a genuine bug found by deliberately trying to
    break this adapter, not by normal usage: `run_step` alone is NOT a
    safe turn identity, because it resets to 1 at the start of every
    fresh `agent.run()`/`run_sync()` call (see
    test_run_step_behavior_matches_the_documented_assumption). A prior
    version of `PydanticSession._resolve_parent_ids` compared `run_step`
    alone, which meant: run 1 does a genuine 2-tool PARALLEL turn at
    run_step=1 (closing with frontier={a,b}); run 2, a completely
    separate, later `agent.run_sync()` call, has its OWN first tool call
    also at run_step=1 (coincidence, not relation) -- and the OLD code
    incorrectly matched run 2's call against run 1's ALREADY-CLOSED
    turn's stale frozen parent set (the axiom, frozen before run 1's
    batch even resolved), rather than either the current frontier {a,b}
    (correct, ambient cross-run chaining) or a fresh axiom. The fix
    (this file's current code) keys turn identity on (run_id, run_step)
    together -- run_id is unique per `agent.run()` call and was verified
    empirically to disambiguate exactly this case."""
    def fake_model_fn_1(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name="a", args={}),
                ToolCallPart(tool_name="b", args={}),
            ])
        return ModelResponse(parts=[TextPart(content="run1 done")])

    agent1 = Agent(FunctionModel(fake_model_fn_1))

    @agent1.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="a")
    def a(ctx: RunContext) -> str:
        return "result_a"

    @agent1.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="b")
    def b(ctx: RunContext) -> str:
        return "result_b"

    with sevenbox_agent_run() as krun:
        krun.session.start("session root")
        agent1.run_sync("run 1: parallel a+b")

        w_a = next(w for w in krun.session.store.all() if "result_a" in w.content)
        w_b = next(w for w in krun.session.store.all() if "result_b" in w.content)

        def fake_model_fn_2(messages, info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:  # run_step will ALSO be 1 here -- a fresh run
                return ModelResponse(parts=[ToolCallPart(tool_name="c", args={})])
            return ModelResponse(parts=[TextPart(content="run2 done")])

        agent2 = Agent(FunctionModel(fake_model_fn_2))

        @agent2.tool
        @sevenbox_tool(source_kind=SourceKind.TOOL, name="c")
        def c(ctx: RunContext) -> str:
            return "result_c"

        agent2.run_sync("run 2: single call c, run_step coincidentally also = 1")

        w_c = next(w for w in krun.session.store.all() if "result_c" in w.content)

    # THE FIX: w_c gets the CURRENT frontier {a, b} -- correct, whether
    # you read this as ambient cross-run chaining or simply "not stale
    # data from a closed, unrelated turn" -- never the axiom alone,
    # which is what the bug produced.
    assert set(w_c.parent_ids) == {w_a.id, w_b.id}


# --- Real bug #2: PydanticSession had no lock at all, despite pydantic-ai ---
# --- genuinely dispatching sync tool calls onto real OS threads (confirmed ---
# --- against the installed SDK's own source -- see module docstring's ---
# --- "Thread safety" section). Found by code review, not by normal usage; ---
# --- reproduced here deterministically via threading.Barrier, bypassing ---
# --- the SDK's own (narrow, timing-dependent) dispatch window entirely, ---
# --- so this test does not depend on scheduling luck the way a real ---
# --- end-to-end SDK run would. ---

class _NullLock:
    """Stand-in for threading.Lock that provides no mutual exclusion at
    all -- used only to prove the test driver below can actually detect
    the corruption it is meant to catch, per _corruption_is_detectable_
    without_the_lock. Never used against the real adapter."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _drive_concurrent_siblings(session, *, n_siblings: int, run_id: str, run_step: int):
    """Directly exercises PydanticSession._resolve_parent_ids and
    _finish_call from n_siblings real OS threads, synchronized with a
    Barrier so every thread attempts to enter the critical section at
    the same instant -- deliberately bypassing pydantic-ai's own tool
    dispatch (real, but narrower and timing-dependent) to isolate the
    exact hazard the adapter's own lock (or, for the no-op variant, lack
    thereof) is responsible for.

    Returns (frontier_after, parent_id_sets_seen, errors) so callers can
    check both that no sibling's contribution silently vanished from the
    frontier and that every sibling resolved the identical parent set.
    """
    from warrant_core import ReversibilityClass, create_warrant

    barrier = threading.Barrier(n_siblings)
    parent_id_sets_seen: list = []
    warrant_ids: list = []
    errors: list = []
    results_lock = threading.Lock()  # guards only this test's own result lists

    def sibling(i: int) -> None:
        try:
            barrier.wait(timeout=5)
            parent_ids = session._resolve_parent_ids(
                run_id, run_step, {"sibling": i}
            )
            w = create_warrant(
                content={"sibling_result": i},
                source_kind=SourceKind.DERIVED,
                epistemic=EpistemicClass.VERIFIED,
                reversibility=ReversibilityClass.PURE,
                parent_ids=tuple(parent_ids),
            )
            session._finish_call(w)
            with results_lock:
                parent_id_sets_seen.append(frozenset(parent_ids))
                warrant_ids.append(w.id)
        except Exception as e:  # noqa: BLE001 -- corruption may surface as any exception
            with results_lock:
                errors.append(e)

    threads = [threading.Thread(target=sibling, args=(i,)) for i in range(n_siblings)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    return set(session._frontier), parent_id_sets_seen, set(warrant_ids), errors


def test_concurrent_siblings_of_one_turn_never_corrupt_shared_frontier():
    """Deterministic regression test for the real race
    `PydanticSession._lock` closes: pydantic-ai genuinely dispatches sync
    tool calls onto real OS threads (`anyio.to_thread`, confirmed against
    the installed pydantic-ai==2.27.0 source -- see module docstring),
    so two siblings of one parallel turn can call `_resolve_parent_ids`/
    `_finish_call` concurrently, not just interleaved cooperatively.

    Uses threading.Barrier + an aggressive switch interval to force
    maximal contention deterministically, across many trials, rather
    than hoping a single real SDK run happens to hit the narrow race
    window (an earlier sleep-based attempt at reproducing this against
    the real SDK did not reliably reproduce corruption in 200 trials --
    not evidence the gap is safe, just evidence the window is narrow;
    see _corruption_is_detectable_without_the_lock below for proof this
    driver itself can and does catch it once locking is removed).
    """
    from sevenbox_adapters.pydantic_ai_adapter import PydanticSession

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for trial in range(200):
            session = PydanticSession()
            frontier, parent_sets, warrant_ids, errors = _drive_concurrent_siblings(
                session, n_siblings=8, run_id="run-R", run_step=1,
            )
            assert not errors, f"trial {trial}: unexpected exception(s): {errors}"
            assert len(parent_sets) == 8, f"trial {trial}: lost a sibling's result entirely"
            assert len(set(parent_sets)) == 1, (
                f"trial {trial}: siblings of the SAME turn resolved DIFFERENT "
                f"parent sets: {parent_sets}"
            )
            assert frontier == warrant_ids, (
                f"trial {trial}: final frontier {frontier} does not match the "
                f"full set of sibling warrant ids {warrant_ids} -- at least one "
                f"sibling's contribution was silently dropped"
            )
    finally:
        sys.setswitchinterval(old_interval)


def test_corruption_is_detectable_without_the_lock():
    """Proves test_concurrent_siblings_of_one_turn_never_corrupt_shared_
    frontier is a meaningful test -- not one that would pass regardless
    of whether PydanticSession actually synchronizes access -- by running
    the identical driver with session._lock swapped for a no-op and
    confirming it reliably observes the exact corruption class described
    in the module docstring (a lost sibling, or siblings of one turn
    disagreeing on their parent set) at least once across many trials.

    This never runs against the real adapter's own lock; it exists
    solely so a future reader trusts the test above actually tests
    something, per this project's own "verify, don't assume" discipline.
    """
    from sevenbox_adapters.pydantic_ai_adapter import PydanticSession

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    corruption_observed = False
    try:
        for _ in range(200):
            session = PydanticSession()
            session._lock = _NullLock()  # remove protection, deliberately, test-only
            frontier, parent_sets, warrant_ids, errors = _drive_concurrent_siblings(
                session, n_siblings=8, run_id="run-R", run_step=1,
            )
            if errors:
                corruption_observed = True
                break
            if len(set(parent_sets)) != 1:
                corruption_observed = True
                break
            if frontier != warrant_ids:
                corruption_observed = True
                break
    finally:
        sys.setswitchinterval(old_interval)

    assert corruption_observed, (
        "expected the unlocked driver to exhibit detectable corruption at "
        "least once across 200 trials -- if this fails, either CPython's "
        "scheduling changed enough to hide the race, or the driver itself "
        "is not exercising the hazard it claims to; either way this test's "
        "counterpart above would no longer be proven meaningful"
    )
