import asyncio
import json

import pytest

agents = pytest.importorskip("agents")
from agents import Agent, Runner, RunConfig, function_tool  # noqa: E402
from agents.models.interface import Model, ModelProvider, ModelResponse  # noqa: E402
from agents.usage import Usage  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from warrant_core import EpistemicClass, SourceKind, compute_frontier  # noqa: E402
from sevenbox_adapters.openai_agents_adapter import sevenbox_agent_run, sevenbox_tool  # noqa: E402


# --- Assumption-pinning test: run this in isolation from the adapter's ---
# --- own code (no @sevenbox_tool, no sevenbox_agent_run) so a future SDK ---
# --- upgrade that changed tool dispatch order fails loudly HERE, not as ---
# --- a mysterious parent-attribution regression in the adapter's own ---
# --- tests. Mirrors tests/test_pydantic_ai_adapter.py's ---
# --- test_run_step_behavior_matches_the_documented_assumption, which ---
# --- pins the analogous fact for that SDK. ---

def test_dispatch_order_of_parallel_tool_calls_matches_the_documented_assumption():
    """sevenbox_adapters.openai_agents_adapter's module docstring, and
    test_parallel_tool_calls_without_hooks_manufacture_a_false_dependency
    specifically, both rely on an unstated fact about the installed
    openai-agents==0.19.4 SDK: when a turn requests two tool calls at
    once, call_1 always finishes before call_2 starts -- no thread pool
    for sync tools (unlike pydantic-ai, see
    tests/test_pydantic_ai_adapter.py's own assumption-pinning test),
    genuinely cooperative asyncio scheduling only. Pinned here, against
    bare openai-agents with no Sevenbox code involved at all, so a future
    SDK version that changed this (e.g. by moving sync tools onto a
    thread pool the way pydantic-ai already does) fails this test
    specifically, rather than silently invalidating the adapter's
    documented ambient-frontier fallback behavior."""
    events = []

    class _TwoCallModel(Model):
        def __init__(self):
            self._turn = 0

        async def get_response(self, system_instructions, input, model_settings, tools,
                                output_schema, handoffs, tracing, *, previous_response_id,
                                conversation_id, prompt) -> ModelResponse:
            self._turn += 1
            if self._turn == 1:
                call_1 = ResponseFunctionToolCall(
                    type="function_call", call_id="call_1", name="tool_a", arguments="{}",
                )
                call_2 = ResponseFunctionToolCall(
                    type="function_call", call_id="call_2", name="tool_b", arguments="{}",
                )
                return ModelResponse(output=[call_1, call_2], usage=Usage(), response_id="fake-1")
            msg = ResponseOutputMessage(
                id="fake-2", type="message", role="assistant", status="completed",
                content=[ResponseOutputText(type="output_text", text="done", annotations=[])],
            )
            return ModelResponse(output=[msg], usage=Usage(), response_id="fake-2")

        async def stream_response(self, *args, **kwargs):
            raise NotImplementedError("not exercised by this test")

    @function_tool
    def tool_a() -> str:
        events.append("a_start")
        events.append("a_end")
        return "a"

    @function_tool
    def tool_b() -> str:
        events.append("b_start")
        events.append("b_end")
        return "b"

    agent = Agent(name="Plain", instructions="Use both tools.", tools=[tool_a, tool_b])
    provider = _ScriptedModelProvider(_TwoCallModel())

    asyncio.run(Runner.run(agent, "go", run_config=RunConfig(model_provider=provider)))

    assert events == ["a_start", "a_end", "b_start", "b_end"], (
        f"expected call_1 to fully complete before call_2 starts (the "
        f"assumption sevenbox_adapters.openai_agents_adapter's documented "
        f"ambient-frontier fallback relies on), got: {events}"
    )


class _ScriptedModel(Model):
    """Turn 1: request a tool call. Turn 2: return a final text answer.
    Deterministic, no network -- the same "verify the real SDK's own
    tool-calling loop offline" approach the adapter's own docstring
    verification relies on."""

    def __init__(self):
        self._turn = 0

    async def get_response(self, system_instructions, input, model_settings, tools,
                            output_schema, handoffs, tracing, *, previous_response_id,
                            conversation_id, prompt) -> ModelResponse:
        self._turn += 1
        if self._turn == 1:
            call = ResponseFunctionToolCall(
                type="function_call", call_id="call_1", name="lookup_order",
                arguments=json.dumps({"order_id": "4471"}),
            )
            return ModelResponse(output=[call], usage=Usage(), response_id="fake-1")
        msg = ResponseOutputMessage(
            id="fake-2", type="message", role="assistant", status="completed",
            content=[ResponseOutputText(type="output_text", text="Your order has shipped.", annotations=[])],
        )
        return ModelResponse(output=[msg], usage=Usage(), response_id="fake-2")

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("not exercised by these tests")


class _ScriptedModelProvider(ModelProvider):
    def __init__(self, model):
        self._model = model

    def get_model(self, model_name):
        return self._model


def _build_agent(tool_fn):
    return Agent(name="SupportAgent", instructions="Help with order questions.", tools=[tool_fn])


def test_tool_called_outside_sevenbox_agent_run_raises_clear_error():
    @sevenbox_tool(source_kind=SourceKind.TOOL)
    def lookup_order(order_id: str) -> dict:
        return {"status": "shipped", "order_id": order_id}

    with pytest.raises(RuntimeError, match="outside a `with sevenbox_agent_run"):
        lookup_order("4471")


def test_wrapped_agent_populates_a_warrant_store():
    @function_tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="lookup_order")
    def lookup_order(order_id: str) -> dict:
        return {"status": "shipped", "order_id": order_id}

    agent = _build_agent(lookup_order)
    provider = _ScriptedModelProvider(_ScriptedModel())

    async def _run():
        with sevenbox_agent_run() as krun:
            krun.session.start({"message": "Where is order 4471?"})
            result = await Runner.run(agent, "Where is order 4471?", run_config=RunConfig(model_provider=provider))
        return result, krun

    result, krun = asyncio.run(_run())

    assert result.final_output == "Your order has shipped."
    # explicit axiom (the user's message) + 1 tool call = 2 warrants
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

    @function_tool
    @sevenbox_tool(source_kind=SourceKind.TOOL)
    def lookup_order(order_id: str) -> dict:
        call_log.append(order_id)
        return {"status": "shipped", "order_id": order_id}

    agent = _build_agent(lookup_order)

    async def _run_live():
        with sevenbox_agent_run() as krun:
            krun.session.start({"message": "Where is order 4471?"})
            await Runner.run(
                agent, "Where is order 4471?",
                run_config=RunConfig(model_provider=_ScriptedModelProvider(_ScriptedModel())),
            )
        return krun

    live_run = asyncio.run(_run_live())
    assert call_log == ["4471"]

    async def _run_replay():
        with sevenbox_agent_run(replay_from=live_run.session.store) as krun:
            krun.session.start({"message": "Where is order 4471?"})
            await Runner.run(
                agent, "Where is order 4471?",
                run_config=RunConfig(model_provider=_ScriptedModelProvider(_ScriptedModel())),
            )
        return krun

    replay_run = asyncio.run(_run_replay())

    assert call_log == ["4471"]  # unchanged -- the real tool body did not run again
    assert {w.id for w in replay_run.session.store.all()} == {w.id for w in live_run.session.store.all()}


# --- turn-aware fan-out: parallel tool calls in a single turn ---

class _ParallelToolCallModel(Model):
    """Turn 1: request TWO tool calls in the SAME response -- a genuine
    single-turn fan-out, unlike _ScriptedModel above which only ever
    requests one call per turn. This is the case the ambient-frontier
    model (hooks not passed) gets wrong, and SevenboxRunHooks (hooks=krun.hooks)
    fixes."""

    def __init__(self):
        self._turn = 0

    async def get_response(self, system_instructions, input, model_settings, tools,
                            output_schema, handoffs, tracing, *, previous_response_id,
                            conversation_id, prompt) -> ModelResponse:
        self._turn += 1
        if self._turn == 1:
            call_1 = ResponseFunctionToolCall(
                type="function_call", call_id="call_1", name="check_watchlist",
                arguments=json.dumps({"applicant_id": "A-991"}),
            )
            call_2 = ResponseFunctionToolCall(
                type="function_call", call_id="call_2", name="verify_document",
                arguments=json.dumps({"applicant_id": "A-991"}),
            )
            return ModelResponse(output=[call_1, call_2], usage=Usage(), response_id="fake-1")
        msg = ResponseOutputMessage(
            id="fake-2", type="message", role="assistant", status="completed",
            content=[ResponseOutputText(type="output_text", text="KYC complete.", annotations=[])],
        )
        return ModelResponse(output=[msg], usage=Usage(), response_id="fake-2")

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("not exercised by these tests")


def _build_kyc_agent():
    @function_tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="check_watchlist")
    def check_watchlist(applicant_id: str) -> dict:
        return {"applicant_id": applicant_id, "watchlist_match": False}

    @function_tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="verify_document")
    def verify_document(applicant_id: str) -> dict:
        return {"applicant_id": applicant_id, "document_verified": True}

    return Agent(name="KYCAgent", instructions="Run KYC checks.", tools=[check_watchlist, verify_document])


def test_parallel_tool_calls_without_hooks_manufacture_a_false_dependency():
    """Documents the pre-existing ambient-frontier behavior when hooks are
    NOT passed: still correct for sequential calls, but a real, provable
    false causal edge appears between two calls that are actually
    independent. This test pins that behavior down explicitly so a future
    change can't silently alter it without a test failing here."""
    agent = _build_kyc_agent()
    provider = _ScriptedModelProvider(_ParallelToolCallModel())

    async def _run():
        with sevenbox_agent_run() as krun:
            krun.session.start({"applicant_id": "A-991"})
            await Runner.run(agent, "Run KYC on A-991", run_config=RunConfig(model_provider=provider))
        return krun

    krun = asyncio.run(_run())
    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    watchlist_w = next(w for w in warrants.values() if "watchlist_match" in w.content)
    document_w = next(w for w in warrants.values() if "document_verified" in w.content)

    assert watchlist_w.parent_ids == (axiom.id,)
    # the false dependency: document_w ends up chained onto watchlist_w,
    # not onto the axiom both calls actually share
    assert document_w.parent_ids == (watchlist_w.id,)


def test_parallel_tool_calls_with_hooks_get_independent_parents():
    """The fix: passing hooks=krun.hooks gives both calls in the same
    turn the same, correct, shared parent -- no false dependency between
    genuinely independent evidence. Zero changes to the tool functions
    themselves versus the test above; the only difference is hooks=krun.hooks."""
    agent = _build_kyc_agent()
    provider = _ScriptedModelProvider(_ParallelToolCallModel())

    async def _run():
        with sevenbox_agent_run() as krun:
            krun.session.start({"applicant_id": "A-991"})
            await Runner.run(
                agent, "Run KYC on A-991",
                hooks=krun.hooks,
                run_config=RunConfig(model_provider=provider),
            )
        return krun

    krun = asyncio.run(_run())
    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    watchlist_w = next(w for w in warrants.values() if "watchlist_match" in w.content)
    document_w = next(w for w in warrants.values() if "document_verified" in w.content)

    assert watchlist_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids != (watchlist_w.id,)


def test_sequential_calls_are_unaffected_by_hooks_being_passed():
    """hooks=krun.hooks must be a no-op for the ordinary one-call-per-turn
    case -- begin_turn(n<=1) is defined to do nothing. Same scenario as
    test_wrapped_agent_populates_a_warrant_store, with hooks added."""
    @function_tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="lookup_order")
    def lookup_order(order_id: str) -> dict:
        return {"status": "shipped", "order_id": order_id}

    agent = _build_agent(lookup_order)
    provider = _ScriptedModelProvider(_ScriptedModel())

    async def _run():
        with sevenbox_agent_run() as krun:
            krun.session.start({"message": "Where is order 4471?"})
            result = await Runner.run(
                agent, "Where is order 4471?", hooks=krun.hooks,
                run_config=RunConfig(model_provider=provider),
            )
        return result, krun

    result, krun = asyncio.run(_run())
    assert result.final_output == "Your order has shipped."
    assert len(krun.session.store) == 2
    tool_warrant = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    assert len(tool_warrant.parent_ids) == 1


# --- Real bug: on_llm_end overcounted turns mixing @sevenbox_tool-wrapped ---
# --- and plain (unwrapped) tools -- found by code review, not normal ---
# --- usage, since every existing test above only ever gives an Agent ---
# --- fully-sevenbox-wrapped tools. ---

class _MixedToolCallModel(Model):
    """Turn 1: requests a call to a @sevenbox_tool-wrapped tool AND a call to
    a PLAIN (unwrapped) tool, in the same response -- the exact shape
    SevenboxRunHooks.on_llm_end's unfiltered call count got wrong: it counted
    both, but only the wrapped one ever reaches AgentSession.record_call
    to decrement _pending_remaining, so the old code left the turn
    permanently "pending" even though it completed successfully."""

    def __init__(self):
        self._turn = 0

    async def get_response(self, system_instructions, input, model_settings, tools,
                            output_schema, handoffs, tracing, *, previous_response_id,
                            conversation_id, prompt) -> ModelResponse:
        self._turn += 1
        if self._turn == 1:
            call_1 = ResponseFunctionToolCall(
                type="function_call", call_id="call_1", name="wrapped_tool",
                arguments=json.dumps({}),
            )
            call_2 = ResponseFunctionToolCall(
                type="function_call", call_id="call_2", name="plain_tool",
                arguments=json.dumps({}),
            )
            return ModelResponse(output=[call_1, call_2], usage=Usage(), response_id="fake-1")
        msg = ResponseOutputMessage(
            id="fake-2", type="message", role="assistant", status="completed",
            content=[ResponseOutputText(type="output_text", text="done.", annotations=[])],
        )
        return ModelResponse(output=[msg], usage=Usage(), response_id="fake-2")

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("not exercised by these tests")


def test_turn_mixing_sevenbox_and_plain_tools_still_closes_correctly():
    """Regression test: an Agent with one @sevenbox_tool-wrapped tool and one
    plain, unwrapped tool, both requested in the same turn. Before the
    fix, on_llm_end's unfiltered count set _pending_remaining=2, but only
    the wrapped tool's call ever decremented it (to 1, never 0) -- the
    turn never closed, and the NEXT begin_turn() would fold it in via the
    "previous turn never fully resolved" recovery path as if it had
    failed mid-flight, even though it had not. Confirms both that this
    turn resolves cleanly (parent is the axiom, not left pending) and
    that a subsequent, unrelated sequential call is unaffected."""
    @function_tool
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="wrapped_tool")
    def wrapped_tool() -> dict:
        return {"wrapped": True}

    @function_tool
    def plain_tool() -> dict:
        return {"plain": True}

    agent = Agent(name="MixedAgent", instructions="Use both tools.", tools=[wrapped_tool, plain_tool])
    provider = _ScriptedModelProvider(_MixedToolCallModel())

    async def _run():
        with sevenbox_agent_run() as krun:
            krun.session.start({"start": True})
            await Runner.run(
                agent, "go", hooks=krun.hooks,
                run_config=RunConfig(model_provider=provider),
            )
        return krun

    krun = asyncio.run(_run())
    session = krun.session

    # THE FIX: the turn closed cleanly -- no residual "pending" state left
    # over despite the plain tool never calling record_call.
    assert session._pending_parent_ids is None
    assert session._pending_remaining == 0

    warrants = {w.id: w for w in session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    wrapped_w = next(w for w in warrants.values() if w.content == {"wrapped": True})
    # Only one Warrant was ever recorded for this turn: the plain tool
    # produces no Warrant at all, by design (it was never wrapped).
    assert wrapped_w.parent_ids == (axiom.id,)
    assert len(session.store) == 2  # axiom + the one wrapped call

    # A second, ordinary sequential call after the mixed turn must behave
    # normally -- proving the old bug's "every future turn permanently
    # sees a stale pending count" failure mode is gone, not just that
    # THIS turn's own bookkeeping looks right in isolation.
    session.begin_turn(1)  # what SevenboxRunHooks would do for a lone call
    assert session._pending_parent_ids is None


# --- Previously untested code paths (found by code review): exercised ---
# --- directly here rather than left implicit. ---

def test_parallel_turn_before_any_explicit_start_shares_one_freshly_created_axiom():
    """AgentSession.record_call's "turn began before any axiom existed"
    branch (lines ~189-195) was untested: the very first LLM response
    already requests parallel calls, and krun.session.start(...) was
    never called explicitly. Confirms all siblings share the SAME
    lazily-created axiom -- not each independently creating (and
    colliding or diverging on) their own."""
    agent = _build_kyc_agent()
    provider = _ScriptedModelProvider(_ParallelToolCallModel())

    async def _run():
        with sevenbox_agent_run() as krun:
            # Deliberately no krun.session.start(...) call.
            await Runner.run(
                agent, "Run KYC on A-991",
                hooks=krun.hooks,
                run_config=RunConfig(model_provider=provider),
            )
        return krun

    krun = asyncio.run(_run())
    warrants = {w.id: w for w in krun.session.store.all()}
    axioms = [w for w in warrants.values() if w.source_kind == SourceKind.AXIOM]
    watchlist_w = next(w for w in warrants.values() if "watchlist_match" in w.content)
    document_w = next(w for w in warrants.values() if "document_verified" in w.content)

    assert len(axioms) == 1, f"expected exactly one lazily-created axiom, got {len(axioms)}"
    assert watchlist_w.parent_ids == (axioms[0].id,)
    assert document_w.parent_ids == (axioms[0].id,)


def test_begin_turn_fold_in_recovers_a_turn_that_never_fully_resolved():
    """AgentSession.begin_turn's "a previous turn never fully resolved"
    fold-in (lines ~161-165) was untested. Simulates a 2-call turn where
    only one call ever lands (e.g. the SDK swallowed a raised tool
    error and moved on with a single result) by opening a turn, letting
    exactly one call finish, then starting the NEXT turn directly --
    confirms the incomplete turn's one real Warrant is folded into the
    frontier (not lost) rather than requiring both calls to complete."""
    from sevenbox_adapters.openai_agents_adapter import AgentSession
    from warrant_core import ReversibilityClass

    session = AgentSession()
    session.start({"start": True})
    axiom_id = next(iter(session._frontier))

    session.begin_turn(2)  # opens a 2-call turn
    session.record_call(
        tool_name="call_1", source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        args=(), kwargs={}, real_fn=lambda: {"result": 1},
    )
    assert session._pending_remaining == 1  # one call still "pending" -- never arrives
    assert session._pending_parent_ids is not None

    w1_id = next(w.id for w in session.store.all() if w.content == {"result": 1})

    # The next turn starts before call_2 ever lands.
    session.begin_turn(1)

    assert session._pending_parent_ids is None
    # THE FIX's fold-in: call_1's Warrant is the frontier now, not lost --
    # this is session._frontier, the adapter's OWN bookkeeping of "current
    # causal tip(s) to attach the next call to." It is a distinct concept
    # from compute_frontier(store).live below: nothing here ever calls
    # supersede() on the axiom, so the axiom remains un-superseded and
    # stays in the frontier right alongside w1 --
    # confirmed here rather than assumed, since the two concepts share a
    # name but not a definition.
    assert session._frontier == {w1_id}

    frontier = session.frontier()
    assert {w.id for w in frontier.live} == {axiom_id, w1_id}
    assert frontier.conflicts == []


def test_begin_turn_directly_is_a_noop_for_zero_or_one_calls():
    """Unit-level check of AgentSession.begin_turn in isolation, no SDK
    round-trip needed."""
    from sevenbox_adapters.openai_agents_adapter import AgentSession

    session = AgentSession()
    session.start({"x": 1})
    frontier_before = set(session._frontier)
    session.begin_turn(0)
    assert session._pending_parent_ids is None
    session.begin_turn(1)
    assert session._pending_parent_ids is None
    assert session._frontier == frontier_before
