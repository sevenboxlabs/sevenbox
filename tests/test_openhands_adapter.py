"""
Real-openhands-sdk integration tests for sevenbox_adapters.openhands_adapter.

Uses `openhands.sdk.llm.llm.litellm_completion` monkeypatched at the name
it is imported under (`from litellm import completion as litellm_completion`,
patched where it is *used*, not where litellm defines it) to return real
`litellm.types.utils.ModelResponse` objects with real tool calls -- no
network, no API key. This drives a real `Agent`/`Conversation` through
real `ToolDefinition`/`ToolExecutor` subclasses, the same "verify against
the actual installed SDK, not training-data assumptions" standard every
other adapter's test file in this project holds to.
"""
import os
import sys
import threading

import pytest

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

pytest.importorskip("openhands.sdk")

import openhands.sdk.llm.llm as _llm_module  # noqa: E402
from litellm.types.utils import (  # noqa: E402
    ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse,
)
from openhands.sdk import (  # noqa: E402
    LLM, Agent, Conversation, Tool, ToolDefinition, register_tool,
)
from openhands.sdk.tool import ToolExecutor  # noqa: E402
from openhands.sdk.tool.schema import Action, Observation  # noqa: E402
from pydantic import Field  # noqa: E402

from warrant_core import SourceKind, EpistemicClass, ReversibilityClass, create_warrant  # noqa: E402
from sevenbox_adapters.openhands_adapter import (  # noqa: E402
    SevenboxConversationCallback, OpenHandsSession, sevenbox_openhands_run,
)


# --- Real tools, registered once at module import time (matching every ---
# --- other test file's "@agent.tool" registration pattern). ---

class LookupOrderAction(Action):
    order_id: str = Field(description="order id")


class LookupOrderObservation(Observation):
    status: str
    order_id: str

    @property
    def agent_observation(self):
        return [{"type": "text", "text": f"status={self.status}"}]


class LookupOrderExecutor(ToolExecutor):
    def __call__(self, action: LookupOrderAction, conversation=None) -> LookupOrderObservation:
        return LookupOrderObservation(status="shipped", order_id=action.order_id)


class LookupOrderTool(ToolDefinition[LookupOrderAction, LookupOrderObservation]):
    @classmethod
    def create(cls, conv_state=None, **params):
        return [cls(
            name="lookup_order", description="Look up an order",
            action_type=LookupOrderAction, observation_type=LookupOrderObservation,
            executor=LookupOrderExecutor(),
        )]


class CheckWatchlistAction(Action):
    applicant_id: str = Field(description="applicant id")


class CheckWatchlistObservation(Observation):
    applicant_id: str
    watchlist_match: bool

    @property
    def agent_observation(self):
        return [{"type": "text", "text": f"match={self.watchlist_match}"}]


class CheckWatchlistExecutor(ToolExecutor):
    def __call__(self, action: CheckWatchlistAction, conversation=None) -> CheckWatchlistObservation:
        return CheckWatchlistObservation(applicant_id=action.applicant_id, watchlist_match=False)


class CheckWatchlistTool(ToolDefinition[CheckWatchlistAction, CheckWatchlistObservation]):
    @classmethod
    def create(cls, conv_state=None, **params):
        return [cls(
            name="check_watchlist", description="Check watchlist",
            action_type=CheckWatchlistAction, observation_type=CheckWatchlistObservation,
            executor=CheckWatchlistExecutor(),
        )]


register_tool("lookup_order", LookupOrderTool)
register_tool("check_watchlist", CheckWatchlistTool)


def _tool_call(call_id, name, args_json):
    return ChatCompletionMessageToolCall(
        id=call_id, type="function", function=Function(name=name, arguments=args_json),
    )


def _response(response_id, *, tool_calls=None, text=None, finish_reason):
    msg = Message(role="assistant", content=text, tool_calls=tool_calls)
    return ModelResponse(
        id=response_id, created=0, model="fake", object="chat.completion",
        choices=[Choices(finish_reason=finish_reason, index=0, message=msg)],
    )


def _scripted(turns):
    """turns: list of callables (messages) -> ModelResponse."""
    state = {"i": 0}

    def fake_completion(**kwargs):
        i = state["i"]
        state["i"] += 1
        return turns[min(i, len(turns) - 1)](kwargs.get("messages", []))

    return fake_completion


def _agent():
    llm = LLM(model="fake/fake-model", usage_id="test")
    return Agent(llm=llm, tools=[Tool(name="lookup_order"), Tool(name="check_watchlist")])


def _run(turns, callback=None):
    old = _llm_module.litellm_completion
    _llm_module.litellm_completion = _scripted(turns)
    try:
        agent = _agent()
        events = []
        callbacks = [events.append]
        if callback is not None:
            callbacks.append(callback)
        conversation = Conversation(agent=agent, callbacks=callbacks, workspace=".", visualizer=None)
        conversation.send_message("go")
        conversation.run()
        return events
    finally:
        _llm_module.litellm_completion = old


# --- Basic wiring ---

def test_tool_called_outside_sevenbox_openhands_run_raises_clear_error():
    callback = SevenboxConversationCallback()
    turns = [
        lambda msgs: _response("r1", tool_calls=[_tool_call("c1", "lookup_order", '{"order_id": "4471"}')], finish_reason="tool_calls"),
        lambda msgs: _response("r2", text="done", finish_reason="stop"),
    ]
    with pytest.raises(RuntimeError, match="outside a `with sevenbox_openhands_run"):
        _run(turns, callback=callback)


def test_wrapped_run_populates_a_warrant_store():
    turns = [
        lambda msgs: _response("r1", tool_calls=[_tool_call("c1", "lookup_order", '{"order_id": "4471"}')], finish_reason="tool_calls"),
        lambda msgs: _response("r2", text="done", finish_reason="stop"),
    ]
    with sevenbox_openhands_run() as krun:
        krun.session.start("check order 4471")
        _run(turns, callback=krun.callback)

    assert len(krun.session.store) == 2  # axiom + 1 tool call
    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)

    assert axiom.content == "check order 4471"
    assert tool_w.parent_ids == (axiom.id,)
    assert tool_w.content["status"] == "shipped"
    assert tool_w.effect_record is not None
    assert tool_w.effect_record.source_name == "lookup_order"
    assert tool_w.effect_record.request["args"]["order_id"] == "4471"


# --- Genuine parallel fan-out, grouped by the EXACT llm_response_id signal ---

def test_two_parallel_tool_calls_get_correct_independent_parents():
    turns = [
        lambda msgs: _response(
            "r1",
            tool_calls=[
                _tool_call("c1", "lookup_order", '{"order_id": "4471"}'),
                _tool_call("c2", "check_watchlist", '{"applicant_id": "A-991"}'),
            ],
            finish_reason="tool_calls",
        ),
        lambda msgs: _response("r2", text="done", finish_reason="stop"),
    ]
    with sevenbox_openhands_run() as krun:
        krun.session.start("run checks")
        _run(turns, callback=krun.callback)

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if w.content and "status" in w.content)
    watchlist_w = next(w for w in warrants.values() if w.content and "watchlist_match" in w.content)

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids != (lookup_w.id,), "no false dependency between siblings"


def test_event_emission_is_single_threaded_even_when_tool_bodies_run_on_real_worker_threads():
    """Resolves, with direct evidence, exactly why `OpenHandsSession`
    carries a `threading.Lock`. `ParallelToolExecutor.execute_batch`
    (module docstring question 3) genuinely dispatches tool BODIES to a
    real `concurrent.futures.ThreadPoolExecutor` once `tool_concurrency_
    limit > 1` -- confirmed here by two tools recording their own
    executing thread, with a real overlap in wall-clock time via
    `time.sleep()`. But `_execute_action_event`'s own docstring in the
    installed SDK says plainly: "Events are NOT emitted here -- the
    caller is responsible for emitting them in order," and
    `_ActionBatch.emit()` is a plain sequential `for` loop, called once,
    from the thread that called `Conversation.run()`/`.send_message()`
    -- only *after* the `ThreadPoolExecutor` context has already joined
    every worker (`execute_batch`'s `with ThreadPoolExecutor(...) as
    executor:` block exits, which blocks until all submitted work
    finishes, before `future.result()` is ever read). So
    `SevenboxConversationCallback.__call__` -- and therefore `OpenHandsSession
    .on_action`/`resolve_pending`/`close_call` -- is never actually
    invoked concurrently within one Conversation's run, regardless of
    `tool_concurrency_limit`: this test proves it directly by recording
    the thread identity of every tool-body execution AND every callback
    invocation in the same real parallel run."""
    import time

    executor_threads = []
    callback_threads = []

    class SlowAction(Action):
        n: int

    class SlowObservation(Observation):
        n: int

        @property
        def agent_observation(self):
            return [{"type": "text", "text": f"n={self.n}"}]

    class SlowExecutor(ToolExecutor):
        def __call__(self, action, conversation=None):
            executor_threads.append(threading.current_thread().ident)
            time.sleep(0.05)
            return SlowObservation(n=action.n)

    class SlowTool(ToolDefinition[SlowAction, SlowObservation]):
        name = "slow_tool"

        @classmethod
        def create(cls, conv_state=None, **params):
            return [cls(
                description="slow", action_type=SlowAction, observation_type=SlowObservation,
                executor=SlowExecutor(),
            )]

    register_tool("slow_tool", SlowTool)

    def recording_callback(event):
        callback_threads.append(threading.current_thread().ident)

    turns = [
        lambda msgs: _response(
            "r1",
            tool_calls=[
                _tool_call("c1", "slow_tool", '{"n": 1}'),
                _tool_call("c2", "slow_tool", '{"n": 2}'),
            ],
            finish_reason="tool_calls",
        ),
        lambda msgs: _response("r2", text="done", finish_reason="stop"),
    ]
    old = _llm_module.litellm_completion
    _llm_module.litellm_completion = _scripted(turns)
    try:
        with sevenbox_openhands_run() as krun:
            krun.session.start("go")
            llm = LLM(model="fake/fake-model", usage_id="test3")
            agent = Agent(llm=llm, tools=[Tool(name="slow_tool")], tool_concurrency_limit=2)
            conversation = Conversation(
                agent=agent, callbacks=[krun.callback, recording_callback],
                workspace=".", visualizer=None,
            )
            conversation.send_message("go")
            main_thread_ident = threading.current_thread().ident
            conversation.run()
    finally:
        _llm_module.litellm_completion = old

    assert len(set(executor_threads)) == 2, (
        f"expected the two tool bodies to run on two DIFFERENT real worker "
        f"threads (proving real parallel dispatch happened), got {executor_threads}"
    )
    assert main_thread_ident not in executor_threads, "tool bodies must not run on the calling thread"
    assert len(set(callback_threads)) == 1, (
        f"expected every callback invocation (ActionEvent + ObservationEvent "
        f"x2) to happen on a single thread, got threads: {set(callback_threads)}"
    )
    assert callback_threads[0] == main_thread_ident, (
        "expected the callback to always run on the thread that called "
        "Conversation.run(), never on a ParallelToolExecutor worker thread"
    )


def test_sequential_calls_still_chain_correctly_when_not_parallel():
    turns = [
        lambda msgs: _response("r1", tool_calls=[_tool_call("c1", "lookup_order", '{"order_id": "4471"}')], finish_reason="tool_calls"),
        lambda msgs: _response("r2", tool_calls=[_tool_call("c2", "check_watchlist", '{"applicant_id": "A-991"}')], finish_reason="tool_calls"),
        lambda msgs: _response("r3", text="done", finish_reason="stop"),
    ]
    with sevenbox_openhands_run() as krun:
        krun.session.start("go")
        _run(turns, callback=krun.callback)

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if w.content and "status" in w.content)
    watchlist_w = next(w for w in warrants.values() if w.content and "watchlist_match" in w.content)

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (lookup_w.id,), "the later, sequential call must chain onto the first"


def test_agent_error_event_still_produces_a_warrant_and_closes_the_batch():
    """A tool call that errors (AgentErrorEvent, not ObservationEvent)
    must still resolve its pending entry and close the batch -- module
    docstring question 6."""

    class BrokenExecutor(ToolExecutor):
        def __call__(self, action, conversation=None):
            raise RuntimeError("boom")

    class BrokenTool(ToolDefinition[LookupOrderAction, LookupOrderObservation]):
        # ToolDefinition.name is a ClassVar auto-derived from the class
        # name by __init_subclass__ (stripping a trailing "Tool" ->
        # "_tool" suffix) UNLESS set explicitly in the class body --
        # passing name= to the constructor below does NOT override it,
        # since ClassVar fields are not pydantic model fields. Without
        # this explicit override, "BrokenTool" would auto-derive to
        # "broken" (removesuffix("_tool") strips too much), not
        # "broken_tool" -- a real, verified SDK detail, not a guess.
        name = "broken_tool"

        @classmethod
        def create(cls, conv_state=None, **params):
            return [cls(
                description="always fails",
                action_type=LookupOrderAction, observation_type=LookupOrderObservation,
                executor=BrokenExecutor(),
            )]

    register_tool("broken_tool", BrokenTool)

    turns = [
        lambda msgs: _response("r1", tool_calls=[_tool_call("c1", "broken_tool", '{"order_id": "4471"}')], finish_reason="tool_calls"),
        lambda msgs: _response("r2", text="done", finish_reason="stop"),
    ]
    old = _llm_module.litellm_completion
    _llm_module.litellm_completion = _scripted(turns)
    try:
        with sevenbox_openhands_run() as krun:
            krun.session.start("go")
            llm = LLM(model="fake/fake-model", usage_id="test2")
            agent = Agent(llm=llm, tools=[Tool(name="broken_tool")])
            conversation = Conversation(
                agent=agent, callbacks=[krun.callback], workspace=".", visualizer=None,
            )
            conversation.send_message("go")
            conversation.run()
    finally:
        _llm_module.litellm_completion = old

    assert len(krun.session.store) == 2  # axiom + 1 (failed) tool call
    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    assert tool_w.content is None
    assert tool_w.effect_record.response == {"error": tool_w.effect_record.response.get("error")}
    assert "boom" in tool_w.effect_record.response["error"]
    assert krun.session._open_in_batch == 0  # batch closed despite the error


# --- Deterministic concurrency: OpenHandsSession under real thread ---
# --- contention, reproduced via threading.Barrier -- same methodology ---
# --- already used for the PydanticAI/Agent Framework locking fixes. ---
# --- Unlike Agent Framework's session, grouping here is EXACT (shared ---
# --- llm_response_id), not a timing heuristic -- see module docstring ---
# --- question 3 -- so this test asserts full agreement with no known- ---
# --- limitation carve-out. ---


def _drive_concurrent_siblings(session: OpenHandsSession, *, n_siblings: int, llm_response_id: str):
    """Models the real SDK's own guaranteed ordering, confirmed by reading
    `openhands/sdk/agent/agent.py`'s `_execute_actions`/`_ActionBatch
    .prepare`: EVERY `ActionEvent` for one batch is created and its
    `on_event` callback fired -- synchronously, one after another, on a
    single thread -- before `ParallelToolExecutor.execute_batch` ever
    dispatches any of them to worker threads for execution. So `on_action`
    is always called for the WHOLE batch before `close_call` is ever
    called for ANY of it -- this driver opens all `n_siblings` actions
    first (single-threaded, matching that guarantee), then races the
    resolve/close phase across real threads (matching the real
    `ThreadPoolExecutor`-driven concurrent execution phase). An earlier
    version of this driver interleaved `on_action` and `close_call`
    within each thread instead, released together via one `Barrier` --
    that ordering doesn't reflect anything the real SDK can produce, and
    reliably (if the fast siblings' close_call ran before the slow ones'
    on_action) reopened a second, wrongly-separate batch -- a real bug in
    that TEST's modeling, not in `OpenHandsSession`, caught and fixed
    here rather than papered over.
    """
    for i in range(n_siblings):
        session.on_action(
            tool_call_id=f"c{i}", tool_name="sibling_tool", args={"i": i},
            llm_response_id=llm_response_id, fallback_axiom_content={"sibling": i},
        )

    barrier = threading.Barrier(n_siblings)
    parent_id_sets_seen: list = []
    warrant_ids: list = []
    errors: list = []
    results_lock = threading.Lock()

    def sibling(i: int) -> None:
        try:
            barrier.wait(timeout=5)
            tool_call_id = f"c{i}"
            parent_ids, tool_name, args = session.resolve_pending(tool_call_id, "sibling_tool")
            w = create_warrant(
                content={"sibling_result": i}, source_kind=SourceKind.DERIVED,
                epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
                parent_ids=tuple(parent_ids),
            )
            session.store.add(w)
            session.close_call(w.id)
            with results_lock:
                parent_id_sets_seen.append(frozenset(parent_ids))
                warrant_ids.append(w.id)
        except Exception as e:  # noqa: BLE001
            with results_lock:
                errors.append(e)

    threads = [threading.Thread(target=sibling, args=(i,)) for i in range(n_siblings)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    return set(session._frontier), parent_id_sets_seen, set(warrant_ids), errors


def test_concurrent_siblings_of_one_llm_response_always_share_the_same_parent_set():
    """Unlike Agent Framework's in-flight-counter heuristic, batch
    membership here is decided by an EXACT signal (the shared
    `llm_response_id` all siblings are called with) before any thread
    contention can matter -- so, unlike that adapter's equivalent test,
    this one asserts full agreement across all trials, no timing-
    dependent gap."""
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for trial in range(200):
            session = OpenHandsSession()
            frontier, parent_sets, warrant_ids, errors = _drive_concurrent_siblings(
                session, n_siblings=8, llm_response_id="resp-R",
            )
            assert not errors, f"trial {trial}: unexpected exception(s): {errors}"
            assert len(parent_sets) == 8, f"trial {trial}: lost a sibling's result entirely"
            assert len(set(parent_sets)) == 1, (
                f"trial {trial}: siblings sharing one llm_response_id resolved "
                f"DIFFERENT parent sets: {parent_sets}"
            )
            assert frontier == warrant_ids, (
                f"trial {trial}: final frontier {frontier} does not match the "
                f"full set of sibling warrant ids {warrant_ids}"
            )
    finally:
        sys.setswitchinterval(old_interval)


# A "without the lock, corruption is detectable" companion test (the
# pattern used for PydanticAI/Agent Framework) was attempted here too,
# swapping `session._lock` for `_NullLock` and re-running the same
# driver. It could NOT be made to reliably observe corruption -- 0/1000
# trials at 16 siblings and an aggressive `sys.setswitchinterval(1e-7)` --
# because the only genuinely non-atomic operation `close_call` performs
# without the lock is a single `self._open_in_batch -= 1` on a small int
# attribute, whose CPython bytecode sequence apparently does not land a
# GIL-release point inside the load/subtract/store window often enough
# to matter at this scale. This mirrors, more severely, the same "narrow
# window" caveat already recorded in test_pydantic_ai_adapter.py's
# equivalent driver ("not evidence the gap is safe, just evidence the
# window is narrow"). The lock is still correct and still kept -- a
# read-modify-write on shared mutable state is unsafe by construction
# regardless of how hard it is to force the failure empirically -- this
# note exists so a future reader does not go looking for a companion
# test that was tried, found unable to prove anything reliably here, and
# deliberately not kept rather than kept while silently not working.
