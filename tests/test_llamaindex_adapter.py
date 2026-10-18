"""
Real-llama-index-core integration tests for sevenbox_adapters.llamaindex_adapter.

Uses a hand-rolled `FunctionCallingLLM` subclass (network-free, no API
key) to drive a real `FunctionAgent` through real `FunctionTool`s. The
SDK's own offline test double, `MockLLM`, only produces plain text
completions and has no way to script tool calls, so it could not be used
here -- the same situation `openhands_adapter.py`'s tests hit with
`MockLanguageModelV2` needing `vitest`/`msw`. This is the same "verify
against the actual installed SDK, not training-data assumptions"
standard every other adapter's test file in this project holds to.
"""
import asyncio
import sys
import threading
import time
from typing import Any, List, Sequence

import pytest

llama_index_core = pytest.importorskip("llama_index.core")

from llama_index.core.agent.workflow import FunctionAgent  # noqa: E402
from llama_index.core.llms import ChatMessage, ChatResponse, LLMMetadata  # noqa: E402
from llama_index.core.llms.function_calling import FunctionCallingLLM  # noqa: E402
from llama_index.core.llms.llm import ToolSelection  # noqa: E402
from llama_index.core.tools import FunctionTool  # noqa: E402

from warrant_core import EpistemicClass, ReversibilityClass, SourceKind, create_warrant  # noqa: E402
from sevenbox_adapters.llamaindex_adapter import (  # noqa: E402
    SevenboxEventObserver, LlamaIndexSession, sevenbox_llamaindex_run,
)


def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"status for {order_id}: shipped"


def check_watchlist(applicant_id: str) -> str:
    """Check watchlist."""
    return f"watchlist result for {applicant_id}: clear"


class ScriptedLLM(FunctionCallingLLM):
    """A minimal, network-free `FunctionCallingLLM` whose responses are
    driven by a caller-supplied script function `turn(n) -> dict` --
    `{"tool_calls": [...]}` or `{"text": "..."}`. `raw` carries the
    script's tool-call dicts through untouched; `get_tool_calls_from_
    response` (which every concrete provider implements against its own
    wire format) reads them back out, giving this adapter full control
    over both sides without needing to fake a real provider's response
    shape.
    """

    turn: int = 0
    script: Any = None

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    def _next(self) -> ChatResponse:
        self.turn += 1
        step = self.script(self.turn)
        if "tool_calls" in step:
            return ChatResponse(
                message=ChatMessage(role="assistant", content=""),
                raw={"tool_calls": step["tool_calls"]},
            )
        return ChatResponse(
            message=ChatMessage(role="assistant", content=step.get("text", "")),
            raw={"tool_calls": []},
        )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._next()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._next()

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        raise NotImplementedError

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        raise NotImplementedError

    async def acomplete(self, prompt: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def astream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError

    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any,
    ) -> List[ToolSelection]:
        raw_calls = response.raw.get("tool_calls", []) if response.raw else []
        return [
            ToolSelection(tool_id=c["id"], tool_name=c["name"], tool_kwargs=c["kwargs"])
            for c in raw_calls
        ]

    def _prepare_chat_with_tools(
        self, tools, user_msg=None, chat_history=None, verbose=False,
        allow_parallel_tool_calls=False, **kwargs: Any,
    ):
        return {"messages": chat_history or []}


def _agent(script, tools):
    return FunctionAgent(tools=tools, llm=ScriptedLLM(script=script), streaming=False)


async def _consume(handler, observer_call):
    async for event in handler.stream_events():
        observer_call(event)
    return await handler


def _run(coro):
    return asyncio.run(coro)


def _run_agent(agent, message, observer_call):
    """`agent.run(...)` must be called from within a running event loop
    (the installed `workflows` runtime schedules a task synchronously
    inside `.run()`) -- this wraps both the call and the event-stream
    consumption in one coroutine so `asyncio.run()` provides that loop."""
    async def _go():
        handler = agent.run(message)
        return await _consume(handler, observer_call)

    return _run(_go())


# --- Basic wiring ---

def test_tool_called_outside_sevenbox_llamaindex_run_raises_clear_error():
    observer = SevenboxEventObserver()

    def script(turn):
        if turn == 1:
            return {"tool_calls": [{"id": "c1", "name": "lookup_order", "kwargs": {"order_id": "4471"}}]}
        return {"text": "done"}

    agent = _agent(script, [FunctionTool.from_defaults(fn=lookup_order)])

    with pytest.raises(RuntimeError, match="outside a `with sevenbox_llamaindex_run"):
        _run_agent(agent, "check order 4471", observer)


def test_wrapped_run_populates_a_warrant_store():
    def script(turn):
        if turn == 1:
            return {"tool_calls": [{"id": "c1", "name": "lookup_order", "kwargs": {"order_id": "4471"}}]}
        return {"text": "Your order has shipped."}

    agent = _agent(script, [FunctionTool.from_defaults(fn=lookup_order)])

    with sevenbox_llamaindex_run() as krun:
        krun.session.start("check order 4471")
        result = _run_agent(agent, "check order 4471", krun.observe)

    assert str(result) == "Your order has shipped." or "Your order has shipped." in str(result)
    assert len(krun.session.store) == 2  # axiom + 1 tool call

    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)

    assert axiom.content == "check order 4471"
    assert tool_w.parent_ids == (axiom.id,)
    assert tool_w.content == "status for 4471: shipped"
    assert tool_w.effect_record is not None
    assert tool_w.effect_record.source_name == "lookup_order"
    assert tool_w.effect_record.request == {"args": {"order_id": "4471"}}


# --- Genuine parallel fan-out, batch given upfront by AgentOutput.tool_calls ---

def test_two_parallel_tool_calls_get_correct_independent_parents():
    def script(turn):
        if turn == 1:
            return {"tool_calls": [
                {"id": "c1", "name": "lookup_order", "kwargs": {"order_id": "4471"}},
                {"id": "c2", "name": "check_watchlist", "kwargs": {"applicant_id": "A-991"}},
            ]}
        return {"text": "done"}

    agent = _agent(script, [
        FunctionTool.from_defaults(fn=lookup_order),
        FunctionTool.from_defaults(fn=check_watchlist),
    ])

    with sevenbox_llamaindex_run() as krun:
        krun.session.start("run checks")
        _run_agent(agent, "run checks", krun.observe)

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids != (lookup_w.id,), "no false dependency between siblings"


def test_sequential_calls_still_chain_correctly_when_not_parallel():
    def script(turn):
        if turn == 1:
            return {"tool_calls": [{"id": "c1", "name": "lookup_order", "kwargs": {"order_id": "4471"}}]}
        if turn == 2:
            return {"tool_calls": [{"id": "c2", "name": "check_watchlist", "kwargs": {"applicant_id": "A-991"}}]}
        return {"text": "done"}

    agent = _agent(script, [
        FunctionTool.from_defaults(fn=lookup_order),
        FunctionTool.from_defaults(fn=check_watchlist),
    ])

    with sevenbox_llamaindex_run() as krun:
        krun.session.start("go")
        _run_agent(agent, "go", krun.observe)

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (lookup_w.id,), "the later, sequential call must chain onto the first"


# --- Real concurrency: tool bodies run on real worker threads, but the ---
# --- event stream (and therefore this adapter's own session mutation) ---
# --- is always single-threaded -- this is the direct evidence behind ---
# --- the module docstring's "genuinely lock-free design" claim. ---

def test_event_stream_is_single_threaded_even_when_tool_bodies_run_on_real_worker_threads():
    executor_threads = []

    def slow_lookup_order(order_id: str) -> str:
        executor_threads.append(threading.current_thread().ident)
        time.sleep(0.05)
        return f"status for {order_id}: shipped"

    def slow_check_watchlist(applicant_id: str) -> str:
        executor_threads.append(threading.current_thread().ident)
        time.sleep(0.05)
        return f"watchlist result for {applicant_id}: clear"

    def script(turn):
        if turn == 1:
            return {"tool_calls": [
                {"id": "c1", "name": "slow_lookup_order", "kwargs": {"order_id": "4471"}},
                {"id": "c2", "name": "slow_check_watchlist", "kwargs": {"applicant_id": "A-991"}},
            ]}
        return {"text": "done"}

    agent = _agent(script, [
        FunctionTool.from_defaults(fn=slow_lookup_order),
        FunctionTool.from_defaults(fn=slow_check_watchlist),
    ])

    callback_threads = []

    async def _go():
        handler = agent.run("go")
        async for event in handler.stream_events():
            callback_threads.append(threading.current_thread().ident)
        return await handler

    with sevenbox_llamaindex_run() as krun:
        krun.session.start("go")
        main_thread = threading.current_thread().ident
        _run(_go())

    assert len(set(executor_threads)) == 2, (
        f"expected the two tool bodies to run on two DIFFERENT real worker "
        f"threads (proving real parallel dispatch happened), got {executor_threads}"
    )
    assert main_thread not in executor_threads, "tool bodies must not run on the calling thread"
    assert len(set(callback_threads)) == 1, (
        f"expected every event to be observed on a single thread, got: {set(callback_threads)}"
    )
    assert callback_threads[0] == main_thread, (
        "expected every event to be observed on the thread driving the "
        "stream_events() loop, never a worker thread"
    )


# --- Multi-agent fan-out: two independent runs sharing one session ---
# --- SEQUENTIALLY -- see the KNOWN LIMITATION test below for why this ---
# --- must be sequential, not concurrent, unlike Agent Framework's/ ---
# --- Mastra's equivalent multi-agent tests. ---

def test_two_agents_merged_into_one_decision_via_sequential_runs():
    """The `Agent -> Agent A / Agent B -> Decision` shape, done safely:
    two independent FunctionAgent runs, each consumed to completion
    *before* the next starts, sharing one LlamaIndexSession -- the same
    "one session, one turn open at a time" contract `mastra_adapter.py
    .import_mastra_log(store=shared)` uses for its own multi-log merge.
    See the KNOWN LIMITATION test below for what happens if this
    constraint is violated (two agents' turns genuinely overlapping via
    asyncio.gather), and the module docstring's "What this pass did not
    verify" for why this is sequential runs rather than a real
    AgentWorkflow handoff."""
    def make_script(tool_id, tool_name, kwargs):
        def script(turn):
            if turn == 1:
                return {"tool_calls": [{"id": tool_id, "name": tool_name, "kwargs": kwargs}]}
            return {"text": "done"}
        return script

    agent_a = _agent(
        make_script("c1", "lookup_order", {"order_id": "4471"}),
        [FunctionTool.from_defaults(fn=lookup_order)],
    )
    agent_b = _agent(
        make_script("c2", "check_watchlist", {"applicant_id": "A-991"}),
        [FunctionTool.from_defaults(fn=check_watchlist)],
    )

    with sevenbox_llamaindex_run() as krun:
        krun.session.start("dispatch to both agents")
        _run_agent(agent_a, "check order", krun.observe)
        _run_agent(agent_b, "check watchlist", krun.observe)

        decision = create_warrant(
            content={"decision": "proceed"}, source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            parent_ids=tuple(sorted(krun.session._frontier)),
        )
        krun.session.store.add(decision)

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (lookup_w.id,), (
        "sequential runs chain the second agent's call onto the first's "
        "result, same as any other sequential turn -- both still trace "
        "back to the shared axiom"
    )
    assert set(decision.parent_ids) == {watchlist_w.id}, (
        "the merge Decision cites whatever the session's frontier was "
        "after both sequential runs completed"
    )


def test_KNOWN_LIMITATION_concurrent_turns_sharing_one_session_corrupt_each_others_tracking():
    """`LlamaIndexSession` tracks exactly ONE open turn via flat fields
    (`_turn_parent_ids`/`_pending_remaining`/`_new_ids_this_turn`) --
    correct for one agent's own sequential-and/or-parallel turn cycle
    (every other test in this file), but NOT safe if two independent
    agents' turns genuinely overlap while sharing one session: unlike
    OpenHands' `llm_response_id`-keyed dict (question 3 in that
    adapter's docstring), there is no per-turn identity here to key a
    dict on, and LlamaIndex exposes no such id. Found adversarially by
    writing exactly the concurrent multi-agent test the other adapters
    in this package have (Agent Framework, Mastra) and watching it fail:
    `agent_b`'s `AgentOutput` (announcing ITS OWN one-tool-call turn)
    arrived while `agent_a`'s tool call was still pending, silently
    resetting the shared turn state agent_a's `record_new_id` was about
    to close against. This is why the test above runs its two agents
    SEQUENTIALLY rather than via `asyncio.gather` -- documented as a
    real, verified limitation, not a design that was never tried."""
    def make_script(tool_id, tool_name, kwargs):
        def script(turn):
            if turn == 1:
                return {"tool_calls": [{"id": tool_id, "name": tool_name, "kwargs": kwargs}]}
            return {"text": "done"}
        return script

    agent_a = _agent(
        make_script("c1", "lookup_order", {"order_id": "4471"}),
        [FunctionTool.from_defaults(fn=lookup_order)],
    )
    agent_b = _agent(
        make_script("c2", "check_watchlist", {"applicant_id": "A-991"}),
        [FunctionTool.from_defaults(fn=check_watchlist)],
    )

    async def _go():
        handler_a = agent_a.run("check order")
        handler_b = agent_b.run("check watchlist")
        await asyncio.gather(
            _consume(handler_a, krun.observe),
            _consume(handler_b, krun.observe),
        )

    with sevenbox_llamaindex_run() as krun:
        krun.session.start("dispatch to both agents")
        _run(_go())

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    # This IS the known limitation: at least one of the two siblings does
    # NOT correctly cite the shared axiom as its sole parent, because the
    # two agents' concurrently-interleaved turns corrupted each other's
    # flat-field tracking.
    mis_attributed = lookup_w.parent_ids != (axiom.id,) or watchlist_w.parent_ids != (axiom.id,)
    assert mis_attributed, (
        "if this ever starts passing, the known limitation has been "
        "closed -- update the module docstring and this test"
    )


def test_KNOWN_LIMITATION_a_lock_does_not_fix_the_concurrent_turn_corruption_either():
    """Answers the question the KNOWN LIMITATION test above leaves open:
    would serializing `LlamaIndexSession`'s methods behind a lock fix the
    corruption, or is the root cause something a lock cannot touch?
    Verified here, deterministically, not argued from theory: an
    `asyncio.Lock`-wrapped variant of the session's three mutating calls
    (`begin_turn`/`turn_parent_ids`/`record_new_id`), driven directly
    (bypassing the real FunctionAgent to control the exact interleaving
    deterministically instead of relying on timing), STILL produces the
    same corruption -- `agent_a`'s warrant ends up parented on `agent_b`'s
    warrant instead of the shared axiom -- even though every single call
    is now fully serialized, never executing concurrently with another.

    This is the confirmation that the known limitation is a real design
    constraint, not an unfixed race: a lock only prevents two calls from
    executing *at the same time*, which was never how this corruption
    happened (`LlamaIndexSession`'s methods contain no `await`, so they
    were already atomic with respect to each other even unlocked -- see
    module docstring's "What this does and does not buy"). The actual
    cause is that `begin_turn`, once it runs -- lock or no lock --
    unconditionally overwrites the session's single flat turn state with
    no way to know a *different*, still-open turn's bookkeeping is being
    discarded, because `LlamaIndexSession` has no per-turn identity to
    check against (unlike `OpenHandsSession`'s `llm_response_id`-keyed
    dict). Adding a lock changes only *when* the overwrite is allowed to
    happen, never *whether* it happens -- so this is documented as a real
    design limit (one session supports one open turn at a time, by
    design, the same assumption `sevenbox_llamaindex_run()` per logical run
    already makes), not left as an unresolved race condition."""
    import asyncio as _asyncio

    class LockedSession(LlamaIndexSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._verify_lock = _asyncio.Lock()

        async def begin_turn_locked(self, n_tool_calls, fallback_axiom_content=None):
            async with self._verify_lock:
                self.begin_turn(n_tool_calls, fallback_axiom_content)

        async def turn_parent_ids_locked(self):
            async with self._verify_lock:
                return self.turn_parent_ids()

        async def record_new_id_locked(self, warrant_id):
            async with self._verify_lock:
                self.record_new_id(warrant_id)

    async def simulate_agent(session, tool_id, content_str, resolve_gate):
        await session.begin_turn_locked(1, fallback_axiom_content={"tool": tool_id})
        await resolve_gate.wait()  # deterministic: only resolves once released below
        parent_ids = await session.turn_parent_ids_locked()
        w = create_warrant(
            content=content_str, source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            parent_ids=tuple(parent_ids),
        )
        session.store.add(w)
        await session.record_new_id_locked(w.id)
        return w

    async def go():
        session = LockedSession()
        session.start("dispatch to both agents")
        axiom_id = next(iter(session._frontier))

        gate_a = _asyncio.Event()  # agent_a's turn stays open until released
        gate_b = _asyncio.Event()
        gate_b.set()  # agent_b resolves immediately once its turn opens

        task_a = _asyncio.create_task(simulate_agent(session, "c1", "status: shipped", gate_a))
        await _asyncio.sleep(0)  # let agent_a's begin_turn_locked run and open its turn
        task_b = _asyncio.create_task(simulate_agent(session, "c2", "watchlist: clear", gate_b))
        await _asyncio.sleep(0)  # let agent_b's begin_turn_locked run WHILE agent_a's turn is still open
        gate_a.set()  # now let agent_a resolve

        warrant_a, warrant_b = await _asyncio.gather(task_a, task_b)
        return axiom_id, warrant_a, warrant_b

    axiom_id, warrant_a, warrant_b = _run(go())

    corrupted = warrant_a.parent_ids != (axiom_id,) or warrant_b.parent_ids != (axiom_id,)
    assert corrupted, (
        "if this ever starts passing, a lock alone somehow started fixing "
        "the corruption -- re-examine whether LlamaIndexSession gained a "
        "per-turn identity and update the module docstring accordingly"
    )
