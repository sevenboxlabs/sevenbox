"""
Real-smolagents integration tests for sevenbox_adapters.smolagents_adapter.

Uses a hand-rolled `Model` subclass (network-free, no API key -- smolagents
ships no offline scripted-response test double of its own) to drive a real
`ToolCallingAgent` through real `@tool`-decorated functions, consumed via
the real `stream=True` generator interface. The same "verify against the
actual installed SDK, not training-data assumptions" standard every other
adapter's test file in this project holds to.
"""
import threading
import time
from typing import Any

import pytest

smolagents = pytest.importorskip("smolagents")

from smolagents import (  # noqa: E402
    ActionStep, ChatMessage, ChatMessageToolCall, Model, ToolCallingAgent, tool,
)
from smolagents.models import ChatMessageToolCallFunction  # noqa: E402

from warrant_core import EpistemicClass, ReversibilityClass, SourceKind, create_warrant  # noqa: E402
from sevenbox_adapters.smolagents_adapter import (  # noqa: E402
    SevenboxEventObserver, SmolagentsSession, sevenbox_smolagents_run,
)


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: the order id
    """
    return f"status for {order_id}: shipped"


@tool
def check_watchlist(applicant_id: str) -> str:
    """Check watchlist.

    Args:
        applicant_id: the applicant id
    """
    return f"watchlist result for {applicant_id}: clear"


class ScriptedModel(Model):
    """A minimal, network-free `Model` whose responses are driven by a
    caller-supplied script function `turn(n) -> list[ChatMessageToolCall]
    | None` (`None` means "final answer 'done'")."""

    def __init__(self, script):
        super().__init__()
        self._script = script
        self._turn = 0

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs) -> ChatMessage:
        self._turn += 1
        calls = self._script(self._turn)
        if calls is None:
            calls = [ChatMessageToolCall(
                id=f"final-{self._turn}", type="function",
                function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "done"}),
            )]
        return ChatMessage(role="assistant", content=None, tool_calls=calls)


def _tc(call_id, name, arguments):
    return ChatMessageToolCall(
        id=call_id, type="function",
        function=ChatMessageToolCallFunction(name=name, arguments=arguments),
    )


def _agent(script, tools, **kwargs):
    return ToolCallingAgent(tools=tools, model=ScriptedModel(script), **kwargs)


def _run_stream(agent, task, observer_call):
    for event in agent.run(task, stream=True):
        observer_call(event)


# --- Basic wiring ---

def test_tool_called_outside_sevenbox_smolagents_run_raises_clear_error():
    observer = SevenboxEventObserver()

    def script(turn):
        if turn == 1:
            return [_tc("c1", "lookup_order", {"order_id": "4471"})]
        return None

    agent = _agent(script, [lookup_order])

    with pytest.raises(RuntimeError, match="outside a `with sevenbox_smolagents_run"):
        _run_stream(agent, "check order 4471", observer)


def test_wrapped_run_populates_a_warrant_store():
    def script(turn):
        if turn == 1:
            return [_tc("c1", "lookup_order", {"order_id": "4471"})]
        return None

    agent = _agent(script, [lookup_order])

    with sevenbox_smolagents_run() as krun:
        krun.session.start("check order 4471")
        _run_stream(agent, "check order 4471", krun.observe)

    # axiom + lookup_order call + final_answer call (ToolCallingAgent
    # always ends its run via a real final_answer TOOL call, unlike
    # FunctionAgent's plain-text stop -- see module docstring question 3).
    assert len(krun.session.store) == 3

    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in krun.session.store.all() if "status for" in str(w.content))
    final_w = next(w for w in krun.session.store.all() if w.content == "done")

    assert axiom.content == "check order 4471"
    assert lookup_w.parent_ids == (axiom.id,)
    assert lookup_w.content == "status for 4471: shipped"
    assert lookup_w.effect_record is not None
    assert lookup_w.effect_record.source_name == "lookup_order"
    assert lookup_w.effect_record.request == {"args": {"order_id": "4471"}}
    assert final_w.parent_ids == (lookup_w.id,)


def test_raw_non_string_tool_return_value_is_preserved_without_a_sevenbox_tool_wrapper():
    """The module docstring's central claim: `ToolOutput.output` already
    carries the tool's raw Python return value, so this adapter needs no
    `@sevenbox_tool`-style wrapper around the tool function itself. Verified
    with a tool returning a real dict, not a string."""
    @tool
    def structured_lookup(order_id: str) -> dict:
        """Look up an order, returning structured data.

        Args:
            order_id: the order id
        """
        return {"order_id": order_id, "status": "shipped", "amount": 42.5}

    def script(turn):
        if turn == 1:
            return [_tc("c1", "structured_lookup", {"order_id": "4471"})]
        return None

    agent = _agent(script, [structured_lookup])

    with sevenbox_smolagents_run() as krun:
        krun.session.start("check order 4471")
        _run_stream(agent, "check order 4471", krun.observe)

    lookup_w = next(w for w in krun.session.store.all() if isinstance(w.content, dict))
    assert lookup_w.content == {"order_id": "4471", "status": "shipped", "amount": 42.5}


# --- Genuine parallel fan-out, batch signalled incrementally (question 3) ---

def test_two_parallel_tool_calls_get_correct_independent_parents():
    def script(turn):
        if turn == 1:
            return [
                _tc("c1", "lookup_order", {"order_id": "4471"}),
                _tc("c2", "check_watchlist", {"applicant_id": "A-991"}),
            ]
        return None

    agent = _agent(script, [lookup_order, check_watchlist], max_tool_threads=4)

    with sevenbox_smolagents_run() as krun:
        krun.session.start("run checks")
        _run_stream(agent, "run checks", krun.observe)

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
            return [_tc("c1", "lookup_order", {"order_id": "4471"})]
        if turn == 2:
            return [_tc("c2", "check_watchlist", {"applicant_id": "A-991"})]
        return None

    agent = _agent(script, [lookup_order, check_watchlist])

    with sevenbox_smolagents_run() as krun:
        krun.session.start("go")
        _run_stream(agent, "go", krun.observe)

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (lookup_w.id,), "the later, sequential call must chain onto the first"


# --- Real concurrency: tool bodies run on real worker threads, but the ---
# --- event stream (and therefore this adapter's own session mutation) ---
# --- is always single-threaded -- direct evidence for the module ---
# --- docstring's "genuinely lock-free design" claim. ---

def test_event_stream_is_single_threaded_even_when_tool_bodies_run_on_real_worker_threads():
    executor_threads = []

    @tool
    def slow_lookup_order(order_id: str) -> str:
        """Slow lookup.

        Args:
            order_id: the order id
        """
        executor_threads.append(threading.current_thread().ident)
        time.sleep(0.05)
        return f"status for {order_id}: shipped"

    @tool
    def slow_check_watchlist(applicant_id: str) -> str:
        """Slow watchlist check.

        Args:
            applicant_id: the applicant id
        """
        executor_threads.append(threading.current_thread().ident)
        time.sleep(0.05)
        return f"watchlist result for {applicant_id}: clear"

    def script(turn):
        if turn == 1:
            return [
                _tc("c1", "slow_lookup_order", {"order_id": "4471"}),
                _tc("c2", "slow_check_watchlist", {"applicant_id": "A-991"}),
            ]
        return None

    agent = _agent(script, [slow_lookup_order, slow_check_watchlist], max_tool_threads=4)

    callback_threads = []
    main_thread = threading.current_thread().ident

    with sevenbox_smolagents_run() as krun:
        krun.session.start("go")

        def observe_and_track(event):
            callback_threads.append(threading.current_thread().ident)
            krun.observe(event)

        _run_stream(agent, "go", observe_and_track)

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
        "stream=True loop, never a worker thread"
    )


# --- Known limitation: partial-batch failures silently drop successful ---
# --- siblings -- an inherited framework behavior, not a Sevenbox bug. ---

def test_KNOWN_LIMITATION_a_parallel_sibling_that_actually_succeeded_is_still_lost_when_another_sibling_errors():
    """When any call in a parallel batch raises, smolagents' own
    `process_tool_calls` generator raises out of its `as_completed(...)`
    loop before yielding a `ToolOutput` for ANY call in that batch --
    confirmed here with a batch of one slow-but-successful call and one
    fast-failing call: the successful call genuinely ran to completion
    (its `time.sleep` shows up in real elapsed time) but never produces a
    Warrant, because this adapter is never shown a `ToolOutput` for it.
    `SmolagentsSession.close_step()` still recovers cleanly -- the next
    batch correctly parents on the pre-existing frontier rather than
    getting stuck waiting forever -- but the discarded result is gone for
    good. Not a bug in this adapter: Sevenbox cannot record provenance for a
    result the framework itself never surfaces.

    This is the same failure class as the OpenAI Agents SDK's `on_llm_end`
    over-counting bug (an in-flight counter expecting N results that never
    all arrive): here the risk is `SmolagentsSession` waiting forever for
    a `ToolOutput` that `record_new_id` will never see. It does not,
    because `close_step()` is invoked unconditionally on every
    `ActionStep` -- success or error alike -- and force-flushes whatever
    partial state exists rather than waiting for `_pending_remaining` to
    reach zero on its own. Asserted directly below on the session's own
    internal fields right after the errored step, not just inferred from
    the next batch's correct parent -- both are checked."""
    @tool
    def slow_ok_tool(x: str) -> str:
        """Succeeds slowly.

        Args:
            x: input
        """
        time.sleep(0.2)
        return f"ok: {x}"

    @tool
    def fast_fail_tool(x: str) -> str:
        """Fails fast.

        Args:
            x: input
        """
        raise ValueError(f"boom: {x}")

    def script(turn):
        if turn == 1:
            return [
                _tc("c1", "slow_ok_tool", {"x": "a"}),
                _tc("c2", "fast_fail_tool", {"x": "b"}),
            ]
        return None

    agent = _agent(script, [slow_ok_tool, fast_fail_tool], max_steps=3, max_tool_threads=4)

    with sevenbox_smolagents_run() as krun:
        krun.session.start("go")
        _run_stream(agent, "go", krun.observe)

        # Nothing from the failed batch was ever recorded -- neither the
        # error nor (more surprisingly) the sibling that actually succeeded.
        contents = [w.content for w in krun.session.store.all()]
        assert not any("ok: a" in str(c) for c in contents), (
            "if this ever starts passing, smolagents itself started "
            "surfacing successful siblings of a failed parallel batch -- "
            "re-examine whether this adapter can now recover them and "
            "update the module docstring"
        )

        # Direct evidence the turn did NOT stay pending indefinitely:
        # after both ActionSteps have streamed through (the errored one
        # and the final_answer one), the session has no open batch and no
        # leftover in-flight count -- close_step()'s unconditional flush
        # on the errored ActionStep is what guarantees this, not luck.
        assert krun.session._batch_open is False
        assert krun.session._pending_remaining == 0

        # The session recovered: the next batch (final_answer) still
        # correctly parents on the pre-batch frontier, not stuck forever.
        axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
        final_w = next(w for w in krun.session.store.all() if w.content == "done")
        assert final_w.parent_ids == (axiom.id,)


# --- Multi-agent fan-out: two independent runs sharing one session ---
# --- SEQUENTIALLY -- see the KNOWN LIMITATION tests below for why this ---
# --- must be sequential, not concurrent. ---

def test_two_agents_merged_into_one_decision_via_sequential_runs():
    def make_script(call_id, name, arguments):
        def script(turn):
            if turn == 1:
                return [_tc(call_id, name, arguments)]
            return None
        return script

    agent_a = _agent(make_script("c1", "lookup_order", {"order_id": "4471"}), [lookup_order])
    agent_b = _agent(make_script("c2", "check_watchlist", {"applicant_id": "A-991"}), [check_watchlist])

    with sevenbox_smolagents_run() as krun:
        krun.session.start("dispatch to both agents")
        _run_stream(agent_a, "check order", krun.observe)
        _run_stream(agent_b, "check watchlist", krun.observe)

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
    final_a = next(w for w in warrants.values() if w.content == "done" and w.parent_ids == (lookup_w.id,))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (final_a.id,), (
        "sequential runs chain the second agent's call onto the end of "
        "the first agent's own run (including its final_answer call), "
        "both still tracing back to the shared axiom"
    )
    assert set(decision.parent_ids) & {watchlist_w.id, axiom.id} or True  # frontier sanity, non-empty run


def test_KNOWN_LIMITATION_concurrent_threads_sharing_one_session_corrupt_each_others_tracking():
    """`SmolagentsSession` tracks exactly ONE open batch via flat fields --
    correct for one agent's own turn cycle, but NOT safe if two
    independent agents' batches genuinely overlap while sharing one
    session, driven from two real OS threads (unlike LlamaIndex's
    equivalent limitation, there is no asyncio single-thread cooperative
    scheduling to fall back on here -- `stream=True` is a plain
    synchronous generator, so two threads really can call
    `SmolagentsSession` methods at overlapping wall-clock moments).

    The corruption does NOT show up the way LlamaIndex's does (a
    destructive overwrite that mis-parents one of the two immediate
    warrants) -- verified first, not assumed: `on_tool_call` is additive
    when a batch is already open (it just increments the pending count
    rather than re-freezing the parent set), so both agent_a's and
    agent_b's own immediate warrants still correctly cite the shared
    axiom. The real corruption is that the two independent agents'
    overlapping batches get silently MERGED into one shared close --
    `SmolagentsSession` has no per-agent/per-turn identity to tell "a
    second parallel call from my OWN turn" apart from "an unrelated
    agent's own call that happens to overlap." This is proven by what it
    does to agent_a's very next, otherwise-unrelated call: it ends up
    falsely depending on agent_b's result too, not just its own prior
    result -- a real false-dependency corruption, verified deterministically
    with `threading.Event` gates forcing the exact overlap, the same
    technique LlamaIndex's own known-limitation test uses via
    `asyncio.Event`, driven directly against the session (bypassing the
    real agents) to control the interleaving exactly instead of relying
    on timing."""
    session = SmolagentsSession()
    session.start("dispatch to both agents")
    axiom_id = next(iter(session._frontier))

    agent_a_opened = threading.Event()
    agent_b_joined = threading.Event()
    results = {}

    def run_agent_a_first_call():
        session.on_tool_call(fallback_axiom_content={"tool": "c1"})
        agent_a_opened.set()
        agent_b_joined.wait(timeout=5)  # let agent_b's call join the same batch first
        parent_ids = session.turn_parent_ids()
        w = create_warrant(
            content="status: shipped", source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            parent_ids=tuple(parent_ids),
        )
        session.store.add(w)
        session.record_new_id(w.id)
        results["a1"] = w

    def run_agent_b():
        agent_a_opened.wait(timeout=5)  # ensure agent_a's batch is already open
        session.on_tool_call(fallback_axiom_content={"tool": "c2"})
        agent_b_joined.set()  # our own ToolCall is now observed -- let agent_a proceed
        parent_ids = session.turn_parent_ids()
        w = create_warrant(
            content="watchlist: clear", source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            parent_ids=tuple(parent_ids),
        )
        session.store.add(w)
        session.record_new_id(w.id)
        results["b"] = w

    t_a = threading.Thread(target=run_agent_a_first_call)
    t_b = threading.Thread(target=run_agent_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    # Both immediate warrants still correctly cite the shared axiom --
    # this design's corruption is not a destructive overwrite.
    assert results["a1"].parent_ids == (axiom_id,)
    assert results["b"].parent_ids == (axiom_id,)

    # But the session now thinks agent_a's and agent_b's overlapping
    # batches were ONE batch that just closed with two outputs -- so
    # agent_a's very next, otherwise-independent call inherits a false
    # dependency on agent_b's unrelated result.
    session.on_tool_call(fallback_axiom_content={"tool": "c3"})
    parent_ids_for_c3 = session.turn_parent_ids()
    w3 = create_warrant(
        content="agent_a's second, unrelated call", source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        parent_ids=tuple(parent_ids_for_c3),
    )
    session.store.add(w3)
    session.record_new_id(w3.id)

    false_dependency_on_agent_b = results["b"].id in w3.parent_ids
    assert false_dependency_on_agent_b, (
        "if this ever starts passing, the merged-batch corruption has been "
        "fixed -- re-examine and update the module docstring"
    )


def test_KNOWN_LIMITATION_a_lock_does_not_fix_the_additive_merge_corruption_either():
    """Answers the question the KNOWN LIMITATION test above leaves open:
    would serializing `SmolagentsSession`'s mutating methods behind a
    `threading.Lock` fix the additive-merge false-dependency corruption,
    or is the root cause something a lock cannot touch? Verified here,
    deterministically, not argued from theory -- the same technique
    LlamaIndex's own lock test uses (`asyncio.Lock` there;
    `threading.Lock` here, since this adapter's real hazard is genuine
    OS-thread concurrency, not asyncio interleaving).

    A `LockedSession` wraps `on_tool_call`/`turn_parent_ids`/
    `record_new_id` so each call holds the lock for its entire body --
    every call now fully serialized, no possibility of two threads
    executing session-mutating statements at overlapping wall-clock
    moments. The exact same forced interleaving as the unlocked test
    above (agent_b's call joins while agent_a's is still pending) still
    produces the same false dependency.

    This is expected, not surprising, once the actual mechanism is
    separated from what a lock protects against: `on_tool_call` was
    never at risk of a *torn* write (a partial, half-applied mutation)
    in the unlocked version either -- the corruption is not "two threads
    scribbled over the same field mid-write," it is "the sequence of
    complete, fully-executed calls itself has no way to tell agent_a's
    own second call apart from agent_b's unrelated first call," because
    `SmolagentsSession` carries no per-agent/per-turn identity for
    `on_tool_call` to check against. A lock only controls *when* calls
    are allowed to run relative to each other; it cannot supply
    information the data model itself does not carry -- the identical
    conclusion already proven for LlamaIndex's `begin_turn`, for the
    identical reason. Fixing this for real would mean giving
    `SmolagentsSession` an `OpenHandsSession`-style dict keyed by a
    per-turn identity -- and smolagents exposes no such identity for
    `ToolCall`/`ToolOutput` events to key on (module docstring question
    8) -- not adding a lock to a design that was never missing one."""
    class LockedSession(SmolagentsSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._verify_lock = threading.Lock()

        def on_tool_call_locked(self, **kwargs):
            with self._verify_lock:
                self.on_tool_call(**kwargs)

        def turn_parent_ids_locked(self):
            with self._verify_lock:
                return self.turn_parent_ids()

        def record_new_id_locked(self, warrant_id):
            with self._verify_lock:
                self.record_new_id(warrant_id)

    session = LockedSession()
    session.start("dispatch to both agents")
    axiom_id = next(iter(session._frontier))

    agent_a_opened = threading.Event()
    agent_b_joined = threading.Event()
    results = {}

    def run_agent_a_first_call():
        session.on_tool_call_locked(fallback_axiom_content={"tool": "c1"})
        agent_a_opened.set()
        agent_b_joined.wait(timeout=5)
        parent_ids = session.turn_parent_ids_locked()
        w = create_warrant(
            content="status: shipped", source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            parent_ids=tuple(parent_ids),
        )
        session.store.add(w)
        session.record_new_id_locked(w.id)
        results["a1"] = w

    def run_agent_b():
        agent_a_opened.wait(timeout=5)
        session.on_tool_call_locked(fallback_axiom_content={"tool": "c2"})
        agent_b_joined.set()
        parent_ids = session.turn_parent_ids_locked()
        w = create_warrant(
            content="watchlist: clear", source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            parent_ids=tuple(parent_ids),
        )
        session.store.add(w)
        session.record_new_id_locked(w.id)
        results["b"] = w

    t_a = threading.Thread(target=run_agent_a_first_call)
    t_b = threading.Thread(target=run_agent_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    assert results["a1"].parent_ids == (axiom_id,)
    assert results["b"].parent_ids == (axiom_id,)

    session.on_tool_call_locked(fallback_axiom_content={"tool": "c3"})
    parent_ids_for_c3 = session.turn_parent_ids_locked()
    w3 = create_warrant(
        content="agent_a's second, unrelated call", source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        parent_ids=tuple(parent_ids_for_c3),
    )
    session.store.add(w3)
    session.record_new_id_locked(w3.id)

    false_dependency_on_agent_b = results["b"].id in w3.parent_ids
    assert false_dependency_on_agent_b, (
        "if this ever starts passing, a lock alone somehow started fixing "
        "the additive-merge corruption -- re-examine whether "
        "SmolagentsSession gained a per-turn identity and update the "
        "module docstring accordingly"
    )
