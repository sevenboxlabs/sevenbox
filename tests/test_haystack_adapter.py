"""
Real-haystack-ai integration tests for sevenbox_adapters.haystack_adapter.

Uses a hand-rolled `@component`-decorated `ChatGenerator` (network-free,
no API key -- Haystack ships no offline scripted-response test double of
its own) to drive a real `Agent` through real `@tool`-decorated functions
and its real dict-keyed hooks system. Real Haystack orchestration code
runs in-process in every test here, the same verification strength as
Strands, Google ADK, and the first ten adapters, not the weaker
contract-shape-only kind `letta_adapter.py` had to settle for.
"""
import threading
import time
from typing import Any

import pytest

haystack = pytest.importorskip("haystack")

from haystack import component  # noqa: E402
from haystack.components.agents import Agent  # noqa: E402
from haystack.components.agents.state.state import State  # noqa: E402
from haystack.dataclasses import ChatMessage, ToolCall, ToolCallResult  # noqa: E402
from haystack.tools import tool  # noqa: E402

from warrant_core import EpistemicClass, ReversibilityClass, SourceKind, create_warrant  # noqa: E402
from sevenbox_adapters.haystack_adapter import (  # noqa: E402
    HaystackSession, SevenboxEventObserver, sevenbox_haystack_run,
)


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"status for {order_id}: shipped"


@tool
def check_watchlist(applicant_id: str) -> str:
    """Check watchlist."""
    return f"watchlist result for {applicant_id}: clear"


@component
class ScriptedGenerator:
    """A minimal, network-free `ChatGenerator` whose responses are driven
    by a caller-supplied script function `turn(n) -> list[ToolCall] |
    None` (`None` means "final answer 'done'")."""

    def __init__(self, script):
        self._script = script
        self._turn = 0

    @component.output_types(replies=list)
    def run(self, messages, tools=None, **kwargs):
        self._turn += 1
        calls = self._script(self._turn)
        if calls is None:
            reply = ChatMessage.from_assistant(text="done")
        else:
            reply = ChatMessage.from_assistant(text=None, tool_calls=calls)
        return {"replies": [reply]}


def _agent(script, tools, **kwargs):
    return Agent(chat_generator=ScriptedGenerator(script), tools=tools, **kwargs)


def _run_agent(agent, message, hooks=None):
    kwargs = {"hooks": hooks} if hooks is not None else {}
    return agent.run(messages=[ChatMessage.from_user(message)], **kwargs)


# --- Basic wiring ---

def test_hook_invoked_outside_sevenbox_haystack_run_raises_clear_error():
    observer = SevenboxEventObserver()

    def script(turn):
        if turn == 1:
            return [ToolCall(tool_name="lookup_order", arguments={"order_id": "4471"}, id="c1")]
        return None

    agent = _agent(script, [lookup_order], hooks=observer.hooks)

    with pytest.raises(RuntimeError, match="outside a `with sevenbox_haystack_run"):
        _run_agent(agent, "check order 4471")


def test_wrapped_run_populates_a_warrant_store():
    def script(turn):
        if turn == 1:
            return [ToolCall(tool_name="lookup_order", arguments={"order_id": "4471"}, id="c1")]
        return None

    with sevenbox_haystack_run() as krun:
        krun.session.start("check order 4471")
        agent = _agent(script, [lookup_order], hooks=krun.observer.hooks)
        _run_agent(agent, "check order 4471")

    assert len(krun.session.store) == 2  # axiom + 1 tool call

    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)

    assert axiom.content == "check order 4471"
    assert tool_w.parent_ids == (axiom.id,)
    assert tool_w.content == "status for 4471: shipped"
    assert tool_w.effect_record is not None
    assert tool_w.effect_record.source_name == "lookup_order"
    assert tool_w.effect_record.request == {"args": {"order_id": "4471"}}


def test_raw_non_string_tool_return_value_is_preserved():
    @tool
    def structured_lookup(order_id: str) -> dict:
        """Look up an order, returning structured data."""
        return {"order_id": order_id, "status": "shipped", "amount": 42.5}

    def script(turn):
        if turn == 1:
            return [ToolCall(tool_name="structured_lookup", arguments={"order_id": "4471"}, id="c1")]
        return None

    with sevenbox_haystack_run() as krun:
        krun.session.start("check order 4471")
        agent = _agent(script, [structured_lookup], hooks=krun.observer.hooks)
        _run_agent(agent, "check order 4471")

    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    # Haystack stringifies structured tool returns into a JSON-ish text
    # ToolCallResult.result -- confirmed empirically, not assumed -- so
    # the Warrant content is that string representation.
    assert "4471" in str(tool_w.content) and "shipped" in str(tool_w.content)


# --- Genuine parallel fan-out, whole batch given upfront at BEFORE_TOOL ---

def test_two_parallel_tool_calls_get_correct_independent_parents():
    def script(turn):
        if turn == 1:
            return [
                ToolCall(tool_name="lookup_order", arguments={"order_id": "4471"}, id="c1"),
                ToolCall(tool_name="check_watchlist", arguments={"applicant_id": "A-991"}, id="c2"),
            ]
        return None

    with sevenbox_haystack_run() as krun:
        krun.session.start("run checks")
        agent = _agent(script, [lookup_order, check_watchlist], hooks=krun.observer.hooks)
        _run_agent(agent, "run checks")

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
            return [ToolCall(tool_name="lookup_order", arguments={"order_id": "4471"}, id="c1")]
        if turn == 2:
            return [ToolCall(tool_name="check_watchlist", arguments={"applicant_id": "A-991"}, id="c2")]
        return None

    with sevenbox_haystack_run() as krun:
        krun.session.start("go")
        agent = _agent(script, [lookup_order, check_watchlist], hooks=krun.observer.hooks)
        _run_agent(agent, "go")

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (lookup_w.id,), "the later, sequential call must chain onto the first"


# --- Real concurrency: tool bodies run on real worker threads by ---
# --- default (no opt-in config needed, unlike Google ADK), but hook ---
# --- callbacks (and therefore this adapter's own session mutation) ---
# --- always run on the main/orchestrating thread. ---

def test_tool_bodies_run_on_real_worker_threads_but_hook_callbacks_run_on_the_main_thread():
    executor_threads = []
    callback_threads = []

    @tool
    def slow_lookup_order(order_id: str) -> str:
        """Slow lookup."""
        executor_threads.append(threading.current_thread().ident)
        time.sleep(0.3)
        return f"status for {order_id}: shipped"

    @tool
    def slow_check_watchlist(applicant_id: str) -> str:
        """Slow watchlist check."""
        executor_threads.append(threading.current_thread().ident)
        time.sleep(0.3)
        return f"watchlist result for {applicant_id}: clear"

    def script(turn):
        if turn == 1:
            return [
                ToolCall(tool_name="slow_lookup_order", arguments={"order_id": "4471"}, id="c1"),
                ToolCall(tool_name="slow_check_watchlist", arguments={"applicant_id": "A-991"}, id="c2"),
            ]
        return None

    main_thread = threading.current_thread().ident

    with sevenbox_haystack_run() as krun:
        krun.session.start("go")

        class _TrackThreadHook:
            def run(self, state: State) -> None:
                callback_threads.append(threading.current_thread().ident)

        hooks = dict(krun.observer.hooks)
        hooks["after_tool"] = hooks["after_tool"] + [_TrackThreadHook()]

        agent = _agent(script, [slow_lookup_order, slow_check_watchlist], hooks=hooks)
        t0 = time.time()
        _run_agent(agent, "go")
        elapsed = time.time() - t0

    assert len(set(executor_threads)) == 2, (
        f"expected the two tool bodies to run on two DIFFERENT real worker "
        f"threads, got {executor_threads}"
    )
    assert main_thread not in executor_threads, "tool bodies must not run on the calling thread"
    # Two 0.3s sleeps: ~0.3s if genuinely parallel, ~0.6s if sequential --
    # a wide, deliberately generous margin (found the hard way in
    # strands_adapter.py's equivalent test: a narrower 0.2s/0.02s pair
    # was flaky under full-suite CPU contention even though genuinely
    # parallel; fixed there, and designed correctly here from the start).
    assert elapsed < 0.5, (
        f"expected roughly max()-timing (~0.3s, genuine parallel dispatch, "
        f"the Haystack default with no opt-in config needed) well under "
        f"the sequential sum (~0.6s), got {elapsed}s"
    )
    assert len(set(callback_threads)) == 1, (
        f"expected every hook callback to run on a single thread, got: {set(callback_threads)}"
    )
    assert callback_threads[0] == main_thread, (
        "expected every hook callback to run on the thread driving "
        "agent.run(), never a worker thread"
    )


# --- A genuinely positive finding, confirmed independently for a ---
# --- second framework (first found for Strands): partial batch ---
# --- failures do NOT lose the successful sibling's result. ---

def test_partial_batch_failure_still_preserves_the_successful_siblings_result():
    @tool
    def ok_tool(x: str) -> str:
        """Succeeds."""
        return f"ok: {x}"

    @tool
    def erroring_tool(x: str) -> str:
        """Fails."""
        raise ValueError(f"boom: {x}")

    def script(turn):
        if turn == 1:
            return [
                ToolCall(tool_name="ok_tool", arguments={"x": "a"}, id="c1"),
                ToolCall(tool_name="erroring_tool", arguments={"x": "b"}, id="c2"),
            ]
        return None

    with sevenbox_haystack_run() as krun:
        krun.session.start("go")
        agent = _agent(script, [ok_tool, erroring_tool], hooks=krun.observer.hooks)
        _run_agent(agent, "go")  # must not raise

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)

    ok_w = next((w for w in warrants.values() if "ok: a" in str(w.content)), None)
    assert ok_w is not None, (
        "the successful sibling's result must be preserved even though "
        "another call in the same parallel batch errored -- if this fails, "
        "Haystack's error-isolation behavior changed and this adapter now "
        "matches smolagents'/Google ADK's data-loss shape instead"
    )
    assert ok_w.parent_ids == (axiom.id,)

    error_w = next((w for w in warrants.values() if "boom: b" in str(w.content)), None)
    assert error_w is not None
    assert error_w.effect_record.response["error"] is True


# --- Known limitation: concurrent agents sharing one session ---
# --- (destructive overwrite, the same shape as LlamaIndex's ---
# --- begin_turn/Google ADK's/Strands' equivalents) -- driven directly ---
# --- against the session, bypassing the real Agent, to control the ---
# --- interleaving exactly instead of relying on timing. ---

def _fake_result(tool_name, args, call_id, value):
    call = ToolCall(tool_name=tool_name, arguments=args, id=call_id)
    return ToolCallResult(result=value, origin=call, error=False)


def test_KNOWN_LIMITATION_concurrent_agents_sharing_one_session_corrupt_each_others_tracking():
    """A first version of this test opened both agents' batches while
    neither had resolved yet -- it passed cleanly, i.e. the claimed
    corruption did NOT reproduce, because `begin_batch`'s overwrite is
    only destructive once `_frontier` has actually moved past the value
    it already held (here, both opens saw the same still-fresh axiom, so
    overwriting `_turn_parent_ids` with the same value again is a no-op
    in effect). Exactly the "assert -> test -> discovered false ->
    correct" discipline this project follows throughout -- caught before this
    landed in the module docstring, not after.

    The real corruption needs a THIRD step: agent_a must fully resolve
    (which resets `_turn_parent_ids` to `None` and moves `_frontier`
    forward) `while agent_b's batch is still sitting on its own, now
    stale, frozen parent set` -- so that when agent_b finally reads
    `turn_parent_ids()`, `_turn_parent_ids` is `None` again and it falls
    through to "no batch open, use current frontier" -- which is now
    agent_a's result, not the axiom agent_b's batch actually opened
    against."""
    session = HaystackSession()
    session.start("dispatch to both agents")
    axiom_id = next(iter(session._frontier))
    observer = SevenboxEventObserver()

    agent_a_opened = threading.Event()
    agent_b_opened = threading.Event()
    agent_a_resolved = threading.Event()
    results = {}

    def run_agent_a():
        session.begin_batch(0, fallback_axiom_content={"tool": "c1"})
        agent_a_opened.set()
        agent_b_opened.wait(timeout=5)  # let agent_b's batch open while agent_a's is still open
        result = _fake_result("lookup_order", {"order_id": "4471"}, "c1", "status: shipped")
        session.resolve_batch([result], observer._build_warrant)
        results["a"] = next(w for w in session.store.all() if w.content == "status: shipped")
        agent_a_resolved.set()

    def run_agent_b():
        agent_a_opened.wait(timeout=5)
        session.begin_batch(0, fallback_axiom_content={"tool": "c2"})
        agent_b_opened.set()
        agent_a_resolved.wait(timeout=5)  # let agent_a fully resolve BEFORE agent_b reads its own parent ids
        result = _fake_result("check_watchlist", {"applicant_id": "A-991"}, "c2", "watchlist: clear")
        session.resolve_batch([result], observer._build_warrant)
        results["b"] = next(w for w in session.store.all() if w.content == "watchlist: clear")

    t_a = threading.Thread(target=run_agent_a)
    t_b = threading.Thread(target=run_agent_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    warrant_a, warrant_b = results["a"], results["b"]
    assert warrant_a.parent_ids == (axiom_id,), "agent_a's own batch should resolve correctly regardless"
    assert warrant_b.parent_ids != (axiom_id,), (
        "if this ever starts passing, HaystackSession gained a per-turn "
        "identity that survives concurrent sharing -- re-examine and "
        "update the module docstring"
    )
    assert warrant_b.parent_ids == (warrant_a.id,), (
        "the known-limitation shape: agent_b's batch actually opened "
        "against the axiom, but by the time it resolves it picks up "
        "whatever the frontier currently is -- agent_a's result -- "
        "instead of what it actually froze at open time"
    )


def test_KNOWN_LIMITATION_a_lock_does_not_fix_the_concurrent_session_corruption_either():
    """Answers the question the KNOWN LIMITATION test above leaves open:
    would a `threading.Lock` around `begin_batch`/`turn_parent_ids` fix
    the corruption, or is the root cause something a lock cannot touch?
    Verified here, deterministically, not argued from theory -- the same
    technique `llamaindex_adapter.py`'s/`google_adk_adapter.py`'s/
    `strands_adapter.py`'s own lock tests use."""
    class LockedSession(HaystackSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._verify_lock = threading.Lock()

        def begin_batch_locked(self, n, fallback_axiom_content=None):
            with self._verify_lock:
                self.begin_batch(n, fallback_axiom_content)

        def resolve_batch_locked(self, results, builder):
            with self._verify_lock:
                self.resolve_batch(results, builder)

    session = LockedSession()
    session.start("dispatch to both agents")
    axiom_id = next(iter(session._frontier))
    observer = SevenboxEventObserver()

    agent_a_opened = threading.Event()
    agent_b_opened = threading.Event()
    agent_a_resolved = threading.Event()
    results = {}

    def run_agent_a():
        session.begin_batch_locked(0, fallback_axiom_content={"tool": "c1"})
        agent_a_opened.set()
        agent_b_opened.wait(timeout=5)
        result = _fake_result("lookup_order", {"order_id": "4471"}, "c1", "status: shipped")
        session.resolve_batch_locked([result], observer._build_warrant)
        results["a"] = next(w for w in session.store.all() if w.content == "status: shipped")
        agent_a_resolved.set()

    def run_agent_b():
        agent_a_opened.wait(timeout=5)
        session.begin_batch_locked(0, fallback_axiom_content={"tool": "c2"})
        agent_b_opened.set()
        agent_a_resolved.wait(timeout=5)
        result = _fake_result("check_watchlist", {"applicant_id": "A-991"}, "c2", "watchlist: clear")
        session.resolve_batch_locked([result], observer._build_warrant)
        results["b"] = next(w for w in session.store.all() if w.content == "watchlist: clear")

    t_a = threading.Thread(target=run_agent_a)
    t_b = threading.Thread(target=run_agent_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    warrant_a, warrant_b = results["a"], results["b"]
    corrupted = (
        warrant_a.parent_ids != (axiom_id,)
        or warrant_b.parent_ids != (axiom_id,)
        or warrant_a.parent_ids != warrant_b.parent_ids
    )
    assert corrupted, (
        "if this ever starts passing, a lock alone somehow started fixing "
        "the corruption -- re-examine whether HaystackSession gained a "
        "per-turn identity and update the module docstring accordingly"
    )
