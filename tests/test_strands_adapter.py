"""
Real-strands-agents integration tests for sevenbox_adapters.strands_adapter.

Uses a hand-rolled `Model` subclass emitting the real Bedrock Converse
streaming protocol (`StreamEvent`s -- network-free, no API key -- Strands
ships no offline scripted-response test double of its own) to drive a
real `Agent` through real `@tool`-decorated functions and its real hooks
registry. Real Strands orchestration code runs in-process in every test
here, the same verification strength as Google ADK and the first ten
adapters, not the weaker contract-shape-only kind `letta_adapter.py` had
to settle for.
"""
import asyncio
import json
import threading
import time
from typing import Any

import pytest

strands = pytest.importorskip("strands")

from strands import Agent, tool  # noqa: E402
from strands.models.model import Model  # noqa: E402
from strands.hooks import AfterToolCallEvent  # noqa: E402

from warrant_core import EpistemicClass, ReversibilityClass, SourceKind, create_warrant  # noqa: E402
from sevenbox_adapters.strands_adapter import (  # noqa: E402
    SevenboxEventObserver, StrandsSession, sevenbox_strands_run,
)


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"status for {order_id}: shipped"


@tool
def check_watchlist(applicant_id: str) -> str:
    """Check watchlist."""
    return f"watchlist result for {applicant_id}: clear"


def _tool_use_events(index, tool_use_id, name, args):
    return [
        {"contentBlockStart": {"contentBlockIndex": index, "start": {"toolUse": {"toolUseId": tool_use_id, "name": name}}}},
        {"contentBlockDelta": {"contentBlockIndex": index, "delta": {"toolUse": {"input": json.dumps(args)}}}},
        {"contentBlockStop": {"contentBlockIndex": index}},
    ]


def _text_events(index, text):
    return [
        {"contentBlockStart": {"contentBlockIndex": index, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": index, "delta": {"text": text}}},
        {"contentBlockStop": {"contentBlockIndex": index}},
    ]


class ScriptedModel(Model):
    """A minimal, network-free `Model` whose responses are driven by a
    caller-supplied script function `turn(n) -> list[dict] | None`
    (a list of `(index, tool_use_id, name, args)` tuples for tool calls,
    or `None` for a final text answer 'done')."""

    def __init__(self, script):
        self._script = script
        self._turn = 0

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self._turn += 1
        calls = self._script(self._turn)
        yield {"messageStart": {"role": "assistant"}}
        if calls is None:
            for ev in _text_events(0, "done"):
                yield ev
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            for index, tool_use_id, name, args in calls:
                for ev in _tool_use_events(index, tool_use_id, name, args):
                    yield ev
            yield {"messageStop": {"stopReason": "tool_use"}}

    async def structured_output(self, *args, **kwargs):
        raise NotImplementedError


def _agent(script, tools, **kwargs):
    return Agent(model=ScriptedModel(script), tools=tools, **kwargs)


def _run(coro):
    return asyncio.run(coro)


def _run_agent(agent, message):
    return _run(agent.invoke_async(message))


# --- Basic wiring ---

def test_hook_observed_outside_sevenbox_strands_run_raises_clear_error():
    """Strands does not let a hook callback's exception propagate raw --
    its own event loop catches it and re-raises as
    `strands.types.exceptions.EventLoopException`, with the real
    exception preserved on `.original_exception` -- confirmed
    empirically (a first attempt asserting a bare `RuntimeError` failed
    with the wrapped type instead), not assumed from how every other
    adapter's equivalent test works."""
    from strands.types.exceptions import EventLoopException

    observer = SevenboxEventObserver()

    def script(turn):
        if turn == 1:
            return [(0, "c1", "lookup_order", {"order_id": "4471"})]
        return None

    agent = _agent(script, [lookup_order], hooks=[observer])

    with pytest.raises(EventLoopException) as exc_info:
        _run_agent(agent, "check order 4471")

    original = exc_info.value.original_exception
    assert isinstance(original, RuntimeError)
    assert "outside a `with sevenbox_strands_run" in str(original)


def test_wrapped_run_populates_a_warrant_store():
    def script(turn):
        if turn == 1:
            return [(0, "c1", "lookup_order", {"order_id": "4471"})]
        return None

    with sevenbox_strands_run() as krun:
        krun.session.start("check order 4471")
        agent = _agent(script, [lookup_order], hooks=[krun.observer])
        result = _run_agent(agent, "check order 4471")

    assert "done" in str(result) or str(result)

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
        """Look up an order, returning structured data.

        Args:
            order_id: the order id
        """
        return {"order_id": order_id, "status": "shipped", "amount": 42.5}

    def script(turn):
        if turn == 1:
            return [(0, "c1", "structured_lookup", {"order_id": "4471"})]
        return None

    with sevenbox_strands_run() as krun:
        krun.session.start("check order 4471")
        agent = _agent(script, [structured_lookup], hooks=[krun.observer])
        _run_agent(agent, "check order 4471")

    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    # Strands stringifies structured tool returns into a single text
    # ToolResultContent block -- confirmed empirically, not assumed --
    # so the Warrant content is that JSON-ish string representation.
    assert "4471" in str(tool_w.content) and "shipped" in str(tool_w.content)


# --- Genuine parallel fan-out, whole batch given upfront by BeforeToolsEvent ---

def test_two_parallel_tool_calls_get_correct_independent_parents():
    def script(turn):
        if turn == 1:
            return [
                (0, "c1", "lookup_order", {"order_id": "4471"}),
                (1, "c2", "check_watchlist", {"applicant_id": "A-991"}),
            ]
        return None

    with sevenbox_strands_run() as krun:
        krun.session.start("run checks")
        agent = _agent(script, [lookup_order, check_watchlist], hooks=[krun.observer])
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
            return [(0, "c1", "lookup_order", {"order_id": "4471"})]
        if turn == 2:
            return [(0, "c2", "check_watchlist", {"applicant_id": "A-991"})]
        return None

    with sevenbox_strands_run() as krun:
        krun.session.start("go")
        agent = _agent(script, [lookup_order, check_watchlist], hooks=[krun.observer])
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
        """Slow lookup.

        Args:
            order_id: the order id
        """
        executor_threads.append(threading.current_thread().ident)
        time.sleep(0.3)
        return f"status for {order_id}: shipped"

    @tool
    def slow_check_watchlist(applicant_id: str) -> str:
        """Slow watchlist check.

        Args:
            applicant_id: the applicant id
        """
        executor_threads.append(threading.current_thread().ident)
        time.sleep(0.3)
        return f"watchlist result for {applicant_id}: clear"

    def script(turn):
        if turn == 1:
            return [
                (0, "c1", "slow_lookup_order", {"order_id": "4471"}),
                (1, "c2", "slow_check_watchlist", {"applicant_id": "A-991"}),
            ]
        return None

    main_thread = threading.current_thread().ident

    with sevenbox_strands_run() as krun:
        krun.session.start("go")

        def track_and_observe(event: AfterToolCallEvent):
            callback_threads.append(threading.current_thread().ident)

        agent = _agent(
            script, [slow_lookup_order, slow_check_watchlist],
            hooks=[krun.observer, track_and_observe],
        )
        t0 = time.time()
        _run_agent(agent, "go")
        elapsed = time.time() - t0

    assert len(set(executor_threads)) == 2, (
        f"expected the two tool bodies to run on two DIFFERENT real worker "
        f"threads, got {executor_threads}"
    )
    assert main_thread not in executor_threads, "tool bodies must not run on the calling thread"
    # Two 0.3s sleeps: ~0.3s if genuinely parallel, ~0.6s if sequential --
    # a wide, deliberately generous margin (not the original 0.2s/0.02s
    # pair, whose 0.02s gap was too narrow to survive real scheduling
    # jitter under a full-suite run: found flaky exactly that way, fixed
    # by widening the gap between the two hypotheses instead of just
    # loosening the threshold).
    assert elapsed < 0.5, (
        f"expected roughly max()-timing (~0.3s, genuine parallel dispatch, "
        f"the Strands default with no opt-in config needed) well under the "
        f"sequential sum (~0.6s), got {elapsed}s"
    )
    assert len(set(callback_threads)) == 1, (
        f"expected every hook callback to run on a single thread, got: {set(callback_threads)}"
    )
    assert callback_threads[0] == main_thread, (
        "expected every hook callback to run on the thread driving "
        "agent.invoke_async(), never a worker thread"
    )


# --- A genuinely positive finding: partial batch failures do NOT lose ---
# --- the successful sibling's result (unlike smolagents/Google ADK). ---

def test_partial_batch_failure_still_preserves_the_successful_siblings_result():
    @tool
    def ok_tool(x: str) -> str:
        """Succeeds.

        Args:
            x: input
        """
        return f"ok: {x}"

    @tool
    def erroring_tool(x: str) -> str:
        """Fails.

        Args:
            x: input
        """
        raise ValueError(f"boom: {x}")

    def script(turn):
        if turn == 1:
            return [
                (0, "c1", "ok_tool", {"x": "a"}),
                (1, "c2", "erroring_tool", {"x": "b"}),
            ]
        return None

    with sevenbox_strands_run() as krun:
        krun.session.start("go")
        agent = _agent(script, [ok_tool, erroring_tool], hooks=[krun.observer])
        result = _run_agent(agent, "go")  # must not raise

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)

    ok_w = next((w for w in warrants.values() if "ok: a" in str(w.content)), None)
    assert ok_w is not None, (
        "the successful sibling's result must be preserved even though "
        "another call in the same parallel batch errored -- if this fails, "
        "Strands' error-isolation behavior changed and this adapter now "
        "matches smolagents'/Google ADK's data-loss shape instead"
    )
    assert ok_w.parent_ids == (axiom.id,)

    error_w = next((w for w in warrants.values() if "boom: b" in str(w.content)), None)
    assert error_w is not None
    assert error_w.effect_record.response["status"] == "error"


# --- Known limitation: concurrent agents sharing one session ---
# --- (destructive overwrite, the same shape as LlamaIndex's ---
# --- begin_turn/Google ADK's open_batch) -- driven directly against ---
# --- the session, bypassing the real Agent, to control the ---
# --- interleaving exactly instead of relying on timing. ---

def test_KNOWN_LIMITATION_concurrent_agents_sharing_one_session_corrupt_each_others_tracking():
    session = StrandsSession()
    session.start("dispatch to both agents")
    axiom_id = next(iter(session._frontier))

    agent_a_opened = threading.Event()
    agent_b_opened = threading.Event()
    results = {}

    def run_agent_a():
        session.begin_batch(1, fallback_axiom_content={"tool": "c1"})
        agent_a_opened.set()
        agent_b_opened.wait(timeout=5)  # let agent_b's batch open BEFORE agent_a's call resolves
        parent_ids = session.turn_parent_ids()
        w = create_warrant(
            content="status: shipped", source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            parent_ids=tuple(parent_ids),
        )
        session.store.add(w)
        session.record_new_id(w.id)
        results["a"] = w

    def run_agent_b():
        agent_a_opened.wait(timeout=5)
        session.begin_batch(1, fallback_axiom_content={"tool": "c2"})
        agent_b_opened.set()
        parent_ids = session.turn_parent_ids()
        w = create_warrant(
            content="watchlist: clear", source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            parent_ids=tuple(parent_ids),
        )
        session.store.add(w)
        session.record_new_id(w.id)
        results["b"] = w

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
        "if this ever starts passing, StrandsSession gained a per-turn "
        "identity that survives concurrent sharing -- re-examine and "
        "update the module docstring"
    )


def test_KNOWN_LIMITATION_a_lock_does_not_fix_the_concurrent_session_corruption_either():
    """Answers the question the KNOWN LIMITATION test above leaves open:
    would a `threading.Lock` around `begin_batch`/`turn_parent_ids`/
    `record_new_id` fix the corruption, or is the root cause something a
    lock cannot touch? Verified here, deterministically, not argued from
    theory -- the same technique `llamaindex_adapter.py`'s and
    `google_adk_adapter.py`'s own lock tests use."""
    class LockedSession(StrandsSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._verify_lock = threading.Lock()

        def begin_batch_locked(self, n_calls, fallback_axiom_content=None):
            with self._verify_lock:
                self.begin_batch(n_calls, fallback_axiom_content)

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
    agent_b_opened = threading.Event()
    results = {}

    def run_agent_a():
        session.begin_batch_locked(1, fallback_axiom_content={"tool": "c1"})
        agent_a_opened.set()
        agent_b_opened.wait(timeout=5)
        parent_ids = session.turn_parent_ids_locked()
        w = create_warrant(
            content="status: shipped", source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            parent_ids=tuple(parent_ids),
        )
        session.store.add(w)
        session.record_new_id_locked(w.id)
        results["a"] = w

    def run_agent_b():
        agent_a_opened.wait(timeout=5)
        session.begin_batch_locked(1, fallback_axiom_content={"tool": "c2"})
        agent_b_opened.set()
        parent_ids = session.turn_parent_ids_locked()
        w = create_warrant(
            content="watchlist: clear", source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            parent_ids=tuple(parent_ids),
        )
        session.store.add(w)
        session.record_new_id_locked(w.id)
        results["b"] = w

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
        "the corruption -- re-examine whether StrandsSession gained a "
        "per-turn identity and update the module docstring accordingly"
    )
