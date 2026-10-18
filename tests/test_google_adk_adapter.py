"""
Real-google-adk integration tests for sevenbox_adapters.google_adk_adapter.

Uses a hand-rolled `BaseLlm` subclass (network-free, no API key) to drive
a real `Runner`/`LlmAgent` through real Python tool functions. Unlike
`test_letta_adapter.py` (which could only mock the HTTP transport, never
running Letta's real server code), every test here drives real ADK
orchestration code in-process -- the same verification strength as the
first ten adapters in this package, stated explicitly, the same disclosure convention used throughout this package.
"""
import logging
import threading
from typing import Any

import pytest

google_adk = pytest.importorskip("google.adk")

from google.adk import Runner  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig  # noqa: E402
from google.adk.models import BaseLlm, LlmResponse  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from warrant_core import EpistemicClass, ReversibilityClass, SourceKind, create_warrant  # noqa: E402
from sevenbox_adapters.google_adk_adapter import (  # noqa: E402
    GoogleAdkSession, SevenboxEventObserver, sevenbox_google_adk_run,
)

logging.disable(logging.CRITICAL)  # ADK logs full tracebacks for tool errors -- silence for clean test output


def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"status for {order_id}: shipped"


def check_watchlist(applicant_id: str) -> str:
    """Check watchlist."""
    return f"watchlist result for {applicant_id}: clear"


class ScriptedLlm(BaseLlm):
    """A minimal, network-free `BaseLlm` whose responses are driven by a
    caller-supplied script function `turn(n) -> list[types.Part] | None`
    (`None` means "final answer 'done'")."""

    model: str = "scripted"
    script: Any = None

    async def generate_content_async(self, llm_request, stream: bool = False):
        parts = self.script(getattr(self, "_turn", 0) + 1)
        self._turn = getattr(self, "_turn", 0) + 1
        if parts is None:
            parts = [types.Part(text="done")]
        yield LlmResponse(content=types.Content(role="model", parts=parts), turn_complete=True)


def _call_part(call_id, name, args):
    return types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))


def _new_runner(script, tools):
    agent = LlmAgent(name="support_agent", model=ScriptedLlm(script=script), tools=tools)
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(app_name="test_app", user_id="user1")
    runner = Runner(app_name="test_app", agent=agent, session_service=session_service)
    return runner, session


def _run(runner, session, message="go", run_config=None):
    kwargs = {}
    if run_config is not None:
        kwargs["run_config"] = run_config
    return list(runner.run(
        user_id="user1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        **kwargs,
    ))


# --- Basic wiring ---

def test_event_observed_outside_sevenbox_google_adk_run_raises_clear_error():
    observer = SevenboxEventObserver()

    def script(turn):
        if turn == 1:
            return [_call_part("c1", "lookup_order", {"order_id": "4471"})]
        return None

    runner, session = _new_runner(script, [lookup_order])
    events = _run(runner, session)

    with pytest.raises(RuntimeError, match="outside a `with sevenbox_google_adk_run"):
        for e in events:
            observer.on_event(e)


def test_wrapped_run_populates_a_warrant_store():
    def script(turn):
        if turn == 1:
            return [_call_part("c1", "lookup_order", {"order_id": "4471"})]
        return None

    runner, session = _new_runner(script, [lookup_order])

    with sevenbox_google_adk_run() as krun:
        krun.session.start("check order 4471")
        for e in _run(runner, session, "check order 4471"):
            krun.observe(e)

    # axiom + 1 tool call -- the final text-only event produces no Warrant.
    assert len(krun.session.store) == 2

    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)

    assert axiom.content == "check order 4471"
    assert tool_w.parent_ids == (axiom.id,)
    assert tool_w.content == "status for 4471: shipped"
    assert tool_w.effect_record is not None
    assert tool_w.effect_record.source_name == "lookup_order"
    assert tool_w.effect_record.request == {"args": {"order_id": "4471"}}


def test_raw_non_string_tool_return_value_is_preserved():
    """ADK wraps a non-dict tool return in `{"result": <value>}` before
    this adapter ever sees it -- confirmed empirically, not assumed --
    so `_function_response_content` unwraps that specific convention.
    Verified here with a tool returning a real dict, which ADK leaves
    as-is (no extra wrapping, since it's already a dict)."""
    def structured_lookup(order_id: str) -> dict:
        return {"order_id": order_id, "status": "shipped", "amount": 42.5}

    def script(turn):
        if turn == 1:
            return [_call_part("c1", "structured_lookup", {"order_id": "4471"})]
        return None

    runner, session = _new_runner(script, [structured_lookup])

    with sevenbox_google_adk_run() as krun:
        krun.session.start("check order 4471")
        for e in _run(runner, session):
            krun.observe(e)

    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    assert tool_w.content == {"order_id": "4471", "status": "shipped", "amount": 42.5}


# --- Genuine parallel fan-out, whole batch bundled into one Event ---

def test_two_parallel_tool_calls_get_correct_independent_parents():
    def script(turn):
        if turn == 1:
            return [
                _call_part("c1", "lookup_order", {"order_id": "4471"}),
                _call_part("c2", "check_watchlist", {"applicant_id": "A-991"}),
            ]
        return None

    runner, session = _new_runner(script, [lookup_order, check_watchlist])

    with sevenbox_google_adk_run() as krun:
        krun.session.start("run checks")
        for e in _run(runner, session, "run checks"):
            krun.observe(e)

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
            return [_call_part("c1", "lookup_order", {"order_id": "4471"})]
        if turn == 2:
            return [_call_part("c2", "check_watchlist", {"applicant_id": "A-991"})]
        return None

    runner, session = _new_runner(script, [lookup_order, check_watchlist])

    with sevenbox_google_adk_run() as krun:
        krun.session.start("go")
        for e in _run(runner, session):
            krun.observe(e)

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (lookup_w.id,), "the later, sequential call must chain onto the first"


# --- Real concurrency: sync tools do NOT genuinely parallelize in the ---
# --- regular flow (module docstring question 3) -- confirmed here too. ---

def test_sync_tools_do_not_run_concurrently_in_the_regular_flow_even_with_thread_pool_config():
    import time

    executor_threads = []

    def slow_lookup_order(order_id: str) -> str:
        executor_threads.append(threading.current_thread().ident)
        time.sleep(0.2)
        return f"status for {order_id}: shipped"

    def fast_check_watchlist(applicant_id: str) -> str:
        executor_threads.append(threading.current_thread().ident)
        time.sleep(0.02)
        return f"watchlist result for {applicant_id}: clear"

    def script(turn):
        if turn == 1:
            return [
                _call_part("c1", "slow_lookup_order", {"order_id": "4471"}),
                _call_part("c2", "fast_check_watchlist", {"applicant_id": "A-991"}),
            ]
        return None

    runner, session = _new_runner(script, [slow_lookup_order, fast_check_watchlist])

    t0 = time.time()
    events = _run(
        runner, session, "go",
        run_config=RunConfig(tool_thread_pool_config=ToolThreadPoolConfig()),
    )
    elapsed = time.time() - t0

    assert len(set(executor_threads)) == 1, (
        f"expected both sync tool bodies to run on the SAME thread (no real "
        f"parallel dispatch in the regular flow, config or no config), got "
        f"{executor_threads}"
    )
    assert elapsed > 0.2 + 0.02 - 0.05, (
        f"expected roughly sequential timing (sum of both sleeps), got "
        f"{elapsed}s -- if this ever starts showing ~max() timing instead, "
        f"tool_thread_pool_config started applying to the regular flow -- "
        f"re-examine and update the module docstring"
    )


# --- Errors are real, recorded facts -- and a partial-batch-failure ---
# --- limitation matching smolagents' own discovery, for a related but ---
# --- distinct underlying reason (asyncio.gather without ---
# --- return_exceptions=True, not a framework-level discard decision). ---

def test_error_produces_no_warrant_and_does_not_crash_the_run():
    """This test may trigger a `PytestUnhandledThreadExceptionWarning`
    from ADK's own internal background thread (`_asyncio_thread_main`)
    re-raising the same tool exception during its own cleanup path --
    confirmed benign, not a sign of lost data, by the direct assertion
    below (see module docstring question 6's footnote)."""
    def erroring_tool(x: str) -> str:
        raise ValueError(f"boom: {x}")

    def script(turn):
        if turn == 1:
            return [_call_part("c1", "erroring_tool", {"x": "hi"})]
        return None

    runner, session = _new_runner(script, [erroring_tool])

    with sevenbox_google_adk_run() as krun:
        krun.session.start("go")
        events = _run(runner, session)  # must not raise
        for e in events:
            krun.observe(e)

    assert len(krun.session.store) == 1  # axiom only -- no Warrant for the failed call
    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    assert axiom.content == "go"


def test_KNOWN_LIMITATION_a_parallel_sibling_that_succeeded_is_still_lost_when_another_sibling_errors():
    """When one call in a parallel batch raises, `asyncio.gather(*tasks)`
    (no `return_exceptions=True`) discards whatever the OTHER gathered
    task already returned -- confirmed empirically: the successful
    sibling's real, computed result never appears in any `Event`. The
    identical failure shape smolagents has, for a related but distinct
    underlying reason (there, the framework's own generator raises
    before yielding any `ToolOutput` for the batch; here, `asyncio.
    gather`'s own semantics discard an already-available result)."""
    def ok_tool(x: str) -> str:
        return f"ok: {x}"

    def erroring_tool(x: str) -> str:
        raise ValueError(f"boom: {x}")

    def script(turn):
        if turn == 1:
            return [
                _call_part("c1", "ok_tool", {"x": "a"}),
                _call_part("c2", "erroring_tool", {"x": "b"}),
            ]
        return None

    runner, session = _new_runner(script, [ok_tool, erroring_tool])

    with sevenbox_google_adk_run() as krun:
        krun.session.start("go")
        for e in _run(runner, session):
            krun.observe(e)

    contents = [w.content for w in krun.session.store.all()]
    assert not any("ok: a" in str(c) for c in contents), (
        "if this ever starts passing, google-adk itself started surfacing "
        "successful siblings of a failed parallel batch -- re-examine and "
        "update the module docstring"
    )
    # Session recovered cleanly -- no leftover pending state.
    assert krun.session._pending_calls == {}
    assert krun.session._batch_parent_ids is None


# --- Known limitation: concurrent sessions sharing one store lose data ---
# --- (destructive overwrite, the same shape as LlamaIndex's begin_turn, ---
# --- not smolagents' additive merge) -- driven directly against the ---
# --- session, bypassing the real Runner, to control the interleaving ---
# --- exactly instead of relying on timing (same technique used by ---
# --- every other adapter's own known-limitation test in this package). ---

def test_KNOWN_LIMITATION_concurrent_sessions_sharing_one_store_lose_the_first_agents_results():
    session = GoogleAdkSession()
    session.start("dispatch to both agents")
    axiom_id = next(iter(session._frontier))
    observer = SevenboxEventObserver()

    agent_a_opened = threading.Event()
    agent_b_opened = threading.Event()
    results = {}

    def run_agent_a():
        call = types.FunctionCall(id="a-c1", name="lookup_order", args={"order_id": "4471"})
        session.open_batch([call], fallback_axiom_content={"tool": "a-c1"})
        agent_a_opened.set()
        agent_b_opened.wait(timeout=5)  # let agent_b's batch open BEFORE agent_a's response resolves
        response = types.FunctionResponse(id="a-c1", name="lookup_order", response={"result": "status for 4471: shipped"})
        session.resolve_batch([response], observer._build_warrant)
        results["a_warrants_before"] = len(session.store)

    def run_agent_b():
        agent_a_opened.wait(timeout=5)
        call = types.FunctionCall(id="b-c1", name="check_watchlist", args={"applicant_id": "A-991"})
        session.open_batch([call], fallback_axiom_content={"tool": "b-c1"})
        agent_b_opened.set()
        response = types.FunctionResponse(id="b-c1", name="check_watchlist", response={"result": "watchlist result for A-991: clear"})
        session.resolve_batch([response], observer._build_warrant)

    t_a = threading.Thread(target=run_agent_a)
    t_b = threading.Thread(target=run_agent_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    contents = [w.content for w in session.store.all()]
    lost_agent_a_result = not any("status for" in str(c) for c in contents)
    assert lost_agent_a_result, (
        "if this ever starts passing, GoogleAdkSession gained a per-turn "
        "identity that survives concurrent sharing -- re-examine and "
        "update the module docstring"
    )
    # agent_b's result, opened AFTER agent_a's, still resolves fine --
    # only the FIRST agent's in-flight call is silently discarded.
    assert any("watchlist result" in str(c) for c in contents)


def test_KNOWN_LIMITATION_a_lock_does_not_fix_the_concurrent_session_data_loss_either():
    """Answers the question the KNOWN LIMITATION test above leaves open:
    would a `threading.Lock` around `on_event` fix the data loss, or is
    the root cause something a lock cannot touch? Verified here,
    deterministically, not argued from theory -- the same technique
    `llamaindex_adapter.py`'s and `smolagents_adapter.py`'s own lock
    tests use."""
    class LockedSession(GoogleAdkSession):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._verify_lock = threading.Lock()

        def open_batch_locked(self, calls, fallback_axiom_content=None):
            with self._verify_lock:
                self.open_batch(calls, fallback_axiom_content)

        def resolve_batch_locked(self, responses, builder):
            with self._verify_lock:
                self.resolve_batch(responses, builder)

    session = LockedSession()
    session.start("dispatch to both agents")
    observer = SevenboxEventObserver()

    agent_a_opened = threading.Event()
    agent_b_opened = threading.Event()

    def run_agent_a():
        call = types.FunctionCall(id="a-c1", name="lookup_order", args={"order_id": "4471"})
        session.open_batch_locked([call], fallback_axiom_content={"tool": "a-c1"})
        agent_a_opened.set()
        agent_b_opened.wait(timeout=5)
        response = types.FunctionResponse(id="a-c1", name="lookup_order", response={"result": "status for 4471: shipped"})
        session.resolve_batch_locked([response], observer._build_warrant)

    def run_agent_b():
        agent_a_opened.wait(timeout=5)
        call = types.FunctionCall(id="b-c1", name="check_watchlist", args={"applicant_id": "A-991"})
        session.open_batch_locked([call], fallback_axiom_content={"tool": "b-c1"})
        agent_b_opened.set()
        response = types.FunctionResponse(id="b-c1", name="check_watchlist", response={"result": "watchlist result for A-991: clear"})
        session.resolve_batch_locked([response], observer._build_warrant)

    t_a = threading.Thread(target=run_agent_a)
    t_b = threading.Thread(target=run_agent_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    contents = [w.content for w in session.store.all()]
    lost_agent_a_result = not any("status for" in str(c) for c in contents)
    assert lost_agent_a_result, (
        "if this ever starts passing, a lock alone somehow started fixing "
        "the data loss -- re-examine whether GoogleAdkSession gained a "
        "per-turn identity and update the module docstring accordingly"
    )
