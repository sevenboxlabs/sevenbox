"""
Real-letta-client integration tests for sevenbox_adapters.letta_adapter.

Letta's real boundary is an HTTP API, not an in-process agent loop, so
"network-free" here does not mean a hand-rolled model double the way it
does for every other adapter in this package -- it means injecting a
custom `httpx.Client` backed by `httpx.MockTransport` into the real,
installed `letta_client.Letta` client (`http_client=`, a real public
constructor parameter) and returning scripted, schema-accurate JSON.
Every response body below is parsed by the real, installed Pydantic
models (`LettaResponse`/`ToolCallMessage`/`ToolReturnMessage`/...) --
only the transport is faked, matching the same "verify against the
actual installed SDK" standard every other adapter's test file holds to.
"""
import json
import threading

import httpx
import pytest

letta_client = pytest.importorskip("letta_client")

from letta_client import Letta  # noqa: E402

from warrant_core import SourceKind, create_warrant, EpistemicClass, ReversibilityClass  # noqa: E402
from sevenbox_adapters.letta_adapter import (  # noqa: E402
    SevenboxEventObserver, LettaSession, sevenbox_letta_run,
)


def _usage():
    return {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2, "step_count": 1}


def _response_body(messages):
    return {
        "messages": messages,
        "stop_reason": {"stop_reason": "end_turn"},
        "usage": _usage(),
    }


def _tool_call_message(step_id, calls, msg_id="m-call"):
    """`calls`: list of (tool_call_id, name, arguments_dict)."""
    tool_calls = [
        {"tool_call_id": cid, "name": name, "arguments": json.dumps(args)}
        for cid, name, args in calls
    ]
    return {
        "id": msg_id, "date": "2026-01-01T00:00:00Z", "message_type": "tool_call_message",
        "step_id": step_id, "tool_call": tool_calls[0], "tool_calls": tool_calls,
    }


def _tool_return_message(step_id, returns, msg_id="m-return"):
    """`returns`: list of (tool_call_id, status, value)."""
    tool_returns = [
        {"tool_call_id": cid, "status": status, "tool_return": value}
        for cid, status, value in returns
    ]
    first = tool_returns[0]
    return {
        "id": msg_id, "date": "2026-01-01T00:00:01Z", "message_type": "tool_return_message",
        "step_id": step_id, "tool_call_id": first["tool_call_id"], "status": first["status"],
        "tool_return": first["tool_return"], "tool_returns": tool_returns,
    }


def _assistant_message(step_id, content, msg_id="m-assistant"):
    return {
        "id": msg_id, "date": "2026-01-01T00:00:02Z", "message_type": "assistant_message",
        "step_id": step_id, "content": content,
    }


def _mock_client(response_bodies):
    """Returns a real `Letta` client whose transport is faked, cycling
    through `response_bodies` (one per call to `messages.create`)."""
    bodies = list(response_bodies)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = min(calls["n"], len(bodies) - 1)
        calls["n"] += 1
        return httpx.Response(200, json=bodies[i])

    return Letta(base_url="http://mock.local", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


# --- Basic wiring ---

def test_tool_response_observed_outside_sevenbox_letta_run_raises_clear_error():
    observer = SevenboxEventObserver()
    client = _mock_client([_response_body([
        _tool_call_message("step-1", [("c1", "lookup_order", {"order_id": "4471"})]),
        _tool_return_message("step-1", [("c1", "success", "status for 4471: shipped")]),
    ])])

    response = client.agents.messages.create(agent_id="agent-1", input="check order 4471")

    with pytest.raises(RuntimeError, match="outside a `with sevenbox_letta_run"):
        observer.observe_response(response)


def test_wrapped_response_populates_a_warrant_store():
    client = _mock_client([_response_body([
        _tool_call_message("step-1", [("c1", "lookup_order", {"order_id": "4471"})]),
        _tool_return_message("step-1", [("c1", "success", "status for 4471: shipped")]),
        _assistant_message("step-2", "Your order has shipped."),
    ])])

    with sevenbox_letta_run() as krun:
        krun.session.start("check order 4471")
        response = client.agents.messages.create(agent_id="agent-1", input="check order 4471")
        krun.observe(response)

    # axiom + 1 tool call -- the AssistantMessage produces no Warrant
    # (matching llamaindex_adapter.py's stance on
    # plain-text turns).
    assert len(krun.session.store) == 2

    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)

    assert axiom.content == "check order 4471"
    assert tool_w.parent_ids == (axiom.id,)
    assert tool_w.content == "status for 4471: shipped"
    assert tool_w.effect_record is not None
    assert tool_w.effect_record.source_name == "lookup_order"
    assert tool_w.effect_record.request == {"args": {"order_id": "4471"}}


# --- Genuine parallel fan-out, whole batch delivered pre-resolved ---

def test_two_parallel_tool_calls_in_one_step_get_correct_independent_parents():
    client = _mock_client([_response_body([
        _tool_call_message("step-1", [
            ("c1", "lookup_order", {"order_id": "4471"}),
            ("c2", "check_watchlist", {"applicant_id": "A-991"}),
        ]),
        _tool_return_message("step-1", [
            ("c1", "success", "status for 4471: shipped"),
            ("c2", "success", "watchlist result for A-991: clear"),
        ]),
    ])])

    with sevenbox_letta_run() as krun:
        krun.session.start("run checks")
        response = client.agents.messages.create(agent_id="agent-1", input="run checks")
        krun.observe(response)

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids != (lookup_w.id,), "no false dependency between siblings"


def test_sequential_steps_within_one_response_chain_correctly():
    client = _mock_client([_response_body([
        _tool_call_message("step-1", [("c1", "lookup_order", {"order_id": "4471"})], msg_id="m1"),
        _tool_return_message("step-1", [("c1", "success", "status for 4471: shipped")], msg_id="m2"),
        _tool_call_message("step-2", [("c2", "check_watchlist", {"applicant_id": "A-991"})], msg_id="m3"),
        _tool_return_message("step-2", [("c2", "success", "watchlist result for A-991: clear")], msg_id="m4"),
    ])])

    with sevenbox_letta_run() as krun:
        krun.session.start("go")
        response = client.agents.messages.create(agent_id="agent-1", input="go")
        krun.observe(response)

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (lookup_w.id,), "the later step must chain onto the earlier step's result"


def test_sequential_calls_across_two_responses_still_chain():
    """The common real-world shape: one agent, two separate
    `messages.create()` calls (two conversation turns), sharing one
    session."""
    client = _mock_client([
        _response_body([
            _tool_call_message("step-1", [("c1", "lookup_order", {"order_id": "4471"})]),
            _tool_return_message("step-1", [("c1", "success", "status for 4471: shipped")]),
        ]),
        _response_body([
            _tool_call_message("step-1", [("c2", "check_watchlist", {"applicant_id": "A-991"})]),
            _tool_return_message("step-1", [("c2", "success", "watchlist result for A-991: clear")]),
        ]),
    ])

    with sevenbox_letta_run() as krun:
        krun.session.start("go")
        r1 = client.agents.messages.create(agent_id="agent-1", input="check order 4471")
        krun.observe(r1)
        r2 = client.agents.messages.create(agent_id="agent-1", input="now check watchlist")
        krun.observe(r2)

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (lookup_w.id,), "the second response must chain onto the first's result"


# --- Errors are real, recorded facts (Letta's own data model, verified) ---

def test_error_status_is_recorded_as_a_real_fact():
    client = _mock_client([_response_body([
        _tool_call_message("step-1", [("c1", "flaky_tool", {"x": "hi"})]),
        _tool_return_message("step-1", [("c1", "error", "ValueError: boom")]),
    ])])

    with sevenbox_letta_run() as krun:
        krun.session.start("go")
        response = client.agents.messages.create(agent_id="agent-1", input="go")
        krun.observe(response)

    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    assert tool_w.content == "ValueError: boom"
    assert tool_w.effect_record.response["status"] == "error"


def test_a_tool_call_with_no_matching_return_produces_no_warrant_and_does_not_corrupt_siblings():
    """Defensive correctness, not an observed framework bug (unlike
    smolagents' discarded-sibling limitation): `ToolReturn.status` is a
    real, non-optional field in Letta's own schema, so this should not
    happen with a real server -- but a hand-assembled or corrupted
    response should still degrade gracefully rather than crash or
    corrupt the sibling call's own correct attribution."""
    client = _mock_client([_response_body([
        _tool_call_message("step-1", [
            ("c1", "lookup_order", {"order_id": "4471"}),
            ("c2", "check_watchlist", {"applicant_id": "A-991"}),
        ]),
        _tool_return_message("step-1", [
            ("c1", "success", "status for 4471: shipped"),
            # c2 has no matching return.
        ]),
    ])])

    with sevenbox_letta_run() as krun:
        krun.session.start("run checks")
        response = client.agents.messages.create(agent_id="agent-1", input="run checks")
        krun.observe(response)

    contents = [w.content for w in krun.session.store.all()]
    assert any("status for" in str(c) for c in contents)
    assert not any("watchlist" in str(c) for c in contents)

    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in krun.session.store.all() if "status for" in str(w.content))
    assert lookup_w.parent_ids == (axiom.id,)


# --- Provenance-agnostic consumption -- NOT the same guarantee as ---
# --- ReplayEffectExecutor (see module docstring question 7): this ---
# --- shows observe_response() doesn't care where its input came from, ---
# --- not that a real call was skipped -- no real call ever runs in ---
# --- this file at all (see "What this pass did not verify"). ---

def test_observe_response_does_not_distinguish_a_live_response_from_a_stored_one():
    """`observe_response()` cannot tell a live response from a
    reconstructed one -- verified here by round-tripping through real
    JSON (`response.model_dump_json()` / re-parsing), then confirming
    the reconstructed Warrant has the IDENTICAL id to the live run's, via
    `compute_id()`'s own content-addressing, not a
    separate "looks the same" assertion. This is NOT the same claim as
    "replay avoids a real re-execution" (module docstring question 7) --
    no real HTTP call or real Letta server code runs anywhere in this
    test file, live or otherwise, so there is nothing here that could be
    said to have been "skipped." What this demonstrates is narrower: the
    consuming code has no live/replay branching in it, which is a real,
    useful property, just not the same one `ReplayEffectExecutor` gives
    the six fully-live adapters."""
    from letta_client.types.agents.letta_response import LettaResponse

    body = _response_body([
        _tool_call_message("step-1", [("c1", "lookup_order", {"order_id": "4471"})]),
        _tool_return_message("step-1", [("c1", "success", "status for 4471: shipped")]),
    ])
    client = _mock_client([body])

    with sevenbox_letta_run() as krun_live:
        krun_live.session.start("check order 4471")
        live_response = client.agents.messages.create(agent_id="agent-1", input="check order 4471")
        krun_live.observe(live_response)

    live_tool_w = next(w for w in krun_live.session.store.all() if w.source_kind == SourceKind.TOOL)

    # Reconstruct a LettaResponse purely from stored JSON -- no client,
    # no HTTP, no mock transport at all.
    stored_json = json.dumps(body)
    replayed_response = LettaResponse.model_validate_json(stored_json)

    with sevenbox_letta_run() as krun_replay:
        krun_replay.session.start("check order 4471")
        krun_replay.observe(replayed_response)

    replayed_tool_w = next(w for w in krun_replay.session.store.all() if w.source_kind == SourceKind.TOOL)

    assert replayed_tool_w.id == live_tool_w.id, (
        "replaying from stored JSON must produce a content-addressed id "
        "identical to the live run's -- if this fails, something about "
        "the replay path is producing different Warrant content than the "
        "live path did"
    )


# --- A lock that actually earns its keep (module docstring) ---

def _chained_response_body(prefix, n_steps):
    messages = []
    for i in range(n_steps):
        step_id = f"{prefix}-step-{i}"
        cid = f"{prefix}-c{i}"
        messages.append(_tool_call_message(step_id, [(cid, f"tool_{prefix}", {"i": i})], msg_id=f"{prefix}-m{i}a"))
        messages.append(_tool_return_message(step_id, [(cid, "success", f"{prefix}-result-{i}")], msg_id=f"{prefix}-m{i}b"))
    return _response_body(messages)


def _extract_response(body):
    from letta_client.types.agents.letta_response import LettaResponse
    return LettaResponse.model_validate(body)


def _step_index(content):
    """`"a-result-0"` -> `0`, `"a-result-11"` -> `11` -- numeric, not
    lexicographic, sort key (lexicographic would put "a-result-10"
    before "a-result-2")."""
    return int(str(content).rsplit("-", 1)[-1])


def _chain_is_self_consistent(chain):
    """True iff every warrant from the second onward parents on exactly
    the one immediately before it in this same chain -- checks EVERY
    link including the first (chain[1] against chain[0]), not just links
    from the second onward."""
    if len(chain) < 2:
        return True
    expected_prev = chain[0].id
    for w in chain[1:]:
        if w.parent_ids != (expected_prev,):
            return False
        expected_prev = w.id
    return True


def test_KNOWN_LIMITATION_unlocked_concurrent_responses_can_tear_the_frontier_mid_processing():
    """Demonstrates why `observe_response()` needs its lock: unlike every
    other adapter's session-mutating methods (short, single-purpose,
    effectively atomic even unlocked), this adapter's response processing
    loops over multiple steps, reading and rewriting `session._frontier`
    once per step -- so two threads sharing one UNLOCKED session can have
    their loop iterations interleave.

    A first version of this test relied on `sys.setswitchinterval` plus
    60-step responses to force this probabilistically -- it was flaky
    (passed "clean" on one real run, i.e. the claimed corruption did NOT
    reproduce that time), caught only by actually reading the result
    instead of trusting a single green run, exactly the
    "test before trusting a claim" discipline this project follows throughout. Replaced with a deterministic
    `threading.Event`-driven interleaving (the same technique used
    throughout this package's other concurrency tests), forcing the
    exact bad sequence rather than hoping for it -- and simpler than it
    first looks, because the corruption does not require catching a
    write torn mid-statement, only ordinary interleaving BETWEEN whole
    step iterations, which the unlocked design does nothing to prevent:

    1. Thread A completes its own step 0 in full (reads frontier=axiom,
       writes frontier={a0}).
    2. Thread B then completes its own step 0 in full (reads frontier --
       gets {a0}, since A already finished -- writes frontier={b0}). Not
       corruption on its own: B genuinely read a real value that existed
       at read-time (the general "sharing a session with another agent"
       caveat every adapter in this package documents, not this specific
       bug).
    3. Thread A proceeds to its OWN step 1: reads frontier -- but B's
       write in step 2 already replaced it, so A reads {b0}, not its own
       {a0} from step 1. A's step-1 warrant ends up parented on B's
       result instead of its own -- the torn chain, reproduced on every
       run, not most runs, because no fine-grained mid-statement timing
       is required at all."""
    session = LettaSession()
    session.start("dispatch to both chains")
    observer = SevenboxEventObserver()

    response_a = _extract_response(_chained_response_body("a", 2))  # a-step-0, a-step-1
    response_b = _extract_response(_chained_response_body("b", 1))  # b-step-0 only

    a_step0_done = threading.Event()
    b_step0_done = threading.Event()

    def run_a():
        messages = list(response_a.messages)
        step0_msgs, step1_msgs = messages[0:2], messages[2:4]
        observer._apply_response_locked(session, step0_msgs)  # bypasses the lock deliberately
        a_step0_done.set()
        b_step0_done.wait(timeout=5)  # let B's step 0 land in between
        observer._apply_response_locked(session, step1_msgs)

    def run_b():
        a_step0_done.wait(timeout=5)
        messages = list(response_b.messages)
        observer._apply_response_locked(session, messages)  # bypasses the lock deliberately
        b_step0_done.set()

    t_a = threading.Thread(target=run_a)
    t_b = threading.Thread(target=run_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    warrants = {w.id: w for w in session.store.all()}
    a_chain = sorted((w for w in warrants.values() if "a-result-" in str(w.content)), key=lambda w: _step_index(w.content))

    assert not _chain_is_self_consistent(a_chain), (
        "if this ever starts passing, the unlocked processing loop "
        "stopped being interleavable in this exact forced sequence -- "
        "re-examine before concluding the lock is unnecessary"
    )
    b_warrant = next(w for w in warrants.values() if "b-result-" in str(w.content))
    assert b_warrant.id in a_chain[-1].parent_ids, (
        "the torn chain should show A's own step-1 warrant citing B's "
        "warrant as a parent instead of A's own step-0 warrant"
    )


def test_KNOWN_LIMITATION_bare_threading_thread_loses_the_ambient_session():
    """Found adversarially while writing the concurrency-stress test
    below: a first attempt drove two `threading.Thread(target=krun.observe,
    args=(response,))` threads directly and got a silent background-
    thread `RuntimeError` from ONE of the two (swallowed by pytest as a
    `PytestUnhandledThreadExceptionWarning`, not a failure -- caught only
    because the resulting causal chain was suspiciously short, i.e. by
    checking the actual result, not by trusting the test run "green").
    Root cause: `_current_letta_session` (a `contextvars.ContextVar`) is
    set via `sevenbox_letta_run()` on the thread that opens the `with` block;
    a plain new `threading.Thread` does NOT inherit the creating thread's
    `contextvars.Context` -- confirmed as a general Python behavior, not
    specific to this adapter or to Letta. Every other adapter in this
    package never surfaced this because their real concurrency was always
    internal to the framework's own dispatch (`ThreadPoolExecutor.submit(
    copy_context().run, ...)`, verified case by case), never the calling
    application's own bare threads. Fixed in the test below by driving
    each thread via `contextvars.copy_context().run(...)` instead --
    the caller's responsibility, documented in the module docstring, not
    a bug in `LettaSession`/`SevenboxEventObserver` themselves."""
    with sevenbox_letta_run() as krun:
        krun.session.start("go")
        response = _extract_response(_response_body([
            _tool_call_message("step-1", [("c1", "lookup_order", {"order_id": "4471"})]),
            _tool_return_message("step-1", [("c1", "success", "status for 4471: shipped")]),
        ]))

        errors = []

        def run():
            try:
                krun.observe(response)
            except RuntimeError as e:
                errors.append(e)

        t = threading.Thread(target=run)
        t.start()
        t.join()

    assert len(errors) == 1, (
        "if this ever starts passing with zero errors, bare threading.Thread "
        "started inheriting contextvars.Context in some Python version -- "
        "re-examine and update the module docstring / the workaround in "
        "the stress tests below"
    )
    assert "outside a `with sevenbox_letta_run" in str(errors[0])


def test_locked_concurrent_responses_do_not_tear_the_frontier():
    """Confirms the lock genuinely fixes the torn-chain corruption, not
    merely fails to make it worse -- and, unlike the KNOWN LIMITATION
    test above, needs no artificial interleaving to prove it: `observe()`
    holds `session._lock` for an entire response's processing (every step
    of it), so two threads each calling `observe()` with their own
    multi-step response literally cannot have their steps interleave at
    all once locked -- true mutual exclusion, not a probabilistic
    improvement. Regular thread scheduling (no `sys.setswitchinterval`
    tuning) is enough to demonstrate this deterministically.

    Each thread runs via `contextvars.copy_context().run(...)` rather
    than a bare `threading.Thread(target=...)` -- required here, and
    found adversarially, not assumed: a first attempt using a bare
    `threading.Thread` raised `RuntimeError: ... outside a
    "with sevenbox_letta_run():" block` inside the child thread (silently
    swallowed by pytest as a background-thread warning, not a test
    failure -- caught only because the resulting chain was suspiciously
    short). `ContextVar.set()` on the main thread does not propagate to
    a plain new OS thread by default; every other adapter in this
    package never hit this because the thread dispatch was always
    internal to the framework itself (`ThreadPoolExecutor.submit(
    copy_context().run, ...)`, verified for each). Here the *calling
    application* controls the threading, not this adapter or Letta's own
    (server-side) tool execution -- so propagating context is the
    caller's responsibility, the same way it would be for any
    `ContextVar`-based design. Also found adversarially: a `Context`
    object cannot be entered concurrently, even from different threads
    (`RuntimeError: cannot enter context: ... is already entered`) -- a
    first attempt reused ONE `copy_context()` result for both threads;
    each needs its own. See
    `test_KNOWN_LIMITATION_bare_threading_thread_loses_the_ambient_session`
    below for the propagation failure this works around."""
    import contextvars

    response_a = _extract_response(_chained_response_body("a", 20))
    response_b = _extract_response(_chained_response_body("b", 20))

    with sevenbox_letta_run() as krun:
        krun.session.start("dispatch to both chains")

        def run(response):
            krun.observe(response)

        ctx_a = contextvars.copy_context()
        ctx_b = contextvars.copy_context()
        t_a = threading.Thread(target=ctx_a.run, args=(run, response_a))
        t_b = threading.Thread(target=ctx_b.run, args=(run, response_b))
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

    warrants = {w.id: w for w in krun.session.store.all()}
    a_chain = sorted((w for w in warrants.values() if "a-result-" in str(w.content)), key=lambda w: _step_index(w.content))
    b_chain = sorted((w for w in warrants.values() if "b-result-" in str(w.content)), key=lambda w: _step_index(w.content))

    assert len(a_chain) == 20 and len(b_chain) == 20
    assert _chain_is_self_consistent(a_chain), "response a's own chain must not be torn by response b's concurrent processing"
    assert _chain_is_self_consistent(b_chain), "response b's own chain must not be torn by response a's concurrent processing"
