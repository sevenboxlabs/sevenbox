"""
Real-agent-framework integration tests for
sevenbox_adapters.agent_framework_adapter.

Uses a hand-built, network-free `BaseChatClient` subclass (agent-framework
ships no offline/scripted client of its own, unlike pydantic-ai's
`FunctionModel` or openai-agents' custom-`Model` support) composed exactly
the way `agent_framework/_clients.py._apply_get_response_docstrings`
documents as the framework's own canonical layering order
(`ChatMiddlewareLayer -> FunctionInvocationLayer -> ChatTelemetryLayer ->
BaseChatClient`) -- the same composition real provider clients (e.g. the
separate `agent-framework-openai` extra, not installed in this
environment) are expected to use, per that same source. This is the same
"verify against the actual installed SDK, not training-data assumptions"
standard the other five adapters' test files hold to; see
sevenbox_adapters/agent_framework_adapter.py's module docstring for the real
framework bug this composition choice was what actually surfaced.
"""
import asyncio
import sys
import threading

import pytest

agent_framework = pytest.importorskip("agent_framework")
from agent_framework import (  # noqa: E402
    Agent, BaseChatClient, ChatResponse, Content, FunctionMiddleware, Message,
)
from agent_framework._middleware import ChatMiddlewareLayer  # noqa: E402
from agent_framework._tools import FunctionInvocationLayer  # noqa: E402
from agent_framework.observability import ChatTelemetryLayer  # noqa: E402

from warrant_core import (  # noqa: E402
    EpistemicClass, ReversibilityClass, SourceKind, create_warrant,
)
from sevenbox_adapters.agent_framework_adapter import (  # noqa: E402
    AgentFrameworkSession, SevenboxFunctionMiddleware, sevenbox_agent_framework_run,
)


class ScriptedClient(ChatMiddlewareLayer, FunctionInvocationLayer, ChatTelemetryLayer, BaseChatClient):
    """A minimal, network-free chat client whose responses are driven by a
    caller-supplied script function `turn(messages) -> ChatResponse`, the
    same shape used by pydantic-ai's `FunctionModel`/openai-agents'
    scripted `Model` fakes elsewhere in this test suite. Deliberately
    composes all four layers agent-framework's own docstring-layering code
    (`_clients.py._apply_get_response_docstrings`) documents as canonical
    -- this is what actually reproduces the real framework bug documented
    in the adapter's module docstring; a bare `BaseChatClient` subclass
    would not.
    """

    def __init__(self, turn, **kwargs):
        super().__init__(**kwargs)
        self._turn_fn = turn

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        return self._turn_fn(list(messages))


def _agent(client, tools, name="tester"):
    return Agent(client=client, name=name, tools=tools)


# --- A real framework bug: FunctionMiddleware passed the documented, ---
# --- conventional way never fires when the client composes ---
# --- ChatMiddlewareLayer above FunctionInvocationLayer. Pinned here, in ---
# --- isolation from any Sevenbox code, so a future agent-framework upgrade ---
# --- that fixes this changes THIS test's result first, not a silent ---
# --- change in whether sevenbox_agent_framework_run() is even necessary. ---

def lookup_order(order_id: str) -> str:
    return f"status for {order_id}: shipped"


def check_watchlist(applicant_id: str) -> str:
    return f"watchlist result for {applicant_id}: clear"


def test_function_middleware_via_agent_constructor_middleware_is_silently_dropped():
    """Pins the real agent-framework bug documented in the adapter's
    module docstring, with no Sevenbox code involved: a `FunctionMiddleware`
    passed via `Agent(middleware=[...])` -- the framework's own documented
    mechanism -- never fires, for a client shaped like the framework's own
    canonical layering order. If a future agent-framework release fixes
    this, this test (not a mysterious downstream failure) is what breaks
    first."""
    events = []

    class ProbeMiddleware(FunctionMiddleware):
        async def process(self, context, call_next):
            events.append(context.function.name)
            await call_next()

    def turn(messages):
        if len(messages) == 1:
            return ChatResponse(
                messages=[Message(role="assistant", contents=[
                    Content.from_function_call(call_id="c1", name="lookup_order", arguments={"order_id": "4471"}),
                ])],
                response_id="turn-1",
            )
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])], response_id="turn-2")

    client = ScriptedClient(turn)
    agent = _agent(client, [lookup_order])
    result = asyncio.run(agent.run("check order 4471", middleware=[ProbeMiddleware()]))

    assert events == [], (
        "if this fails, agent-framework's own middleware= wiring now "
        "delivers FunctionMiddleware correctly -- sevenbox_agent_framework_run()'s "
        "instrument() workaround may no longer be necessary, see the "
        "adapter's module docstring"
    )
    assert result.text == "done"


def test_instrument_appends_directly_to_client_function_middleware():
    """The documented workaround: appending directly to the client's own
    `function_middleware` attribute bypasses the broken kwarg-forwarding
    path entirely and fires correctly."""
    events = []

    class ProbeMiddleware(FunctionMiddleware):
        async def process(self, context, call_next):
            events.append(context.function.name)
            await call_next()

    def turn(messages):
        if len(messages) == 1:
            return ChatResponse(
                messages=[Message(role="assistant", contents=[
                    Content.from_function_call(call_id="c1", name="lookup_order", arguments={"order_id": "4471"}),
                ])],
                response_id="turn-1",
            )
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])], response_id="turn-2")

    client = ScriptedClient(turn)
    client.function_middleware.append(ProbeMiddleware())
    agent = _agent(client, [lookup_order])
    asyncio.run(agent.run("check order 4471"))

    assert events == ["lookup_order"]


# --- Basic wiring ---

def test_tool_called_outside_sevenbox_agent_framework_run_produces_an_error_result_not_a_raised_exception():
    """Every other adapter in this package raises a RuntimeError that
    propagates straight out of the caller's `agent.run()`/`run_sync()`
    when a tool fires outside its `with sevenbox_*_run():` block. Agent
    Framework does NOT behave the same way, verified here rather than
    assumed: `_auto_invoke_function`'s `except Exception as exc:` wraps
    ANY exception raised inside `FunctionMiddleware.process()` -- including
    one raised before `call_next()` is ever awaited -- into a synthetic
    tool-error result fed back to the model, and the run completes
    normally. `SevenboxFunctionMiddleware` still raises (useful for logs/
    observability, and consistent with every other adapter's guard), but
    a caller must check the conversation history, not wrap `agent.run()`
    in `pytest.raises`, to detect this. Documented here so a future
    reader does not mistake the silence for the guard not firing at all."""
    captured_messages = []

    def turn(messages):
        captured_messages.append(list(messages))
        if len(messages) == 1:
            return ChatResponse(
                messages=[Message(role="assistant", contents=[
                    Content.from_function_call(call_id="c1", name="lookup_order", arguments={"order_id": "4471"}),
                ])],
                response_id="turn-1",
            )
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])], response_id="turn-2")

    client = ScriptedClient(turn)
    client.function_middleware.append(SevenboxFunctionMiddleware())
    agent = _agent(client, [lookup_order])

    result = asyncio.run(agent.run("check order 4471"))  # does NOT raise

    assert result.text == "done"
    second_turn_messages = captured_messages[1]
    error_contents = [
        c for m in second_turn_messages for c in m.contents
        if getattr(c, "type", None) == "function_result" and getattr(c, "exception", None)
    ]
    assert error_contents, "expected a function_result content carrying the guard's exception"
    assert "outside a `with sevenbox_agent_framework_run" in error_contents[0].exception


def test_wrapped_agent_populates_a_warrant_store():
    def turn(messages):
        if len(messages) == 1:
            return ChatResponse(
                messages=[Message(role="assistant", contents=[
                    Content.from_function_call(call_id="c1", name="lookup_order", arguments={"order_id": "4471"}),
                ])],
                response_id="turn-1",
            )
        return ChatResponse(messages=[Message(role="assistant", contents=["Your order has shipped."])], response_id="turn-2")

    client = ScriptedClient(turn)

    with sevenbox_agent_framework_run() as krun:
        krun.instrument(client)
        krun.session.start("check order 4471")
        agent = _agent(client, [lookup_order])
        result = asyncio.run(agent.run("check order 4471"))

    assert result.text == "Your order has shipped."
    assert len(krun.session.store) == 2  # explicit axiom + 1 tool call

    live = krun.session.frontier().live
    kinds = {w.source_kind for w in live}
    assert SourceKind.TOOL in kinds
    assert SourceKind.AXIOM in kinds

    tool_warrant = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    assert tool_warrant.content == "status for 4471: shipped"
    assert tool_warrant.effect_record is not None
    assert tool_warrant.effect_record.source_name == "lookup_order"
    assert len(tool_warrant.parent_ids) == 1


def test_live_then_replay_never_calls_the_real_tool_again():
    call_log = []

    def lookup_order(order_id: str) -> str:
        call_log.append(order_id)
        return f"status for {order_id}: shipped"

    def turn(messages):
        if len(messages) == 1:
            return ChatResponse(
                messages=[Message(role="assistant", contents=[
                    Content.from_function_call(call_id="c1", name="lookup_order", arguments={"order_id": "4471"}),
                ])],
                response_id="turn-1",
            )
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])], response_id="turn-2")

    live_client = ScriptedClient(turn)
    with sevenbox_agent_framework_run() as live_run:
        live_run.instrument(live_client)
        live_run.session.start("check order 4471")
        agent = _agent(live_client, [lookup_order])
        asyncio.run(agent.run("check order 4471"))
    assert call_log == ["4471"]

    replay_client = ScriptedClient(turn)
    with sevenbox_agent_framework_run(replay_from=live_run.session.store) as replay_run:
        replay_run.instrument(replay_client)
        replay_run.session.start("check order 4471")
        agent2 = _agent(replay_client, [lookup_order])
        asyncio.run(agent2.run("check order 4471"))

    assert call_log == ["4471"]  # unchanged -- the real tool body did not run again
    assert {w.id for w in replay_run.session.store.all()} == {w.id for w in live_run.session.store.all()}


# --- Genuine parallel fan-out ---

def test_two_parallel_tool_calls_get_correct_independent_parents():
    """Uses tool bodies with a small, real `time.sleep()` -- not the
    instant, near-zero-cost `lookup_order`/`check_watchlist` used
    elsewhere in this file -- deliberately, because trivially-fast tool
    bodies turned out to reliably NOT overlap in this environment (see
    module docstring's "Known limitation" section): a call that returns a
    plain f-string completes its real-thread round trip fast enough that
    the second sibling's `begin_call()` often does not arrive until after
    the first's `end_call()` already closed the batch, exactly
    reproducing test_KNOWN_LIMITATION_a_fast_sibling_finishing_before_a_
    slow_one_starts_is_misattributed -- confirmed by first writing this
    test with the fast tools and watching it fail with exactly that
    signature. A short sleep is representative of real tools (network
    calls, DB queries) that take non-trivial time, and is what actually
    demonstrates the in-flight-counter mechanism working, as opposed to
    the instant-tool-body edge case its own known-limitation test already
    covers deterministically."""
    import time

    def slow_lookup_order(order_id: str) -> str:
        time.sleep(0.05)
        return f"status for {order_id}: shipped"

    def slow_check_watchlist(applicant_id: str) -> str:
        time.sleep(0.05)
        return f"watchlist result for {applicant_id}: clear"

    def turn(messages):
        if len(messages) == 1:  # one turn, two tool calls -- genuine fan-out
            return ChatResponse(
                messages=[Message(role="assistant", contents=[
                    Content.from_function_call(call_id="c1", name="slow_lookup_order", arguments={"order_id": "4471"}),
                    Content.from_function_call(call_id="c2", name="slow_check_watchlist", arguments={"applicant_id": "A-991"}),
                ])],
                response_id="turn-1",
            )
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])], response_id="turn-2")

    client = ScriptedClient(turn)
    with sevenbox_agent_framework_run() as krun:
        krun.instrument(client)
        krun.session.start("run checks")
        agent = _agent(client, [slow_lookup_order, slow_check_watchlist])
        asyncio.run(agent.run("run checks"))

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids != (lookup_w.id,), "no false dependency between siblings"


def test_sequential_calls_still_chain_correctly_when_not_parallel():
    def turn(messages):
        if len(messages) == 1:
            return ChatResponse(
                messages=[Message(role="assistant", contents=[
                    Content.from_function_call(call_id="c1", name="lookup_order", arguments={"order_id": "4471"}),
                ])],
                response_id="turn-1",
            )
        if len(messages) == 3:
            return ChatResponse(
                messages=[Message(role="assistant", contents=[
                    Content.from_function_call(call_id="c2", name="check_watchlist", arguments={"applicant_id": "A-991"}),
                ])],
                response_id="turn-2",
            )
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])], response_id="turn-3")

    client = ScriptedClient(turn)
    with sevenbox_agent_framework_run() as krun:
        krun.instrument(client)
        krun.session.start("go")
        agent = _agent(client, [lookup_order, check_watchlist])
        asyncio.run(agent.run("go"))

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    lookup_w = next(w for w in warrants.values() if "status for" in str(w.content))
    watchlist_w = next(w for w in warrants.values() if "watchlist result" in str(w.content))

    assert lookup_w.parent_ids == (axiom.id,)
    assert watchlist_w.parent_ids == (lookup_w.id,), "the later, sequential call must chain onto the first"


# --- Multi-agent fan-out: Agent A / Agent B -> shared session (question 4) ---

def test_two_agents_running_concurrently_produce_a_correctly_forked_and_merged_causal_graph():
    """The `Agent -> Agent A / Agent B -> Decision` shape: two independent
    Agent.run() calls, each with its own client instrumented with the SAME
    SevenboxFunctionMiddleware instance sharing one AgentFrameworkSession, run
    concurrently via asyncio.gather. See the adapter's module docstring
    ("What this pass did not verify") for why this is asyncio.gather over
    two Agent.run() calls rather than a real
    agent_framework_orchestrations.ConcurrentBuilder run -- the latter was
    not independently verified in this pass. Uses a small `time.sleep()`
    in each tool body for the same reason
    test_two_parallel_tool_calls_get_correct_independent_parents does --
    see that test's docstring."""
    import time

    def make_turn(tool_name, arg_name, arg_value):
        def turn(messages):
            if len(messages) == 1:
                return ChatResponse(
                    messages=[Message(role="assistant", contents=[
                        Content.from_function_call(call_id="c1", name=tool_name, arguments={arg_name: arg_value}),
                    ])],
                    response_id="turn-1",
                )
            return ChatResponse(messages=[Message(role="assistant", contents=["done"])], response_id="turn-2")
        return turn

    def slow_lookup_order(order_id: str) -> str:
        time.sleep(0.05)
        return f"status for {order_id}: shipped"

    def slow_check_watchlist(applicant_id: str) -> str:
        time.sleep(0.05)
        return f"watchlist result for {applicant_id}: clear"

    client_a = ScriptedClient(make_turn("slow_lookup_order", "order_id", "4471"))
    client_b = ScriptedClient(make_turn("slow_check_watchlist", "applicant_id", "A-991"))

    with sevenbox_agent_framework_run() as krun:
        krun.instrument(client_a)
        krun.instrument(client_b)
        krun.session.start("dispatch to both agents")
        agent_a = _agent(client_a, [slow_lookup_order], name="agent_a")
        agent_b = _agent(client_b, [slow_check_watchlist], name="agent_b")

        async def _run_both():
            return await asyncio.gather(agent_a.run("check order"), agent_b.run("check watchlist"))

        asyncio.run(_run_both())

        # NOTE: krun.session.frontier() is compute_frontier(store) -- the
        # the actual frontier (everything not superseded),
        # which still includes the axiom itself (it has children, but
        # nothing supersedes it). session._frontier is this adapter's own
        # "current DAG sink" bookkeeping -- exactly the two tool-call
        # warrants a merge Decision should cite -- the same distinction
        # documented in test_openai_agents_adapter.py's fold-in test.
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
    assert watchlist_w.parent_ids == (axiom.id,)
    assert set(decision.parent_ids) == {lookup_w.id, watchlist_w.id}, (
        "the merge Decision must cite BOTH agents' tool-call warrants as parents"
    )


# --- Deterministic concurrency: AgentFrameworkSession.begin_call/end_call ---
# --- under real thread contention, reproduced via threading.Barrier -- ---
# --- same methodology already used for the PydanticAI locking fix. ---

class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _drive_concurrent_siblings(session: AgentFrameworkSession, *, n_siblings: int):
    barrier = threading.Barrier(n_siblings)
    parent_id_sets_seen: list = []
    warrant_ids: list = []
    errors: list = []
    results_lock = threading.Lock()

    def sibling(i: int) -> None:
        try:
            barrier.wait(timeout=5)
            parent_ids = session.begin_call(fallback_axiom_content={"sibling": i})
            w = create_warrant(
                content={"sibling_result": i}, source_kind=SourceKind.DERIVED,
                epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
                parent_ids=tuple(parent_ids),
            )
            session.store.add(w)
            session.end_call(w.id)
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


def test_concurrent_siblings_never_corrupt_the_in_flight_counter_or_lose_a_warrant():
    """Unlike PydanticSession's equivalent test, this one does NOT assert
    all 8 siblings share one parent set -- `AgentFrameworkSession` groups
    batches by dispatch *timing* (module docstring question 8), not by an
    exact per-call identity, so 8 threads released simultaneously by a
    Barrier but otherwise running independently (no synchronization
    between one thread's begin_call/end_call pair and the next) WILL
    sometimes split into more than one batch -- that is the accepted,
    separately-tested heuristic gap
    (test_KNOWN_LIMITATION_a_fast_sibling_finishing_before_a_slow_one_
    starts_is_misattributed), not something this test should paper over.

    What the lock actually has to guarantee, independent of that timing
    gap: every begin_call()/end_call() pair is internally consistent
    (never a crash, never a lost or duplicated sibling), and the in-flight
    counter always returns to exactly zero once every sibling has
    completed its round trip -- a stuck non-zero counter would mean a
    lost decrement (a genuine data race), not just a timing-dependent
    grouping choice.
    """
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for trial in range(200):
            session = AgentFrameworkSession()
            frontier, parent_sets, warrant_ids, errors = _drive_concurrent_siblings(session, n_siblings=8)
            assert not errors, f"trial {trial}: unexpected exception(s): {errors}"
            assert len(parent_sets) == 8, f"trial {trial}: lost a sibling's result entirely"
            assert len(warrant_ids) == 8, f"trial {trial}: duplicate or lost warrant id(s)"
            assert session._in_flight == 0, (
                f"trial {trial}: in-flight counter did not return to zero "
                f"({session._in_flight}) -- a lost decrement, not a timing-dependent "
                f"grouping choice"
            )
            assert frontier and frontier.issubset(warrant_ids), (
                f"trial {trial}: final frontier {frontier} contains an id that was "
                f"never produced by this run"
            )
    finally:
        sys.setswitchinterval(old_interval)


def test_corruption_is_detectable_without_the_lock():
    """Proves the test above is meaningful: with `_lock` swapped for a
    no-op, the SAME driver reliably produces a non-zero in-flight counter
    at the end (a lost decrement -- real corruption, not the accepted
    timing-dependent grouping gap) at least once across many trials."""
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    corruption_observed = False
    try:
        for _ in range(200):
            session = AgentFrameworkSession()
            session._lock = _NullLock()
            frontier, parent_sets, warrant_ids, errors = _drive_concurrent_siblings(session, n_siblings=8)
            if errors or len(warrant_ids) != 8 or session._in_flight != 0:
                corruption_observed = True
                break
    finally:
        sys.setswitchinterval(old_interval)

    assert corruption_observed, (
        "expected the unlocked driver to exhibit detectable corruption at "
        "least once across 200 trials -- if this fails, the Barrier-based "
        "test above may not actually be exercising the race it claims to"
    )


def test_KNOWN_LIMITATION_a_fast_sibling_finishing_before_a_slow_one_starts_is_misattributed():
    """Documents, with a deterministic reproduction, exactly what the
    module docstring's "Known limitation: batch grouping is a real-
    concurrency heuristic, not an exact signal" section describes: if one
    call of what SHOULD be a two-call parallel batch completes (closing
    the in-flight counter back to zero) before its sibling has even
    called begin_call(), the sibling is wrongly treated as opening a NEW
    batch, parented on the first call's already-published result instead
    of sharing its parent set. This is a real, accepted gap -- not
    silently hidden -- because agent-framework exposes no per-call batch
    identity for FunctionMiddleware to use instead (module docstring
    question 8)."""
    session = AgentFrameworkSession()
    session.start("axiom")
    axiom_id = next(iter(session._frontier))

    # "Fast" sibling: begins and fully completes before the slow one starts.
    fast_parents = session.begin_call()
    fast_w = create_warrant(
        content={"result": "fast"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        parent_ids=tuple(fast_parents),
    )
    session.store.add(fast_w)
    session.end_call(fast_w.id)

    # "Slow" sibling of what was, in the real batch, meant to be the SAME
    # turn -- but by the time it calls begin_call(), the counter has
    # already returned to zero, so it is mis-attributed as a new batch.
    slow_parents = session.begin_call()

    assert fast_parents == (axiom_id,)
    assert slow_parents == (fast_w.id,), (
        "this IS the known limitation: the slow sibling's parent set "
        "wrongly includes the fast sibling's result instead of sharing "
        "the axiom as a true sibling would"
    )
    assert slow_parents != fast_parents, (
        "if this ever starts passing, the heuristic's known gap has been "
        "closed -- update the module docstring's Known Limitation section"
    )
