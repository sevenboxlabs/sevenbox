"""
Adversarial validation of sevenbox_sdk itself with NO framework and NO adapter
in between -- the fourteenth and final entry in the cross-framework
validation program. Every other file in
this pattern (test_letta_adapter.py, test_google_adk_adapter.py, ...) tests
an adapter's Session class absorbing a third-party framework's concurrency
model. There is no adapter here -- these tests exercise sevenbox_sdk.session /
sevenbox_sdk.decorators directly, under real asyncio and real OS-thread
concurrency, the way an application team that built its own agent loop
(no LangGraph, no OpenAI Agents SDK, nothing) actually would.

Findings, each with a test below:

1. asyncio.gather() correctly propagates SevenboxSession's contextvar to every
   gathered task (CPython copies the current Context at Task-creation time)
   -- real parallel dispatch of two @tool calls, sharing one oracle Warrant
   as parent, works with zero adapter code.
2. A sibling tool call failing inside the same asyncio.gather() batch does
   NOT retroactively erase an already-succeeded sibling's Warrant --
   because sevenbox_sdk.decorators commits each call's Warrant to the store
   immediately and individually (there is no batch-buffering step
   anywhere in _SevenboxFunction.__call__), there is no shared buffer for a
   failure to corrupt. This is structurally stronger than the equivalent
   finding in Strands/Haystack, where the framework's own batching
   happened to preserve the sibling by design choice -- here there is no
   batch as a synchronization point at all.
3. KNOWN LIMITATION: @tool/@oracle/@derive have no async support. Wrapping
   an `async def` function crashes inside create_warrant()'s
   copy.deepcopy(content) with a loud TypeError (not silent corruption),
   because _SevenboxFunction.__call__ calls self.fn(...) synchronously and
   receives a coroutine object back as "the result." Documented as real,
   current SDK scope -- not fixed in this pass (would require a
   deliberate async-decorator design, out of scope for closing this
   validation entry; see the module docstring of
   examples/refund_and_parallel_kyc_raw_loop.py).
4. KNOWN LIMITATION: contextvars do not propagate to a raw
   threading.Thread automatically -- the exact same general Python
   behavior already found (and fixed, inside LettaSession) for
   letta_adapter.py, now reproduced directly against sevenbox_sdk with no
   adapter involved. The correct fix is the standard library idiom
   (contextvars.copy_context().run(...)), demonstrated working in the
   test right after the one that demonstrates the failure -- this is an
   application-level concurrency responsibility, not something sevenbox_sdk
   needs a new module to solve, since a no-framework loop by definition
   owns its own concurrency strategy.
5. Real OS-thread parallel tool dispatch (ThreadPoolExecutor, context
   correctly copied per submitted task) does not corrupt the shared store
   -- verified through the *full* decorator path (create_warrant +
   apply_redaction + store.add), not just store.add() in isolation.
6. KNOWN LIMITATION, found while building test 5, not assumed: a single
   copy_context() result cannot be reused across two concurrent
   pool.submit() calls -- contextvars.Context objects refuse concurrent
   re-entry from a second thread while already entered on a first
   (`RuntimeError: cannot enter context: ... is already entered`), the
   exact same reentrancy rule that was already a real bug source in
   LettaSession.observe_response(). The first draft of test 5 shared one
   ctx across both submissions and would have hit this under real
   contention; the fix is one copy_context() call per concurrent
   submission, not one shared across all of them -- see
   test_KNOWN_LIMITATION_sharing_one_copied_context_across_concurrent_threads_raises.
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from sevenbox_sdk import SevenboxSession, derive, oracle, tool


@oracle
def plan_two_tool_calls(ticket):
    return {
        "tool_calls": [
            {"name": "lookup_order", "args": {"order_id": ticket["order_id"]}},
            {"name": "check_watchlist", "args": {"applicant_id": ticket["applicant_id"]}},
        ],
    }


@tool
def lookup_order(order_id):
    return {"status": "shipped", "order_id": order_id}


@tool
def check_watchlist(applicant_id):
    return {"watchlist": "clear", "applicant_id": applicant_id}


@derive
def eligibility(order, watchlist):
    return {"eligible": order["status"] == "shipped" and watchlist["watchlist"] == "clear"}


@tool
def ok_tool(x):
    return {"ok": x}


@tool
def erroring_tool(x):
    raise ValueError(f"boom: {x}")


@tool
async def async_lookup(order_id):
    await asyncio.sleep(0.01)
    return {"status": "shipped", "order_id": order_id}


def test_asyncio_gather_gives_both_parallel_calls_the_shared_oracle_parent():
    async def run(session):
        ticket = session.axiom({"order_id": "4471", "applicant_id": "A-991"})
        plan = plan_two_tool_calls(ticket)
        order_id_arg = plan["tool_calls"][0]["args"]["order_id"]
        applicant_id_arg = plan["tool_calls"][1]["args"]["applicant_id"]

        async def call_order():
            return lookup_order(order_id_arg)

        async def call_watchlist():
            return check_watchlist(applicant_id_arg)

        order, watchlist = await asyncio.gather(call_order(), call_watchlist())
        return plan, order, watchlist

    with SevenboxSession() as session:
        plan, order, watchlist = asyncio.run(run(session))
        assert order.warrant_id in session.store.children(plan.warrant_id)
        assert watchlist.warrant_id in session.store.children(plan.warrant_id)


def test_sequential_calls_still_chain_correctly_when_not_parallel():
    with SevenboxSession() as session:
        ticket = session.axiom({"order_id": "4471"})
        order = lookup_order(ticket["order_id"])
        assert order.warrant_id in session.store.children(ticket.warrant_id)


def test_partial_batch_failure_still_preserves_the_successful_siblings_warrant():
    async def run(session):
        ticket = session.axiom({"seed": True})

        async def call_ok():
            return ok_tool(ticket["seed"])

        async def call_err():
            return erroring_tool(ticket["seed"])

        await asyncio.gather(call_ok(), call_err())

    with SevenboxSession() as session:
        with pytest.raises(ValueError, match="boom"):
            asyncio.run(run(session))

        contents = [w.content for w in session.store.all()]
        assert {"ok": True} in contents, (
            "the successful sibling's Warrant should survive a co-batched "
            "sibling raising, because sevenbox_sdk commits each call's Warrant "
            "individually and immediately -- there is no batch buffer for "
            "the failure to have corrupted in the first place"
        )


def test_KNOWN_LIMITATION_an_async_def_tool_function_crashes_deepcopy_rather_than_being_awaited():
    async def run(session):
        ticket = session.axiom({"order_id": "4471"})
        return async_lookup(ticket["order_id"])

    with SevenboxSession() as session:
        with pytest.raises(TypeError, match="pickle"):
            asyncio.run(run(session))
        # Confirms *why* it crashes: _SevenboxFunction.__call__ called the
        # async function synchronously and got a live, un-awaited
        # coroutine object back as "the result" -- create_warrant()'s
        # defensive deepcopy is what actually raises, not a purpose-built
        # guard. A loud crash, not silent corruption, but still a real
        # gap: nothing in sevenbox_sdk rejects an async def at decoration time.


def test_KNOWN_LIMITATION_raw_threading_thread_does_not_propagate_session_context():
    with SevenboxSession() as session:
        ticket = session.axiom({"order_id": "4471"})
        errors = []

        def worker():
            try:
                lookup_order(ticket["order_id"])
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "No active SevenboxSession" in str(errors[0])


def test_a_thread_with_explicitly_copied_context_propagates_session_correctly():
    # The documented fix for the limitation above -- a standard library
    # idiom, not new sevenbox_sdk surface, since a no-framework loop owns its
    # own concurrency strategy by definition.
    with SevenboxSession() as session:
        ticket = session.axiom({"order_id": "4471"})
        results = {}

        def worker():
            results["order"] = lookup_order(ticket["order_id"])

        ctx = copy_context()
        t = threading.Thread(target=lambda: ctx.run(worker))
        t.start()
        t.join()

        assert "order" in results
        assert results["order"].warrant_id in session.store.children(ticket.warrant_id)


def test_real_threadpoolexecutor_parallel_dispatch_does_not_corrupt_the_shared_store():
    # Verifies this through the *full* decorator path -- create_warrant,
    # redaction, store.add -- not just store.add() called directly in
    # isolation.
    #
    # The first version of this test called copy_context() *inside*
    # call_order/call_watchlist -- i.e. inside the functions the pool
    # actually runs in its worker threads. That failed with the same
    # "No active SevenboxSession" RuntimeError as the KNOWN_LIMITATION test
    # above: by the time those functions run, execution is already on a
    # pool worker thread with its own empty context, so copy_context()
    # there copies nothing useful. The context must be captured ONCE, in
    # the main thread, *before* handing work to the pool, then threaded
    # into each submission explicitly -- the exact ordering mistake this
    # comment now exists to warn a future no-framework loop away from.
    with SevenboxSession() as session:
        ticket = session.axiom({"order_id": "4471", "applicant_id": "A-991"})
        # Each concurrent submission needs its OWN copy of the context --
        # contextvars.Context objects cannot be entered concurrently from
        # two threads at once (the same reentrancy rule that caused a real
        # bug in letta_adapter.py's Context handling). Sharing one ctx
        # across both pool.submit() calls below reproduces that exact
        # RuntimeError under real thread-pool concurrency; see
        # test_KNOWN_LIMITATION_sharing_one_copied_context_across_concurrent_threads_raises.
        order_ctx = copy_context()
        watchlist_ctx = copy_context()

        with ThreadPoolExecutor(max_workers=2) as pool:
            order_future = pool.submit(order_ctx.run, lookup_order, ticket["order_id"])
            watchlist_future = pool.submit(watchlist_ctx.run, check_watchlist, ticket["applicant_id"])
            order = order_future.result()
            watchlist = watchlist_future.result()

        elig = eligibility(order, watchlist)

        assert order.warrant_id in session.store.children(ticket.warrant_id)
        assert watchlist.warrant_id in session.store.children(ticket.warrant_id)
        assert set(elig.value.keys()) == {"eligible"}
        assert elig.value["eligible"] is True
        # No duplicate/corrupted index entries, proven through real
        # @tool calls.
        assert len(session.store.children(ticket.warrant_id)) == 2


def test_KNOWN_LIMITATION_sharing_one_copied_context_across_concurrent_threads_raises():
    # Found while writing test_real_threadpoolexecutor_parallel_dispatch_
    # does_not_corrupt_the_shared_store: the first draft shared a single
    # copy_context() result across two pool.submit() calls and would have
    # raised under real contention. Deterministic reproduction below (a
    # threading.Event pair forces genuine overlap instead of hoping two
    # fast calls happen to race) rather than relying on timing.
    with SevenboxSession() as session:
        session.axiom({"seed": True})
        ctx = copy_context()

        first_entered = threading.Event()
        release_first = threading.Event()

        def hold_context_open():
            def inner():
                first_entered.set()
                release_first.wait(timeout=5)
            ctx.run(inner)

        holder = threading.Thread(target=hold_context_open)
        holder.start()
        assert first_entered.wait(timeout=5)

        errors = []

        def try_reenter():
            try:
                ctx.run(lambda: None)
            except Exception as e:
                errors.append(e)

        intruder = threading.Thread(target=try_reenter)
        intruder.start()
        intruder.join(timeout=5)
        release_first.set()
        holder.join(timeout=5)

        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "already entered" in str(errors[0])
