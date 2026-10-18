"""
sevenbox_adapters.pydantic_ai_adapter -- the "sevenbox wrap" story for PydanticAI
(`pydantic_ai` package, `pydantic-ai` on PyPI).

Verified against the actual installed pydantic-ai==2.27.0 package, not
training-data assumptions about its API -- verification includes a full
offline run using pydantic_ai.models.function.FunctionModel (a real,
network-free Model implementation driving a real Agent through actual
tool calls, sync and async, sequential and genuinely parallel), plus a
targeted empirical check of RunContext.run_step's behavior specifically
(see tests/test_pydantic_ai_adapter.py's test_run_step_behavior_matches_
the_documented_assumption, which exists precisely so a future pydantic-ai
upgrade that changed this behavior would fail loudly here rather than
silently break parent attribution).

A real bug was found and fixed via deliberate adversarial testing (not
normal usage): `run_step` alone is not a safe turn identity, because it
resets to 1 at the start of every fresh `agent.run()`/`run_sync()` call
-- two calls from separate, unrelated runs can share a `run_step` number
by pure coincidence. An earlier version of this adapter compared
`run_step` alone and could match a call against a STALE, already-closed
turn from a prior, unrelated run. The fix keys turn identity on
`(run_id, run_step)` together -- `run_id` is unique per `agent.run()`
call, verified empirically. See
tests/test_pydantic_ai_adapter.py::test_a_fresh_run_never_matches_a_
stale_closed_turn_from_a_prior_run for the full reproduction and fix.
Both halves of that identity now have their own isolated,
adapter-code-free pinning test (`test_run_step_behavior_matches_the_
documented_assumption` and `test_run_id_behavior_matches_the_documented_
assumption`), so a future pydantic-ai upgrade that changed either fact
fails specifically there, not as a mysterious regression somewhere else.

A second real bug was found via code review, then confirmed against the
installed SDK's own source rather than assumed: `PydanticSession` had no
locking at all, despite pydantic-ai genuinely dispatching sync tool
calls onto real OS threads for a parallel batch -- confirmed directly in
`pydantic_ai/_utils.py`'s `run_in_executor` (routes through
`loop.run_in_executor`/an `anyio` worker-thread runner under the default
configuration) and `pydantic_ai/_tool_execution.py` (`asyncio.create_task`
per call in a batch). `_resolve_parent_ids`/`_finish_call` mutated
`_last_turn_run_id`/`_last_turn_step`/`_last_turn_parent_ids`/
`_current_turn_new_ids` with no synchronization -- the same class of
hazard `sevenbox_adapters.crewai_adapter.CrewSession` already carries a
`threading.RLock` for. A first attempt at a fix tried to close a second,
broader hazard too (a genuinely different turn interrupting one still in
flight) via a dict keyed on the full turn identity with a per-turn
in-flight call count; it introduced a real regression instead, caught
immediately by this adapter's own existing parallel-tool-call tests --
this adapter has no way to know in advance how many siblings a turn will
have, so "in-flight count reached zero" is not a safe proxy for "every
sibling has landed." The fix kept here is narrower and correct: a plain
`threading.Lock` guarding `start`/`_resolve_parent_ids`/`_finish_call`
each individually, preserving the original flat-field design ("the
tracked turn is whichever (run_id, run_step) was most recently seen")
rather than trying to track multiple turns at once. Verified with a
deterministic regression test
(`test_concurrent_siblings_of_one_turn_never_corrupt_shared_frontier`,
`threading.Barrier` + an aggressive `sys.setswitchinterval`, 200 trials)
paired with a second test
(`test_corruption_is_detectable_without_the_lock`) that swaps the lock
for a no-op and confirms the SAME driver reliably observes corruption --
proof the first test is actually exercising the hazard, not merely
passing regardless of whether the lock exists. Known, documented, not
solved: two genuinely concurrent, overlapping `agent.run()` calls
sharing one `PydanticSession` (distinct from parallel siblings of ONE
turn, which is covered) are not protected -- structurally this cannot
happen within a single run (the framework never dispatches turn N+1
until every one of turn N's tool calls has returned), so the remaining
gap is scoped to a narrower, more unusual usage pattern than the one
this fix targets.

Integration surface, deliberately small, same shape as
sevenbox_adapters.langgraph_adapter and sevenbox_adapters.openai_agents_adapter:

    from sevenbox_adapters.pydantic_ai_adapter import sevenbox_tool, sevenbox_agent_run
    from pydantic_ai import Agent, RunContext
    from warrant_core import SourceKind

    agent = Agent('openai:gpt-4o')

    @agent.tool
    @sevenbox_tool(source_kind=SourceKind.TOOL)
    def lookup_order(ctx: RunContext, order_id: str) -> dict:
        return {"status": "shipped"}

    with sevenbox_agent_run() as krun:
        krun.session.start("Where is my order?")
        result = agent.run_sync("Where is my order?")
        print(krun.session.frontier().live)

One change to an existing PydanticAI app: `@sevenbox_tool` under `@agent.tool`
on each tool function, `sevenbox_agent_run()` around the `agent.run(...)` /
`agent.run_sync(...)` call site. Tool bodies untouched. Unlike the other
two adapters, there is no third change needed to get correct parents for
genuinely parallel tool calls -- see "Design" below.

## Why `@sevenbox_tool` goes UNDER `@agent.tool`, not the other way around

Same reasoning as `sevenbox_adapters.openai_agents_adapter`: `@agent.tool`
inspects the wrapped callable's signature (via pydantic_ai's own
`takes_run_context` check) to decide whether to pass a `RunContext` as
the first argument, and to build the tool's JSON schema from the
remaining parameters -- verified directly against the real SDK that
`functools.wraps` preservation is enough for this to still work through
a wrapper (`inspect.signature` follows `__wrapped__`).

## Design: automatic turn-aware fan-out, no hooks required

The other two adapters each pay a real cost for the same problem --
telling a genuinely parallel tool call apart from a sequential one, so
two independent pieces of evidence don't get a false causal edge between
them. `sevenbox_adapters.langgraph_adapter.GraphSession` requires a node to
manually declare `ADDITIONAL_PARENTS_KEY`. `sevenbox_adapters.
openai_agents_adapter.SevenboxRunHooks` requires the caller to additionally
pass `hooks=krun.hooks` to `Runner.run(...)`, wiring an SDK-level
lifecycle hook to observe turn boundaries from the outside.

PydanticAI needs neither, because the information already reaches the
tool call itself: every `@agent.tool`-decorated function receives a
`RunContext` whose `run_step` field is the current turn number --
confirmed empirically to be IDENTICAL for two tool calls the model
requested in the same turn (genuine fan-out) and to increase between
tool calls from different, sequential turns. `PydanticSession.
record_call` (and its async counterpart) compares each call's
`run_step` to the previous call's:

  - same `run_step` as the last call => this call is a sibling of a
    turn already in flight; it gets the SAME frozen parent set the
    first call in that turn got, not chained onto that sibling.
  - different `run_step` => a new turn; seeded from whatever the
    frontier settled to at the end of the previous turn (a set, with
    more than one element if that turn had genuine fan-out) -- correct,
    multi-tip fan-out is therefore the DEFAULT behavior here, not
    something requiring extra wiring or manual declaration.

## Why `RunContext` is mandatory: `@agent.tool`, not `@agent.tool_plain`

`@agent.tool_plain`-registered functions receive no `RunContext` at all
-- no `run_step`, nothing for `sevenbox_tool` to key turn-awareness off of.
`sevenbox_tool` raises a clear `TypeError` at call time if the wrapped
function is invoked without a `RunContext` carrying `run_step`, rather
than silently falling back to less-correct behavior.

## Sync and async tools both supported

PydanticAI is async-first; real tool bodies are commonly `async def`.
`sevenbox_tool` inspects the wrapped function once, at decoration time
(`inspect.iscoroutinefunction`), and produces a wrapper of the same
kind -- an async tool stays `async def` (awaited internally via
`PydanticSession.record_call_async`), a sync tool stays a plain
function (`PydanticSession.record_call`). Replay never needs to await
anything regardless of which kind the live tool was: `ReplayEffectExecutor
.invoke()` is a pure, synchronous cache lookup with no I/O (verified
directly against the class), so only the LIVE path branches on sync/async.

`source_kind` defaults to `SourceKind.TOOL` (effectful) for the same
reason the other two adapters default there: the adapter cannot know by
inspection whether a tool body is genuinely pure.
"""
from __future__ import annotations

import functools
import inspect
import threading
from contextvars import ContextVar
from typing import Any, Callable, Optional

from warrant_core import (
    EpistemicClass,
    Origin,
    RedactionPolicy,
    ReversibilityClass,
    SourceKind,
    WarrantStore,
    apply_redaction,
    compute_frontier,
    create_warrant,
    find_transitive_obligations,
)
from warrant_core import PURE_SOURCE_KINDS, EffectRecord
from warrant_runtime import ReplayEffectExecutor


class PydanticSession:
    """Tracks the causal frontier of one PydanticAI run. See module
    docstring's "Design" section for the turn-aware fan-out mechanism
    and why it needs no hooks here, unlike the other two adapters."""

    def __init__(
        self, *, store: Optional[WarrantStore] = None, replay_from: Optional[WarrantStore] = None
    ) -> None:
        self.store = store if store is not None else WarrantStore()
        self.mode = "replay" if replay_from is not None else "live"
        self._replay_executor = (
            ReplayEffectExecutor(reference_store=replay_from) if replay_from is not None else None
        )
        self._frontier: set = set()
        # Guards every read/write of the turn-tracking fields below
        # (including inside start()) -- see module docstring's "Thread
        # safety" section for the exact hazard this closes, confirmed
        # against pydantic-ai's real dispatch mechanism, not assumed.
        # Plain Lock, not RLock: no locked method here calls another
        # locked method on the same object while already holding the
        # lock (each of start/_resolve_parent_ids/_finish_call acquires
        # it independently, for the duration of that one method only --
        # never held across the tool's own execution, which happens
        # between _resolve_parent_ids returning and _finish_call being
        # called), so there is no reentrancy to accommodate.
        #
        # What this lock does and does not guarantee, stated precisely:
        # it makes each individual read-modify-write of the fields below
        # atomic, which is sufficient for genuinely parallel SIBLINGS of
        # ONE turn (the scenario module docstring point 3 confirms
        # pydantic-ai actually dispatches across real threads) -- two
        # threads racing to resolve or finish calls for the SAME
        # (run_id, run_step) cannot corrupt `_current_turn_new_ids` or
        # misidentify which parent set to share, because whichever
        # thread's lock-holding turn happens first is what the other
        # observes once it acquires the lock. It does NOT protect against
        # a genuinely different, concurrently-arriving turn (a different
        # run_step, or a second agent.run() call sharing this session)
        # landing in between one turn's own resolve and finish and
        # resetting `_current_turn_new_ids` before that turn's other
        # siblings land -- an earlier version of this fix attempted to
        # close that gap too, with a per-turn-keyed dict tracking an
        # "in-flight" count per (run_id, run_step); it introduced a real
        # regression instead (caught immediately by this adapter's own
        # existing parallel-tool-call tests, not left unnoticed): this
        # adapter has no reliable way to know in advance how many
        # siblings a turn will have, so treating "in-flight count reached
        # zero" as "this turn is fully closed" is only safe if EVERY
        # sibling has already called _resolve_parent_ids by then, which
        # is exactly the fact this adapter cannot observe. The narrower,
        # correct fix kept here matches this constraint instead of
        # fighting it: a turn remains "the currently tracked turn" for as
        # long as it is the MOST RECENTLY SEEN (run_id, run_step), the
        # same rule the original, unlocked version of this class used --
        # only the atomicity of each individual access is new. Structurally
        # different turns physically cannot interleave THIS way within a
        # single agent.run() call (the framework itself does not dispatch
        # turn N+1 until every one of turn N's tool calls has returned) --
        # the remaining gap is scoped to genuinely overlapping, concurrent
        # DIFFERENT agent.run() calls sharing one session, a narrower and
        # more unusual usage than the parallel-single-turn case this fix
        # targets, and not solved here.
        self._lock = threading.Lock()
        # Turn-aware fan-out state -- reactive, no external begin_turn()
        # call needed (contrast sevenbox_adapters.openai_agents_adapter).
        # KEYED ON (run_id, run_step), NOT run_step ALONE: pydantic-ai's
        # run_step resets to 1 at the start of every fresh agent.run()/
        # run_sync() call (verified empirically), so two calls from
        # DIFFERENT, unrelated runs can share the same run_step number.
        # A prior version of this adapter compared run_step alone, which
        # let a call from a brand-new run incorrectly match a STALE,
        # already-closed turn's frozen parent set purely by numeric
        # coincidence -- found via a real cross-run regression test
        # (test_a_fresh_run_never_matches_a_stale_closed_turn_from_a_
        # prior_run), not a hypothetical. run_id is unique per
        # agent.run() call and constant across every tool call within
        # it (also verified empirically), which is exactly the missing
        # half of the identity a turn needs.
        self._last_turn_run_id: Optional[str] = None
        self._last_turn_step: Optional[int] = None
        self._last_turn_parent_ids: Optional[tuple] = None
        self._current_turn_new_ids: set = set()

    def start(self, initial_input: Any) -> None:
        """Optional: root the run on an explicit axiom (e.g. the user's
        prompt) before any tool fires. If the first tool call arrives
        without this having been called, record_call() roots the run on
        that call's own arguments instead -- never silently parentless."""
        with self._lock:
            self._start_locked(initial_input)

    def _resolve_parent_ids(self, run_id: Any, run_step: int, fallback_axiom_content: Any) -> list:
        with self._lock:
            if not self._frontier:
                self._start_locked(fallback_axiom_content)
            if (run_id, run_step) == (self._last_turn_run_id, self._last_turn_step):
                # A sibling of a turn already in flight: reuse the SAME
                # frozen parent set that turn's first call resolved, not
                # whatever self._frontier has become since (which may
                # already include an earlier sibling's new Warrant).
                assert self._last_turn_parent_ids is not None
                return list(self._last_turn_parent_ids)
            # A new turn -- either the next step of the SAME run, or the
            # first call of a DIFFERENT run entirely (run_id changed).
            # Either way, seed fresh from the frontier as it stood at the
            # end of whichever turn, in whichever run, just closed.
            parent_ids = sorted(self._frontier)
            self._last_turn_run_id = run_id
            self._last_turn_step = run_step
            self._last_turn_parent_ids = tuple(parent_ids)
            self._current_turn_new_ids = set()
            return parent_ids

    def _finish_call(self, w) -> None:
        self.store.add(w)
        with self._lock:
            self._current_turn_new_ids.add(w.id)
            # The frontier is always "every Warrant produced so far in the
            # current (possibly still-open) turn" -- a single call's
            # result for the ordinary sequential case, and every sibling
            # once a multi-call turn has fully landed. Updated on every
            # call rather than only once some "expected" count is
            # reached, deliberately -- see this class's `_lock` docstring
            # for exactly why an earlier attempt at the latter regressed
            # the common case: this adapter has no reliable signal for
            # when a turn's siblings are all in.
            self._frontier = set(self._current_turn_new_ids)

    def _start_locked(self, initial_input: Any) -> None:
        """Same as `start()`, for callers that already hold `self._lock`
        (currently only `_resolve_parent_ids`) -- avoids the double-locking
        `start()` itself would need if it acquired the lock again."""
        if self._frontier:
            return
        w = create_warrant(
            content=initial_input, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def _build_warrant(
        self, *, tool_name, source_kind, epistemic, reversibility, parent_ids,
        request, result, raw_response, redaction, origin,
    ):
        if source_kind in PURE_SOURCE_KINDS:
            return create_warrant(
                content=result, source_kind=source_kind, epistemic=epistemic,
                reversibility=reversibility, parent_ids=parent_ids, origin=origin,
            )
        return create_warrant(
            content=result, source_kind=source_kind, epistemic=epistemic,
            reversibility=reversibility, parent_ids=parent_ids, origin=origin,
            effect_record=EffectRecord(
                request=apply_redaction(request, redaction),
                response=apply_redaction(raw_response, redaction),
                source_name=tool_name, redacted=redaction is not None,
            ),
        )

    def record_call(
        self,
        *,
        tool_name: str,
        source_kind: SourceKind,
        epistemic: EpistemicClass,
        reversibility: ReversibilityClass,
        run_id: Any,
        run_step: int,
        call_ctx: Any,
        args: tuple,
        kwargs: dict,
        real_fn: Callable[..., Any],
        redaction: Optional[RedactionPolicy] = None,
        origin: Optional[Origin] = None,
    ) -> Any:
        """Sync path. `call_ctx` (the PydanticAI RunContext) is threaded
        through to `real_fn` -- the tool body the user wrote expects it
        as its own first argument -- but deliberately excluded from
        `request`/effect_record: it is framework plumbing (a tracer,
        model handle, live usage counters), not meaningful audit data,
        and is not reliably JSON-serializable."""
        request = {"args": list(args), "kwargs": kwargs}
        parent_ids = self._resolve_parent_ids(run_id, run_step, request)

        if source_kind in PURE_SOURCE_KINDS:
            result = real_fn(call_ctx, *args, **kwargs)
            raw_response = None
        elif self.mode == "live":
            result = real_fn(call_ctx, *args, **kwargs)
            raw_response = result
        else:
            assert self._replay_executor is not None
            cached = self._replay_executor.invoke(tool_name, parent_ids, request=request)
            result = cached.content
            raw_response = cached.raw

        w = self._build_warrant(
            tool_name=tool_name, source_kind=source_kind, epistemic=epistemic,
            reversibility=reversibility, parent_ids=parent_ids, request=request,
            result=result, raw_response=raw_response, redaction=redaction, origin=origin,
        )
        self._finish_call(w)
        return result

    async def record_call_async(
        self,
        *,
        tool_name: str,
        source_kind: SourceKind,
        epistemic: EpistemicClass,
        reversibility: ReversibilityClass,
        run_id: Any,
        run_step: int,
        call_ctx: Any,
        args: tuple,
        kwargs: dict,
        real_fn: Callable[..., Any],
        redaction: Optional[RedactionPolicy] = None,
        origin: Optional[Origin] = None,
    ) -> Any:
        """Async path -- identical bookkeeping to record_call, the only
        difference is `await real_fn(...)`. Replay is NOT awaited even
        here: ReplayEffectExecutor.invoke() is synchronous regardless of
        whether the live tool was async (see module docstring)."""
        request = {"args": list(args), "kwargs": kwargs}
        parent_ids = self._resolve_parent_ids(run_id, run_step, request)

        if source_kind in PURE_SOURCE_KINDS:
            result = await real_fn(call_ctx, *args, **kwargs)
            raw_response = None
        elif self.mode == "live":
            result = await real_fn(call_ctx, *args, **kwargs)
            raw_response = result
        else:
            assert self._replay_executor is not None
            cached = self._replay_executor.invoke(tool_name, parent_ids, request=request)
            result = cached.content
            raw_response = cached.raw

        w = self._build_warrant(
            tool_name=tool_name, source_kind=source_kind, epistemic=epistemic,
            reversibility=reversibility, parent_ids=parent_ids, request=request,
            result=result, raw_response=raw_response, redaction=redaction, origin=origin,
        )
        self._finish_call(w)
        return result

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


_current_agent_session: ContextVar[Optional[PydanticSession]] = ContextVar(
    "sevenbox_pydantic_ai_session", default=None
)


def sevenbox_agent_run(
    *, store: Optional[WarrantStore] = None, replay_from: Optional[WarrantStore] = None
) -> "_SevenboxAgentRunContext":
    return _SevenboxAgentRunContext(PydanticSession(store=store, replay_from=replay_from))


class _SevenboxAgentRunContext:
    def __init__(self, session: PydanticSession) -> None:
        self.session = session
        self._token = None

    def __enter__(self) -> "_SevenboxAgentRunContext":
        self._token = _current_agent_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_agent_session.reset(self._token)
        return False


def sevenbox_tool(
    fn: Optional[Callable[..., Any]] = None,
    *,
    source_kind: SourceKind = SourceKind.TOOL,
    epistemic: EpistemicClass = EpistemicClass.VERIFIED,
    reversibility: ReversibilityClass = ReversibilityClass.PURE,
    name: Optional[str] = None,
    redaction: Optional[RedactionPolicy] = None,
    origin: Optional[Origin] = None,
):
    """Wrap a tool implementation so every call becomes a Warrant. Must go
    UNDER `@agent.tool` (`@agent.tool` outermost, `@sevenbox_tool` innermost)
    -- see module docstring. The wrapped function's first parameter must
    be a PydanticAI `RunContext` (i.e. register with `@agent.tool`, not
    `@agent.tool_plain`) -- `(run_id, run_step)` together are where
    turn-aware fan-out attribution comes from; see module docstring's
    "Design" section. Both are required, not `run_step` alone: `run_step`
    resets to 1 at the start of every fresh `agent.run()`/`run_sync()`
    call, so two calls from different, unrelated runs can otherwise share
    a number and be confused for turn-siblings -- `run_id` is unique per
    run and is what disambiguates them."""

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or getattr(f, "__qualname__", getattr(f, "__name__", "tool"))
        is_async = inspect.iscoroutinefunction(f)

        def _session_and_turn_identity(ctx: Any) -> "tuple[PydanticSession, Any, int]":
            session = _current_agent_session.get()
            if session is None:
                raise RuntimeError(
                    f"Tool {tool_name!r} was called outside a `with sevenbox_agent_run():` "
                    f"block. Wrap your agent.run(...)/agent.run_sync(...) call site in "
                    f"`with sevenbox_agent_run() as krun:`."
                )
            run_step = getattr(ctx, "run_step", None)
            if run_step is None:
                raise TypeError(
                    f"Tool {tool_name!r} was not called with a PydanticAI RunContext "
                    f"carrying `run_step`. @sevenbox_tool requires the tool to be registered "
                    f"with `@agent.tool` (which always passes a RunContext), not "
                    f"`@agent.tool_plain` (which passes none) -- turn-aware parent "
                    f"attribution has no other source for the current turn number."
                )
            run_id = getattr(ctx, "run_id", None)
            return session, run_id, run_step

        if is_async:
            @functools.wraps(f)
            async def wrapper(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                session, run_id, run_step = _session_and_turn_identity(ctx)
                return await session.record_call_async(
                    tool_name=tool_name, source_kind=source_kind, epistemic=epistemic,
                    reversibility=reversibility, run_id=run_id, run_step=run_step, call_ctx=ctx,
                    args=args, kwargs=kwargs, real_fn=f, redaction=redaction, origin=origin,
                )
        else:
            @functools.wraps(f)
            def wrapper(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                session, run_id, run_step = _session_and_turn_identity(ctx)
                return session.record_call(
                    tool_name=tool_name, source_kind=source_kind, epistemic=epistemic,
                    reversibility=reversibility, run_id=run_id, run_step=run_step, call_ctx=ctx,
                    args=args, kwargs=kwargs, real_fn=f, redaction=redaction, origin=origin,
                )

        return wrapper

    return decorator(fn) if fn is not None else decorator
