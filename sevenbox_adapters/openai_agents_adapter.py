"""
sevenbox_adapters.openai_agents_adapter -- the "sevenbox wrap" story for the
OpenAI Agents SDK (`agents` package, `openai-agents` on PyPI).

Verified against the actual installed openai-agents==0.19.4 package, not
training-data assumptions about its API. Verification includes a full
offline run -- a real, network-free `Model`/`ModelProvider` implementation
driving a real `Agent`/`Runner` through an actual tool call -- not just
import-time correctness (see tests/test_openai_agents_adapter.py).

Integration surface, deliberately small, same shape as
sevenbox_adapters.langgraph_adapter:

    from sevenbox_adapters.openai_agents_adapter import sevenbox_tool, sevenbox_agent_run
    from agents import Agent, Runner, function_tool
    from warrant_core import SourceKind

    @function_tool
    @sevenbox_tool(source_kind=SourceKind.TOOL)
    def lookup_order(order_id: str) -> dict:
        return {"status": "shipped"}

    agent = Agent(name="Support", tools=[lookup_order], instructions="...")

    with sevenbox_agent_run() as krun:
        result = await Runner.run(agent, "...", hooks=krun.hooks)
        print(krun.session.frontier().live)

One change to an existing Agents SDK app: `@sevenbox_tool` under
`@function_tool` on each tool function, `sevenbox_agent_run()` around the
`Runner.run(...)` call site. Tool bodies untouched. `hooks=krun.hooks`
is what gives genuine parallel tool calls correct, independent parents
(see "Design" below) -- harmless to pass even when every tool call in
practice turns out to be sequential.

## Why `@sevenbox_tool` goes UNDER `@function_tool`, not the other way around

`@function_tool` inspects the wrapped callable's signature and docstring
to build the tool's JSON schema -- verified directly against the real SDK
that `functools.wraps` preservation is enough for this to still work
through a wrapper (`inspect.signature` follows `__wrapped__`). Stacking
`sevenbox_sdk.tool` (whose decorated functions become `_SevenboxFunction`
instances, not plain functions with a preserved `__wrapped__` signature)
directly under `@function_tool` would break that introspection -- this
adapter avoids `sevenbox_sdk.tool` entirely rather than risk it silently.

## Design: turn-aware by default when hooks are wired, ambient-frontier otherwise

A tool function here receives plain, JSON-decoded arguments from the
Agents SDK's own tool-calling loop, never a `Traced` wrapper the way
`sevenbox_sdk`'s own decorators expect -- there is no parent_ids breadcrumb
to read off the call arguments themselves. Earlier versions of this
adapter (and `sevenbox_adapters.langgraph_adapter.GraphSession`, which still
has this limitation) stopped there and chained every call onto "whatever
the session's current frontier is" unconditionally, which silently
manufactures a false dependency between two tool calls that both fired
in the same turn from the same prior state (e.g. an independent
watchlist check and an independent document verification both justified
by the same ticket -- see tests/test_openai_agents_adapter.py's
parallel-tool-call tests for the exact failure mode this produces
uncorrected).

The fix does not require a parent_ids breadcrumb on the call arguments,
because the SDK exposes the missing information one level up:
`agents.lifecycle.RunHooksBase.on_llm_end(context, agent, response)`
hands back the full `ModelResponse` -- every tool call the model just
requested -- before any of them individually execute. `SevenboxRunHooks`
(below) uses that hook to freeze the current frontier the instant a
multi-tool-call turn is observed; every call in that turn is given that
SAME frozen parent set (instead of each clobbering the frontier for the
next one to see); once every call in the turn has landed, the frontier
re-opens as the union of everything the turn produced -- genuine
multi-tip fan-out, matching the causal shape of the actual work done.

This requires passing `hooks=krun.hooks` to `Runner.run(...)` (see the
usage example above) -- `AgentSession` has no way to observe a turn
boundary on its own. If `hooks` is not passed, `record_call` falls back
to the original one-call-at-a-time ambient-frontier behavior: correct
for the common case (one tool call at a time, sequentially), and no
worse than before for the parallel case -- nothing breaks either way,
the fix is additive and opt-in via one extra keyword argument.

`source_kind` defaults to `SourceKind.TOOL` (effectful) for the same
reason `sevenbox_node` defaults there: the adapter cannot know by inspection
whether a tool body is genuinely pure.

## A real bug, found by code review, in `SevenboxRunHooks.on_llm_end`'s call count

`on_llm_end` needs to know how many tool calls a turn will make, to
freeze the parent set until all of them land (`AgentSession.begin_turn`).
An earlier version counted every `ResponseFunctionToolCall` in the
response, regardless of whether it targeted a `@sevenbox_tool`-wrapped
function -- but `_pending_remaining` only ever decrements when a wrapped
call actually reaches `record_call`. An `Agent` mixing wrapped and plain
(unwrapped) tools in one turn would overcount: `_pending_remaining` never
reached zero, so the turn stayed "pending" forever, and the next
`begin_turn()` call would fold it in via the "previous turn never fully
resolved" recovery path as though it had failed mid-flight, even though
it completed cleanly. The fix is a module-level `_SEVENBOX_TOOL_NAMES` set,
populated by the `sevenbox_tool` decorator at definition time, that
`on_llm_end` filters the count against -- the same tradeoff
`sevenbox_adapters.crewai_adapter`'s tool registry makes (a process-global
name registry, not scoped per-session, since `sevenbox_tool` decorates a
function before any run/session exists to scope it to), but narrower in
its failure mode: this set only answers "does a sevenbox-wrapped tool with
this name exist," never "use this specific tool's configuration," so a
name collision across unrelated modules can misdirect a count, not
silently apply the wrong redaction policy. See
tests/test_openai_agents_adapter.py::test_turn_mixing_sevenbox_and_plain_
tools_still_closes_correctly for the constructed failure case and fix.

Two previously-untested code paths were also exercised directly, per
this project's own "verify, don't assume" discipline, rather than left
implicit: the lazy axiom creation in `record_call` for a turn that fans
out to parallel calls before `AgentSession.start()` was ever called
(confirmed all siblings share one freshly-created axiom, not each
creating their own -- see test_parallel_turn_before_any_explicit_start_
shares_one_freshly_created_axiom), and `begin_turn`'s fold-in recovery
for a turn that never fully resolved (see test_begin_turn_fold_in_
recovers_a_turn_that_never_fully_resolved, which also confirms
`AgentSession._frontier` -- this adapter's own internal "what to attach
the next call to" bookkeeping -- is a genuinely distinct concept from
`compute_frontier(store).live`, the frontier
based on supersession: they coincide only when nothing has been
superseded, which is not assumed elsewhere in this codebase and should
not be assumed here either).

Finally, `test_parallel_tool_calls_without_hooks_manufacture_a_false_
dependency` (and this module's own "Design" section above) rely on an
unstated fact about the installed SDK: openai-agents dispatches sync
tool calls cooperatively, with no thread pool (unlike pydantic-ai, see
`sevenbox_adapters.pydantic_ai_adapter`'s module docstring), so call_1
always fully completes before call_2 starts. This is now pinned in
isolation from any Sevenbox code, in
test_dispatch_order_of_parallel_tool_calls_matches_the_documented_
assumption, so a future SDK version that changed this fails there
specifically rather than as a silent behavior change in this adapter.
"""
from __future__ import annotations

import functools
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

try:
    from agents.lifecycle import RunHooksBase as _RunHooksBase
    from openai.types.responses import ResponseFunctionToolCall as _ResponseFunctionToolCall
    _AGENTS_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional extra is absent
    _RunHooksBase = object
    _ResponseFunctionToolCall = None
    _AGENTS_SDK_AVAILABLE = False


_SEVENBOX_TOOL_NAMES: set = set()
"""Tool names registered via `@sevenbox_tool` (populated at decoration time,
module load), consulted by `SevenboxRunHooks.on_llm_end` so it counts only
calls this adapter will actually see reach `record_call` -- see that
class's docstring for exactly why an unfiltered count is wrong when an
Agent mixes `@sevenbox_tool`-wrapped and plain tools in one turn.

Process-global, same tradeoff `sevenbox_adapters.crewai_adapter`'s tool
registry makes and documents: two functions in different, unrelated
modules registered under the identical `name=` (explicit or defaulted
`__qualname__`) are indistinguishable here. The failure mode is milder
than CrewAI's registry (which maps name to full per-tool config,
including redaction policy) because this set only answers "does a
sevenbox-wrapped tool with this name exist in this process," never "use
THIS specific tool's configuration" -- a collision can cause a turn's
call-count to include a same-named tool that is not the one actually
firing, which self-corrects once that real call resolves (or, in the
already-covered fold-in case, is absorbed by the next `begin_turn`), not
a silent, permanent misattribution. Not scoped per-session because
`sevenbox_tool` decorates a function once, at definition time, before any
session/run exists to scope it to."""


class AgentSession:
    """Tracks the causal frontier of one Agents SDK run. See module
    docstring's "Design" section for what this does and does not model,
    and how to opt into turn-aware fan-out via `hooks=krun.hooks`."""

    def __init__(
        self, *, store: Optional[WarrantStore] = None, replay_from: Optional[WarrantStore] = None
    ) -> None:
        self.store = store if store is not None else WarrantStore()
        self.mode = "replay" if replay_from is not None else "live"
        self._replay_executor = (
            ReplayEffectExecutor(reference_store=replay_from) if replay_from is not None else None
        )
        self._frontier: set = set()
        # Turn-aware fan-out state (see module docstring's "Design"
        # section). None when no multi-tool-call turn is in flight --
        # record_call falls back to plain ambient-frontier chaining.
        self._pending_parent_ids: Optional[tuple] = None
        self._pending_remaining: int = 0
        self._pending_new_ids: set = set()

    def start(self, initial_input: Any) -> None:
        """Optional: root the run on an explicit axiom (e.g. the user's
        message) before any tool fires. If the first tool call arrives
        without this having been called, record_call() roots the run on
        that call's own arguments instead -- never silently parentless."""
        if self._frontier:
            return
        w = create_warrant(
            content=initial_input, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def begin_turn(self, n_tool_calls: int) -> None:
        """Call before any tool call in a turn runs (SevenboxRunHooks.on_llm_end
        does this automatically). n_tool_calls <= 1 is the ordinary
        sequential case and is a no-op -- existing single-call-at-a-time
        behavior is already correct and untouched. n_tool_calls > 1
        freezes the current frontier as the shared parent set every call
        in this turn will use, instead of each clobbering it for the
        next one to see."""
        if self._pending_parent_ids is not None and self._pending_new_ids:
            # A previous turn never fully resolved (e.g. a tool raised
            # mid-turn) -- fold what it did produce into the frontier
            # rather than losing those warrants' place in it.
            self._frontier = set(self._pending_new_ids)
        self._pending_parent_ids = None
        if n_tool_calls <= 1:
            return
        self._pending_parent_ids = tuple(sorted(self._frontier))
        self._pending_remaining = n_tool_calls
        self._pending_new_ids = set()

    def record_call(
        self,
        *,
        tool_name: str,
        source_kind: SourceKind,
        epistemic: EpistemicClass,
        reversibility: ReversibilityClass,
        args: tuple,
        kwargs: dict,
        real_fn: Callable[..., Any],
        redaction: Optional[RedactionPolicy] = None,
        origin: Optional[Origin] = None,
    ) -> Any:
        request = {"args": list(args), "kwargs": kwargs}
        in_turn = self._pending_parent_ids is not None

        if in_turn:
            if not self._frontier:
                # Turn began before any axiom existed (e.g. the very
                # first LLM response already requests parallel calls).
                # Create it now and let every call in this turn share it.
                self.start(request)
                self._pending_parent_ids = tuple(sorted(self._frontier))
            parent_ids = list(self._pending_parent_ids)
        else:
            self.start(request)
            parent_ids = sorted(self._frontier)

        if source_kind in PURE_SOURCE_KINDS:
            result = real_fn(*args, **kwargs)
            w = create_warrant(
                content=result, source_kind=source_kind, epistemic=epistemic,
                reversibility=reversibility, parent_ids=parent_ids, origin=origin,
            )
        else:
            if self.mode == "live":
                result = real_fn(*args, **kwargs)
                raw_response = result
            else:
                assert self._replay_executor is not None
                cached = self._replay_executor.invoke(tool_name, parent_ids, request=request)
                result = cached.content
                raw_response = cached.raw
            w = create_warrant(
                content=result, source_kind=source_kind, epistemic=epistemic,
                reversibility=reversibility, parent_ids=parent_ids, origin=origin,
                effect_record=EffectRecord(
                    request=apply_redaction(request, redaction),
                    response=apply_redaction(raw_response, redaction),
                    source_name=tool_name, redacted=redaction is not None,
                ),
            )

        self.store.add(w)

        if in_turn:
            self._pending_new_ids.add(w.id)
            self._pending_remaining -= 1
            if self._pending_remaining <= 0:
                # Whole turn landed: the frontier becomes every sibling
                # this turn produced -- genuine multi-tip fan-out, not
                # just whichever call happened to finish last.
                self._frontier = set(self._pending_new_ids)
                self._pending_parent_ids = None
                self._pending_new_ids = set()
        else:
            self._frontier = {w.id}

        return result

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


class SevenboxRunHooks(_RunHooksBase):
    """Bridges the Agents SDK's real turn boundary to AgentSession.begin_turn.

    Pass `hooks=krun.hooks` to `Runner.run(...)` to get genuine multi-tip
    fan-out for parallel tool calls (see module docstring's "Design"
    section). Stateless and reusable -- reads the session from the same
    ContextVar `@sevenbox_tool`'s wrapper does, so a fresh instance per run
    is not required (`krun.hooks` hands you one anyway).
    """

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        if not _AGENTS_SDK_AVAILABLE:  # pragma: no cover - see import guard above
            raise RuntimeError(
                "SevenboxRunHooks.on_llm_end was invoked but the 'openai-agents' "
                "extra is not installed. This should not be reachable: a real "
                "Runner.run(...) call requires the same package."
            )
        session = _current_agent_session.get()
        if session is None:
            return
        # Count only calls this adapter will actually see reach
        # record_call -- i.e. calls to a @sevenbox_tool-registered name. An
        # unfiltered count (every ResponseFunctionToolCall in the turn)
        # overcounts whenever the Agent also has plain, unwrapped tools:
        # _pending_remaining would then never reach zero, since only
        # sevenbox-wrapped calls ever decrement it, leaving the turn
        # perpetually "pending" and every subsequent begin_turn() folding
        # it in as if it had failed mid-flight even though it completed
        # cleanly. See _SEVENBOX_TOOL_NAMES's docstring for the one caveat
        # this filter carries (name collisions across unrelated modules).
        n_calls = sum(
            1 for item in getattr(response, "output", [])
            if isinstance(item, _ResponseFunctionToolCall)
            and getattr(item, "name", None) in _SEVENBOX_TOOL_NAMES
        )
        session.begin_turn(n_calls)


_current_agent_session: ContextVar[Optional[AgentSession]] = ContextVar(
    "sevenbox_agent_session", default=None
)


def sevenbox_agent_run(
    *, store: Optional[WarrantStore] = None, replay_from: Optional[WarrantStore] = None
) -> "_SevenboxAgentRunContext":
    return _SevenboxAgentRunContext(AgentSession(store=store, replay_from=replay_from))


class _SevenboxAgentRunContext:
    def __init__(self, session: AgentSession) -> None:
        self.session = session
        self.hooks = SevenboxRunHooks()
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
    UNDER `@function_tool` (`@function_tool` outermost, `@sevenbox_tool`
    innermost) -- see module docstring."""

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or getattr(f, "__qualname__", getattr(f, "__name__", "tool"))
        _SEVENBOX_TOOL_NAMES.add(tool_name)

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            session = _current_agent_session.get()
            if session is None:
                raise RuntimeError(
                    f"Tool {tool_name!r} was called outside a `with sevenbox_agent_run():` "
                    f"block. Wrap your Runner.run(...) call site in "
                    f"`with sevenbox_agent_run() as krun:`."
                )
            return session.record_call(
                tool_name=tool_name, source_kind=source_kind, epistemic=epistemic,
                reversibility=reversibility, args=args, kwargs=kwargs, real_fn=f,
                redaction=redaction, origin=origin,
            )

        return wrapper

    return decorator(fn) if fn is not None else decorator
