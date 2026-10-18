"""
sevenbox_adapters.agent_framework_adapter -- the "sevenbox wrap" story for Microsoft
Agent Framework (`agent-framework` on PyPI, the 2026 GA fusion of AutoGen
and Semantic Kernel).

Verified against the actual installed `agent-framework==1.13.0` package
(plus the optional `agent-framework-orchestrations` extra), not training-
data assumptions -- including empirical probes (not just source reading)
against a real, network-free `BaseChatClient` fake, because source reading
alone produced a wrong prediction here (see "A real framework bug", below
-- the single most important thing to read before using this adapter).

## Why this adapter looks structurally different from the other five

Every other adapter in this package attaches to a *tool-call* boundary the
framework hands over cleanly: `wrap_tool_call` (Deep Agents), `on_llm_end`
(OpenAI Agents SDK), a `RunContext` carrying `run_step` (PydanticAI),
`before_tool_call`/`after_tool_call` (CrewAI). Agent Framework exposes
*three* middleware tiers -- `AgentMiddleware` (wraps `Agent.run()` as a
whole), `ChatMiddleware` (wraps `ChatClient.get_response()`), and
`FunctionMiddleware` (wraps one tool invocation) -- and the natural
assumption, before checking, is that `ChatMiddleware` is the per-model-turn
hook this project has used elsewhere (`on_llm_end`'s analogue: "here is the
full batch of tool calls this turn requested, before any of them run").
**That assumption is wrong**, and only checking against a real running
agent (not the docs, not the type signatures alone) revealed it -- see
question 1 below.

## The 8 questions, answered from real investigation

1. **Where is a decision observed?** Not where it first appears to be.
   `ChatMiddleware.process(context, call_next)` wraps `ChatClient
   .get_response()` -- but for a client composing `FunctionInvocationLayer`
   *underneath* `ChatMiddlewareLayer` (the framework's own prescribed
   layering order, confirmed in `agent_framework/_clients.py
   ._apply_get_response_docstrings`: `ChatMiddlewareLayer ->
   FunctionInvocationLayer -> ChatTelemetryLayer -> BaseChatClient`),
   `FunctionInvocationLayer` runs the **entire multi-turn tool-calling
   loop** *inside* the single `get_response()` call that `ChatMiddleware`
   wraps. Empirically confirmed with a scripted two-turn client (turn 1
   requests two tool calls, turn 2 returns "done"): `ChatMiddleware
   .process()` fired exactly **once** for the whole run, and
   `context.result` after `call_next()` was turn 2's response -- turn 1's
   tool-call content was never visible to it at all. So `ChatMiddleware`
   is a whole-run hook here, not a per-turn one, despite being named and
   shaped like the OpenAI Agents SDK's `on_llm_end`. The real per-call
   decision point is `FunctionMiddleware.process()` (see question 2) --
   there is no framework-native "here is this turn's whole batch, before
   dispatch" hook at all in the standard `Agent.run()` path.
2. **Where is a tool call identified?** `FunctionMiddleware.process(
   context: FunctionInvocationContext, call_next)`. `context.function` is
   a `FunctionTool`, `context.arguments` the parsed, validated kwargs.
   Confirmed empirically that `context.result` after `await call_next()`
   is always `list[Content]` (the framework normalizes whatever the Python
   tool returned into `Content` items *inside* `tool.invoke()`, before the
   middleware ever sees it -- true whether or not middleware is even
   installed, read directly in `agent_framework/_tools.py
   ._auto_invoke_function`). Not calling `call_next()` at all and setting
   `context.result` directly skips the real invocation and is honored as
   the call's result -- confirmed by reading `FunctionMiddlewarePipeline
   .execute`'s `create_next_handler`, which only assigns `context.result =
   final_handler(context)` in the terminal (`index >= len(self._middleware)`)
   wrapper; a middleware that returns without awaiting `call_next()` never
   reaches it. This is what `SevenboxFunctionMiddleware` uses for replay
   (see question 7).
3. **Parallel vs. sequential?** Genuinely parallel, and via real OS
   threads for sync tools -- confirmed directly in `agent_framework
   ._tools.py._invoke_function`: `res = await asyncio.to_thread(self
   .__call__, **call_kwargs)` for any non-async tool (its own docstring:
   "Run sync tools off the event loop during async invocation"). The
   dispatch site, `_try_execute_function_call_groups`, builds the whole
   batch's `asyncio.Task`s in one tight synchronous loop (`contextvars
   .copy_context().run(asyncio.create_task, ...)` per call, read directly
   in source) before a single `await asyncio.gather(*execution_tasks)` --
   the same hazard class already found once in this project for
   PydanticAI (adversarial review, not normal use) and designed around
   from the start for Deep Agents. `AgentFrameworkSession` uses a
   `threading.Lock` from the start for the same reason.
4. **Branching representation?** Not tested against
   `agent_framework_orchestrations`'s `ConcurrentBuilder`/`HandoffBuilder`
   (built on a separate `agent_framework._workflows` Executor graph) in
   this pass -- scoped out, see "What this pass did not verify" below.
   Multi-agent fan-out (`Agent A / Agent B -> Decision`) is verified at
   the level this adapter actually controls: two `Agent.run()` calls
   sharing one `AgentFrameworkSession` via plain `asyncio.gather`, each
   agent's own client instrumented with the same `SevenboxFunctionMiddleware`
   instance (see `tests/test_agent_framework_adapter.py
   ::test_two_agents_running_concurrently_produce_a_correctly_forked_and_
   merged_causal_graph`).
5. **Context/parent propagation?** `contextvars.copy_context()` is called
   once per parallel call, immediately before `asyncio.create_task`
   dispatch (confirmed directly in `_tools.py`'s batch-dispatch code) --
   the same isolation guarantee already verified for Deep Agents' nested
   sessions, so an ambient `ContextVar`-based session is safe under this
   framework's real concurrency without extra locking around the
   contextvar mechanism itself (the lock in `AgentFrameworkSession`
   protects its own mutable fields, a separate concern).
6. **Retries / errors?** Not framework-mediated at the tool-call level
   Sevenbox observes; `FunctionMiddleware.process` could itself retry via
   multiple `call_next()` calls, but this adapter does not -- a retry
   would show up as two ordinary tool-call Warrants, consistent with
   every other adapter in this package.
7. **Replay without re-execution?** Yes, via the same `ReplayEffectExecutor`
   every other adapter uses, keyed on `(source_name, parent_ids)`. Because
   `context.result` can be set without ever calling `call_next()` (question
   2), a replay hit short-circuits the real tool body entirely -- the
   underlying Python function is never called.
8. **What does the framework not expose?** No native "this batch is a
   unit" signal at all (question 1) -- worse than Deep Agents/OpenAI Agents
   SDK, which at least expose *a* whole-batch pre-dispatch hook, and closer
   to CrewAI's per-call-only situation. `AgentFrameworkSession` falls back
   to the same honestly-approximate technique CrewAI's tool-call layer
   already uses in this codebase: an in-flight counter opens a "batch"
   on the first call seen after the previous one fully closed, and every
   call that arrives before that counter returns to zero shares the same
   frozen parent set. See "Known limitation: batch grouping is a
   real-concurrency heuristic, not an exact signal" below for the failure
   mode this does not cover, and why (unlike the PydanticAI case earlier
   in this project) an in-flight counter is the right tool here rather
   than an over-engineered one -- PydanticAI already had an exact signal
   (`run_step`) available and the counter redesign there was solving a
   problem that did not exist; here, no exact signal exists at all.

## A real framework bug: `ChatMiddlewareLayer` silently drops function
## middleware passed through every conventional wiring path

Found by adversarial testing, not normal use: passing a `FunctionMiddleware`
via `Agent(middleware=[...])` (the documented, recommended way -- see
`agent_framework/_middleware.py`'s own comment, "Chat and function
middleware is passed to the chat client at runtime via kwargs in
AgentMiddlewareLayer.run(), where it's properly combined with run-level
middleware") **never fires**, for a client composing `ChatMiddlewareLayer`
above `FunctionInvocationLayer` -- the framework's own prescribed order.
Also confirmed broken: passing the same middleware to the *client's own
constructor* (`SomeClient(middleware=[...])`).

Root cause, read directly in source, not inferred: `ChatMiddlewareLayer
.get_response` does `call_middleware = effective_client_kwargs.pop(
"middleware", [])` unconditionally, uses it only to build its own
chat-only pipeline, and never restores it to `client_kwargs`/`context
.kwargs` before delegating to `super().get_response(...)` --
`FunctionInvocationLayer.get_response`, sitting immediately below it in
the MRO, therefore never receives the popped list, regardless of whether
it contained `FunctionMiddleware` instances or the chat pipeline had any
middleware to run at all. At construction time the same thing happens one
layer up: `ChatMiddlewareLayer.__init__(self, *, middleware=None,
**kwargs)` captures the `middleware` kwarg **by name** for itself
(`self.chat_middleware = list(middleware) if middleware else []`,
unfiltered -- it does not even check the middleware's type) and calls
`super().__init__(**kwargs)` with no `middleware` key left in `kwargs` at
all, so `FunctionInvocationLayer.__init__` (next in MRO) always sees
`middleware=None` regardless of what was originally passed to the
concrete client's constructor.

Verified empirically, with a `ChatMiddlewareLayer + FunctionInvocationLayer
+ ChatTelemetryLayer + BaseChatClient` scripted client and a probe
`FunctionMiddleware`: zero `process()` calls, via either wiring path.
Removing `ChatMiddlewareLayer` from the MRO alone (keeping
`FunctionInvocationLayer + ChatTelemetryLayer + BaseChatClient`) made the
exact same middleware fire correctly for both parallel calls in a batch --
isolating the cause to `ChatMiddlewareLayer` specifically, not something
else in the composition.

**Workaround used by this adapter, also verified empirically**: append
directly to the already-constructed client's own `function_middleware`
list attribute (`client.function_middleware.append(...)`), which
`FunctionInvocationLayer.get_response` reads directly off `self` --
independent of whatever `client_kwargs["middleware"]` did or didn't
survive the trip through `ChatMiddlewareLayer`. `_SevenboxAgentFrameworkRunContext
.instrument(client)` does exactly this, and is the only supported way to
attach this adapter to a client -- `sevenbox_agent_framework_run()` does not
attempt the standard `Agent(middleware=[...])` path at all, because it is
confirmed broken for any client shaped like the framework's own documented
canonical composition. This is a real, verified gap in `agent-framework`
1.13.0 itself, not a Sevenbox adapter limitation -- documented here per this
project's standing discipline of recording real
limits rather than quietly working around them.

## Known limitation: batch grouping is a real-concurrency heuristic, not
## an exact signal

`AgentFrameworkSession.begin_call`/`end_call` implement an in-flight
counter: the first call after the previous batch's counter returned to
zero freezes the current frontier as the new batch's shared parent set;
every call arriving before the counter returns to zero again shares it.
This is correct under the dispatch guarantee `_try_execute_function_call_groups`
actually provides (all of a batch's tasks are created, via `asyncio
.create_task`, in one synchronous loop with no intervening `await`, before
any of them starts running) for the common case verified by
`tests/test_agent_framework_adapter.py
::test_two_parallel_tool_calls_share_the_same_parent_set` (a
`threading.Barrier`-gated deterministic test, 200 trials, matching the
methodology already used for the PydanticAI locking fix). It is **not**
correct in the adversarial case where one call of a genuinely-parallel
batch completes (and the counter returns to zero) before a slower sibling
has even entered `process()` -- that sibling would be mis-attributed as
opening a *new* batch, parented on the first call's already-published
result, inventing a causal dependency between what were really two
independent siblings. This is the same class of race an earlier,
now-reverted PydanticAI design hit -- the difference is that PydanticAI
had an exact identity signal available (`run_step`) that made a counter
unnecessary and therefore wrong to add; Agent Framework exposes no
per-call turn identity to `FunctionMiddleware` at all (question 8), so a
real-concurrency heuristic is the best signal actually available, not a
shortcut around a better one. Not fixed here because it cannot be, short
of the framework exposing a batch identity -- documented as a real,
tested (`test_KNOWN_LIMITATION_a_fast_sibling_finishing_before_a_slow_one_
starts_is_misattributed`) integration limit, not silently assumed away.

## What this pass did not verify

`agent_framework_orchestrations` (`ConcurrentBuilder`, `HandoffBuilder`,
`GroupChatBuilder`, `MagenticBuilder`) is built on a distinct
`agent_framework._workflows` Executor/graph system, confirmed (in
`_concurrent.py`) to wrap each participant as a normal `Agent` -- meaning
this adapter's per-client instrumentation should, in principle, apply to
each participant without orchestration-specific code, IF the orchestrator
dispatches participants from within the same `contextvars.Context` this
adapter's `with` block establishes. That claim is plausible from reading
the source but was **not independently verified against a real running
orchestration** in this pass, unlike everything else in this docstring --
recorded here as an open question for a future adversarial pass, per this
project's discipline of never presenting an unverified claim with the same
confidence as a verified one. The multi-agent fan-out scenario this
adapter *does* verify (question 4) uses plain `asyncio.gather` over two
independently-instrumented `Agent.run()` calls instead.

## Usage

    from agent_framework import Agent
    from sevenbox_adapters.agent_framework_adapter import sevenbox_agent_framework_run

    with sevenbox_agent_framework_run() as krun:
        client = MyRealChatClient(...)
        krun.instrument(client)          # NOT Agent(middleware=[...]) -- see above
        agent = Agent(client=client, name="tester", tools=[lookup_order, check_watchlist])
        result = await agent.run("check order 4471")
        print(krun.session.frontier().live)

`krun.instrument(...)` must be called once per client used inside the
`with` block (once per participant, for a multi-agent scenario sharing one
`krun.session`) -- `SevenboxFunctionMiddleware` itself holds no per-run state,
reading the ambient session from a `ContextVar` at call time, the same
pattern as every other adapter in this package.
"""
from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Any, Optional

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
    from agent_framework import FunctionMiddleware as _FunctionMiddleware
    _AGENT_FRAMEWORK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional extra is absent
    _FunctionMiddleware = object
    _AGENT_FRAMEWORK_AVAILABLE = False


class AgentFrameworkSession:
    """Tracks the causal frontier of one Agent Framework run, shared across
    every client `instrument()`-ed inside the same `with
    sevenbox_agent_framework_run():` block. Thread-safe: `begin_call`/`end_call`
    are guarded by `_lock`, because -- verified, not assumed, see module
    docstring question 3 -- sync tool functions genuinely run on real OS
    threads (`asyncio.to_thread`) under real parallel dispatch.
    """

    def __init__(
        self, *, store: Optional[WarrantStore] = None, replay_from: Optional[WarrantStore] = None
    ) -> None:
        self.store = store if store is not None else WarrantStore()
        self.mode = "replay" if replay_from is not None else "live"
        self._replay_executor = (
            ReplayEffectExecutor(reference_store=replay_from) if replay_from is not None else None
        )
        self._lock = threading.Lock()
        self._frontier: set = set()
        self._batch_parent_ids: Optional[tuple] = None
        self._in_flight: int = 0
        self._batch_new_ids: set = set()

    def start(self, initial_input: Any = None) -> None:
        """Optional: root the run on an explicit axiom before any tool
        fires. If the first tool call arrives without this having been
        called, `begin_call` roots the run on that call's own fallback
        content instead -- never silently parentless."""
        if self._frontier:
            return
        w = create_warrant(
            content=initial_input, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def begin_call(self, fallback_axiom_content: Any = None) -> tuple:
        """Call once at the top of `FunctionMiddleware.process`, before
        `call_next()`. See module docstring's "Known limitation" section
        for exactly what this in-flight-counter heuristic does and does
        not guarantee under real concurrent dispatch."""
        with self._lock:
            if self._in_flight <= 0:
                if not self._frontier:
                    self.start(fallback_axiom_content)
                self._batch_parent_ids = tuple(sorted(self._frontier))
                self._batch_new_ids = set()
                self._in_flight = 0
            self._in_flight += 1
            assert self._batch_parent_ids is not None
            return self._batch_parent_ids

    def end_call(self, warrant_id: str) -> None:
        """Call once at the bottom of `FunctionMiddleware.process`, after
        the call's Warrant has been created and stored."""
        with self._lock:
            self._batch_new_ids.add(warrant_id)
            self._in_flight -= 1
            if self._in_flight <= 0:
                self._frontier = set(self._batch_new_ids)
                self._in_flight = 0
                self._batch_parent_ids = None
                self._batch_new_ids = set()

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


_current_af_session: ContextVar[Optional["AgentFrameworkSession"]] = ContextVar(
    "sevenbox_agent_framework_session", default=None
)


def sevenbox_agent_framework_run(
    *, store: Optional[WarrantStore] = None, replay_from: Optional[WarrantStore] = None
) -> "_SevenboxAgentFrameworkRunContext":
    return _SevenboxAgentFrameworkRunContext(AgentFrameworkSession(store=store, replay_from=replay_from))


class _SevenboxAgentFrameworkRunContext:
    def __init__(self, session: AgentFrameworkSession) -> None:
        self.session = session
        self.function_middleware = SevenboxFunctionMiddleware()
        self._token = None

    def __enter__(self) -> "_SevenboxAgentFrameworkRunContext":
        self._token = _current_af_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_af_session.reset(self._token)
        return False

    def instrument(self, client: Any) -> Any:
        """Attach Sevenbox's `FunctionMiddleware` directly onto a chat
        client's `function_middleware` list attribute. See module
        docstring's "A real framework bug" section for why this bypasses
        `Agent(middleware=[...])` and the client's own constructor
        `middleware=` kwarg entirely, instead of using either. Returns
        `client` for convenient chaining."""
        if not hasattr(client, "function_middleware"):
            raise RuntimeError(
                "This chat client has no `function_middleware` attribute -- "
                "it does not compose agent_framework._tools.FunctionInvocationLayer, "
                "so sevenbox_agent_framework_run() cannot instrument it."
            )
        client.function_middleware.append(self.function_middleware)
        return client


class SevenboxFunctionMiddleware(_FunctionMiddleware):
    """`FunctionMiddleware` that turns every intercepted tool call into a
    Warrant. Stateless and safe to install on multiple clients sharing one
    `AgentFrameworkSession` (multi-agent fan-out, module docstring
    question 4) -- it reads the ambient session from a `ContextVar` at
    call time, the same pattern every other adapter in this package uses.
    """

    def __init__(
        self,
        *,
        source_kind: SourceKind = SourceKind.TOOL,
        epistemic: EpistemicClass = EpistemicClass.VERIFIED,
        reversibility: ReversibilityClass = ReversibilityClass.PURE,
        redaction: Optional[RedactionPolicy] = None,
        origin: Optional[Origin] = None,
    ) -> None:
        if not _AGENT_FRAMEWORK_AVAILABLE:
            raise RuntimeError(
                "SevenboxFunctionMiddleware requires the 'agent-framework' extra "
                "(pip install -e '.[agent-framework]')."
            )
        super().__init__()
        self._source_kind = source_kind
        self._epistemic = epistemic
        self._reversibility = reversibility
        self._redaction = redaction
        self._origin = origin

    async def process(self, context: Any, call_next: Any) -> None:
        session = _current_af_session.get()
        if session is None:
            raise RuntimeError(
                "An Agent Framework function call was intercepted by "
                "SevenboxFunctionMiddleware outside a `with sevenbox_agent_framework_run():` "
                "block. Call `krun.instrument(client)` inside the block, before "
                "the client is used."
            )
        tool_name = context.function.name
        args = dict(context.arguments) if context.arguments else {}
        req_payload = {"args": args}
        parent_ids = list(session.begin_call(fallback_axiom_content={"tool": tool_name}))

        if self._source_kind in PURE_SOURCE_KINDS:
            await call_next()
            response = _function_result_content(context.result)
            w = create_warrant(
                content=response, source_kind=self._source_kind, epistemic=self._epistemic,
                reversibility=self._reversibility, parent_ids=parent_ids, origin=self._origin,
            )
        elif session.mode == "live":
            await call_next()
            response = _function_result_content(context.result)
            w = create_warrant(
                content=response, source_kind=self._source_kind, epistemic=self._epistemic,
                reversibility=self._reversibility, parent_ids=parent_ids, origin=self._origin,
                effect_record=EffectRecord(
                    request=apply_redaction(req_payload, self._redaction),
                    response=apply_redaction(response, self._redaction),
                    source_name=tool_name, redacted=self._redaction is not None,
                ),
            )
        else:
            assert session._replay_executor is not None
            cached = session._replay_executor.invoke(tool_name, tuple(parent_ids), request=req_payload)
            response = cached.content
            w = create_warrant(
                content=response, source_kind=self._source_kind, epistemic=self._epistemic,
                reversibility=self._reversibility, parent_ids=parent_ids, origin=self._origin,
                effect_record=EffectRecord(
                    request=apply_redaction(req_payload, self._redaction),
                    response=apply_redaction(cached.raw, self._redaction),
                    source_name=tool_name, redacted=self._redaction is not None,
                ),
            )
            # Setting context.result without calling call_next() skips the
            # real tool invocation entirely -- see module docstring
            # question 2/7.
            context.result = response

        session.store.add(w)
        session.end_call(w.id)


def _function_result_content(result: Any) -> Any:
    """`context.result` after `call_next()` is always `list[Content]`
    (verified empirically -- see module docstring question 2): the
    framework normalizes whatever the Python tool returned into `Content`
    items inside `tool.invoke()` before any middleware sees it, whether or
    not middleware is even installed. Text-type items are concatenated
    (mirroring `Content.from_function_result`'s own "concatenated text
    from text items" convention); anything else falls back to `str()`."""
    if isinstance(result, list):
        text_parts = [
            item.text for item in result
            if getattr(item, "type", None) == "text" and getattr(item, "text", None) is not None
        ]
        if text_parts:
            return "".join(text_parts)
        if len(result) == 1:
            item = result[0]
            return getattr(item, "result", None) or getattr(item, "text", None) or str(item)
        return [str(item) for item in result]
    return result
