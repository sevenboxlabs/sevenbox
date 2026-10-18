"""
sevenbox_adapters.deep_agents_adapter -- the "sevenbox wrap" story for Deep Agents
(`deepagents` on PyPI, built on `langchain.agents.create_agent`).

Verified against the actual installed `deepagents==0.7.5` and
`langchain==1.3.14` packages, not training-data assumptions -- including
empirical probes (not just source reading) confirming the exact hook
ordering and concurrency model documented below (see "Design" and
tests/test_deep_agents_adapter.py, which drives a real `create_agent()`
graph through a scripted, network-free `BaseChatModel` fake).

## Why this adapter looks structurally different from the other four

Every other adapter in this package (LangGraph, OpenAI Agents SDK,
PydanticAI, CrewAI) wraps individual tool *functions* -- the developer
decorates each one with `@sevenbox_tool` (or, for LangGraph, wraps each graph
*node*). Deep Agents does not need that: `create_deep_agent()` is built on
`langchain.agents.create_agent()`, whose `AgentMiddleware` base class
exposes `after_model()` (fires once per turn, after the model responds,
with the full list of tool calls it just requested -- before any of them
execute) and `wrap_tool_call()` (fires once per individual tool call,
wrapping its actual execution). Installing ONE `SevenboxMiddleware` instance
in `create_deep_agent(middleware=[...])` instruments **every** tool the
agent has -- the built-in filesystem tools (`ls`/`read_file`/`write_file`/
`edit_file`/`glob`/`grep`), `execute` (shell), and the `task` tool
(subagent delegation) -- with no per-tool decoration at all. This is a
consequence of `langchain.agents`' middleware design, not a Sevenbox design
choice; it is documented here because it is a genuine, verified
architectural difference worth recording, not assumed by analogy to the
other four adapters.

## The 8 questions, answered from real investigation

1. **Where is a decision observed?** At `after_model` -- the model's
   response (an `AIMessage`, possibly carrying `tool_calls`) is the
   decision point. Confirmed by probing a real `create_agent()` graph
   with a scripted fake model: `after_model(state, runtime)` receives
   `state["messages"][-1]` as that exact `AIMessage`, with every
   requested tool call already attached, before any executes.
2. **Where is a tool call identified?** `wrap_tool_call(request, handler)`,
   `request.tool_call` is a `{"name", "args", "id"}` dict (confirmed
   against `langgraph.prebuilt.tool_node.ToolCallRequest`'s real
   dataclass fields). `handler(request)` executes the real tool (or, for
   `task`, the real subagent) and returns the `ToolMessage`/`Command`.
3. **Parallel vs. sequential?** Distinguished the same way the OpenAI
   Agents SDK adapter does it: `after_model` sees the *whole* batch of
   `tool_calls` for a turn at once, before dispatch, so the frontier can
   be frozen once per turn rather than chained call-by-call. Verified
   this is necessary, not optional: `langgraph.prebuilt.tool_node.ToolNode
   ._func` dispatches the sync path via
   `langchain_core.runnables.config.get_executor_for_config(config)`,
   which is a real `ContextThreadPoolExecutor` (`.map()` over
   `concurrent.futures.ThreadPoolExecutor`) -- genuine OS-thread
   concurrency for parallel tool calls, confirmed by reading
   `langchain_core/runnables/config.py` directly, not assumed. This is
   the same class of hazard already found once in this project (the
   PydanticAI adapter's unguarded turn-tracking state, found via
   adversarial review) -- `DeepAgentsSession` is built with a
   `threading.Lock` around every shared, mutable, cross-call field from
   the start, specifically because of that precedent.
4. **Branching representation?** Not applicable at the adapter level --
   `create_agent`'s graph is a fixed model/tools loop, not a
   caller-authored branching graph like `langgraph_adapter`'s. Any
   branching in the agent's *behavior* shows up as ordinary sequential
   turns and tool calls, captured the same as any other turn.
5. **Context/parent propagation?** Two mechanisms, verified separately:
   (a) within one turn, `after_model` freezes the live frontier once,
   before the `ThreadPoolExecutor` dispatch begins, and every sibling
   call in that turn reads the same frozen tuple; (b) into a subagent
   (the `task` tool), via `contextvars.ContextVar` -- verified, not
   assumed, that this is safe under real thread-pool parallelism: LangChain's
   own `ContextThreadPoolExecutor.map()` calls `copy_context()` once per
   parallel call *before* dispatch and runs each worker inside its own
   independent copy (`contexts.pop().run(fn, *args)`, read directly from
   `langchain_core/runnables/config.py`), so a `ContextVar.set()` made
   inside one tool call's `wrap_tool_call` (including the nested-session
   swap this adapter does for `task` calls) is invisible to sibling calls
   dispatched in the same batch and never leaks back to the parent --
   this is what makes concurrent, independent subagent delegations safe
   without any extra locking around the contextvar mechanism itself.
6. **Retries / errors?** Not framework-mediated at the tool-call level
   Sevenbox observes; `wrap_tool_call` can itself retry via multiple
   `handler(request)` calls, but this adapter does not do so -- a retry
   would show up as two ordinary tool-call Warrants, same as calling a
   `@sevenbox_tool`-wrapped function twice in any other adapter.
7. **Replay without re-execution?** Yes, for leaf tool calls, via the
   same `ReplayEffectExecutor` every other adapter uses. For the `task`
   tool specifically, replay treats the *entire subagent invocation* as
   one opaque effectful call at the delegation boundary: a cached
   `(source_name="task", parent_ids)` hit is returned directly without
   ever calling `handler(request)`, so the subagent's own internal
   reasoning is never re-run during replay. This is a deliberate,
   documented scope choice, not an oversight -- see "What replay does
   not cover" below.
8. **What does the framework not expose?** No native "this batch of
   tool calls is a unit" identifier beyond the two-hook mechanism
   above (nothing was missing that needed one, unlike CrewAI's hooks,
   which fire per-call with no batch id at all). No way to intercept a
   subagent's *internal* turns without also installing this same
   middleware inside that subagent's own middleware stack (handled by
   passing `SevenboxMiddleware()` in each `SubAgent`/`CompiledSubAgent`
   spec's own `"middleware"` list -- see usage below); a `CompiledSubAgent`
   whose `runnable` was built without this adapter's middleware installed
   is invisible to Sevenbox, by design (Sevenbox cannot instrument a
   already-compiled graph it wasn't given access to, the same limitation
   `sevenbox_adapters.langgraph_adapter` already documents for graphs built
   outside its own `sevenbox_node()` wrapping).

## Usage

    from deepagents import create_deep_agent
    from sevenbox_adapters.deep_agents_adapter import sevenbox_deep_agent_run, SevenboxMiddleware

    with sevenbox_deep_agent_run() as krun:
        agent = create_deep_agent(
            model=model,
            tools=[lookup_order, check_watchlist],
            middleware=[krun.middleware],
            subagents=[{
                "name": "researcher", "description": "...", "system_prompt": "...",
                "tools": [...], "middleware": [krun.middleware],
            }],
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
        print(krun.session.frontier().live)

The same `krun.middleware` instance is safe to pass to the main agent AND
every subagent spec: `SevenboxMiddleware` holds no per-run state of its own,
it only reads the ambient session from a `ContextVar` at call time (same
pattern as `sevenbox_adapters.langgraph_adapter`'s `_current_graph_session`
and `sevenbox_adapters.openai_agents_adapter`'s `_current_agent_session`) --
so the *session* is what's per-run, not the middleware object.

## What replay does not cover

Restated precisely, per this project's discipline of never overclaiming a
limitation into a false guarantee: replaying a Deep Agents run replays
every tool call (including `task`) as a leaf effect, from cached
`effect_record`s in `replay_from`. It does not, and cannot, replay a
subagent's own internal model turns individually -- those were never
captured as separate top-level "replayable calls" from the parent's
perspective in the first place; they were captured as their own nested
Warrants, rooted at that invocation's delegation-start Warrant, in the
*live* run that produced `replay_from`. A replay run reproduces the
outer graph's Warrant set exactly; it does not re-derive or re-verify the
nested subtree, because doing so would require re-running the subagent,
which is precisely what replay exists to avoid.

## Known limitation: compensation does not reach inside a subagent's own
## internal corrections

Found adversarially, not by design review, and documented rather than
silently left for a future reader to rediscover (see
`tests/test_deep_agents_adapter.py`'s
`test_KNOWN_LIMITATION_compensation_does_not_reach_inside_a_subagents_
own_internal_correction`). The outer task-call Warrant's `parent_ids` is
fixed to `(delegation_start.id,)` only -- deliberately, so that a replay
lookup key is knowable before the subagent ever runs (see
`_wrap_delegation`'s own docstring for the earlier, richer-parent design
this reverted from, and exactly why it broke replay). One direct,
uncorrected cost of that choice: `warrant_core.find_transitive_obligations`
walks a warrant's *ancestors* looking for a live superseder: a correction
landing on `delegation_start` itself is reachable (it is a genuine
ancestor of the outer task-call Warrant), but a correction landing on
something *inside* the subagent's own reasoning -- one of its own tool
calls, deeper than `delegation_start` -- is not, because that Warrant is
never cited as a parent of anything outside the subagent's own nested
subtree. A correction there is fully real, stored, and directly queryable
(`store.is_superseded(...)`) -- it just does not automatically surface as
a transitive obligation on an irreversible action the main agent derived
from that delegation's result, the way it would if the same correction
had landed one hop earlier. Not fixed here: doing so soundly would need a
principled way to reconstruct, during replay, what the subagent's own
final live set was without re-running it (e.g. by reading it back out of
`replay_from` directly, since the original live run's nested Warrants are
still present there) -- real, scoped future work, not attempted in this
pass.
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
    from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware
    from langchain_core.messages import ToolMessage as _ToolMessage
    from langgraph.types import Command as _Command
    _DEEP_AGENTS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional extra is absent
    _AgentMiddleware = object
    _ToolMessage = None
    _Command = None
    _DEEP_AGENTS_AVAILABLE = False

DEFAULT_DELEGATION_TOOL_NAME = "task"
"""Name of deepagents' built-in subagent-invocation tool
(`deepagents.middleware.subagents._build_task_tool`). Configurable on
`SevenboxMiddleware` in case a caller renames it via `task_description`
overrides that also change the registered tool name (not done by
`create_deep_agent` itself as of 0.7.5, but not guaranteed never to be)."""


class DeepAgentsSession:
    """Tracks the causal frontier of one Deep Agents run (or one nested
    subagent invocation -- see `SevenboxMiddleware.wrap_tool_call`'s handling
    of the delegation tool). Thread-safe: `record_new_id` and turn-close
    bookkeeping are guarded by `_lock`, because -- verified, not assumed,
    see module docstring point 3 -- `langgraph.prebuilt.tool_node.ToolNode`
    dispatches parallel tool calls across real OS threads for the sync
    path.
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
        self._turn_parent_ids: Optional[tuple] = None
        self._pending_remaining: int = 0
        self._new_ids_this_turn: set = set()

    def start(self, initial_input: Any) -> None:
        """Optional: root the run on an explicit axiom before any tool
        fires. If the first tool call arrives without this having been
        called, begin_turn()/record_call() root the run on that turn's
        own state instead -- never silently parentless."""
        if self._frontier:
            return
        w = create_warrant(
            content=initial_input, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def begin_turn(self, n_tool_calls: int, fallback_axiom_content: Any = None) -> None:
        """Call once from `after_model`, before any of this turn's tool
        calls execute. Always freezes the current frontier as this turn's
        shared parent set -- unlike the OpenAI Agents SDK adapter's
        n<=1 no-op fast path, this adapter freezes unconditionally
        (including for a single tool call) because every call here goes
        through the same `wrap_tool_call` path regardless of batch size;
        a uniform mechanism is simpler and no less correct."""
        with self._lock:
            if self._turn_parent_ids is not None and self._new_ids_this_turn:
                # A previous turn never fully resolved (e.g. a tool raised
                # mid-turn) -- fold what it did produce into the frontier
                # rather than losing those warrants' place in it.
                self._frontier = set(self._new_ids_this_turn)
            self._turn_parent_ids = None
            self._new_ids_this_turn = set()
            self._pending_remaining = 0
            if n_tool_calls <= 0:
                return
            if not self._frontier:
                self.start(fallback_axiom_content)
            self._turn_parent_ids = tuple(sorted(self._frontier))
            self._pending_remaining = n_tool_calls

    def turn_parent_ids(self) -> tuple:
        """Read-only snapshot of this turn's frozen parent set. Safe to
        call from any of the parallel worker threads a turn's tool calls
        may run on: `_turn_parent_ids` is set once by `begin_turn`,
        synchronously, before the executor dispatch that runs those
        threads even begins, and never mutated while they run."""
        parents = self._turn_parent_ids
        if parents is not None:
            return parents
        if not self._frontier:
            self.start({"note": "tool call with no preceding axiom or model turn"})
        return tuple(sorted(self._frontier))

    def record_new_id(self, warrant_id: str) -> None:
        """Register one tool call's resulting Warrant as this turn's
        output. Thread-safe: multiple calls may race here under real
        parallel tool dispatch (module docstring point 3)."""
        with self._lock:
            self._new_ids_this_turn.add(warrant_id)
            if self._turn_parent_ids is None:
                # Not inside a begin_turn()-opened turn (e.g. a call made
                # outside the middleware's normal after_model/tool-node
                # cycle, such as a directly-invoked tool in a test) --
                # publish immediately, one call at a time, same as the
                # other adapters' ambient-frontier fallback.
                self._frontier = {warrant_id}
                return
            self._pending_remaining -= 1
            if self._pending_remaining <= 0:
                self._frontier = set(self._new_ids_this_turn)
                self._turn_parent_ids = None
                self._new_ids_this_turn = set()

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


_current_deep_agents_session: ContextVar[Optional[DeepAgentsSession]] = ContextVar(
    "sevenbox_deep_agents_session", default=None
)


def sevenbox_deep_agent_run(
    *, store: Optional[WarrantStore] = None, replay_from: Optional[WarrantStore] = None
) -> "_SevenboxDeepAgentRunContext":
    return _SevenboxDeepAgentRunContext(DeepAgentsSession(store=store, replay_from=replay_from))


class _SevenboxDeepAgentRunContext:
    def __init__(self, session: DeepAgentsSession) -> None:
        self.session = session
        self.middleware = SevenboxMiddleware()
        self._token = None

    def __enter__(self) -> "_SevenboxDeepAgentRunContext":
        self._token = _current_deep_agents_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_deep_agents_session.reset(self._token)
        return False


class SevenboxMiddleware(_AgentMiddleware):
    """`AgentMiddleware` that turns every tool call (including the
    `task` subagent-delegation tool) into a Warrant. Install the SAME
    instance (or separate instances -- both are safe, see class
    docstring) on the main agent and every subagent spec via
    `middleware=[...]`; it holds no per-run state, reading the ambient
    `DeepAgentsSession` from a `ContextVar` at call time instead (module
    docstring's "Usage" section).

    `source_kind` defaults to `SourceKind.TOOL` for every non-delegation
    call, the same default `sevenbox_node`/`sevenbox_tool` use elsewhere in this
    package -- the middleware cannot know by inspection whether a given
    tool body is genuinely pure.
    """

    def __init__(
        self,
        *,
        source_kind: SourceKind = SourceKind.TOOL,
        epistemic: EpistemicClass = EpistemicClass.VERIFIED,
        reversibility: ReversibilityClass = ReversibilityClass.PURE,
        redaction: Optional[RedactionPolicy] = None,
        origin: Optional[Origin] = None,
        delegation_tool_name: str = DEFAULT_DELEGATION_TOOL_NAME,
    ) -> None:
        if not _DEEP_AGENTS_AVAILABLE:
            raise RuntimeError(
                "SevenboxMiddleware requires the 'deep-agents' extra "
                "(pip install -e '.[deep-agents]')."
            )
        super().__init__()
        self._source_kind = source_kind
        self._epistemic = epistemic
        self._reversibility = reversibility
        self._redaction = redaction
        self._origin = origin
        self._delegation_tool_name = delegation_tool_name

    def after_model(self, state: Any, runtime: Any) -> None:
        session = _current_deep_agents_session.get()
        if session is None:
            return
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        session.begin_turn(
            len(tool_calls),
            fallback_axiom_content={"role": "user_or_prior_state", "messages": len(state["messages"])},
        )

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        session = _current_deep_agents_session.get()
        if session is None:
            raise RuntimeError(
                "A Deep Agents tool call was intercepted by SevenboxMiddleware outside "
                "a `with sevenbox_deep_agent_run():` block. Wrap your agent.invoke(...) "
                "call site, and pass `middleware=[krun.middleware]` to "
                "create_deep_agent()."
            )
        tool_name = request.tool_call["name"]
        args = request.tool_call.get("args", {})

        if tool_name == self._delegation_tool_name:
            return self._wrap_delegation(session, request, handler, tool_name, args)
        return self._wrap_plain_tool(session, request, handler, tool_name, args)

    def _wrap_plain_tool(self, session, request, handler, tool_name, args):
        parent_ids = list(session.turn_parent_ids())
        req_payload = {"args": args}

        if self._source_kind in PURE_SOURCE_KINDS:
            result = handler(request)
            w = create_warrant(
                content=_tool_result_content(result), source_kind=self._source_kind,
                epistemic=self._epistemic, reversibility=self._reversibility,
                parent_ids=parent_ids, origin=self._origin,
            )
        elif session.mode == "live":
            result = handler(request)
            response = _tool_result_content(result)
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
            result = _ToolMessage(content=str(response), tool_call_id=request.tool_call["id"])

        session.store.add(w)
        session.record_new_id(w.id)
        return result

    def _wrap_delegation(self, session, request, handler, tool_name, args):
        parent_ids = list(session.turn_parent_ids())
        req_payload = {"args": args}

        # The decision to delegate is itself a Warrant, constructed BEFORE
        # the subagent runs (and identically in live and replay modes,
        # since content-addressing makes its id a pure function of
        # `parent_ids` and `args`, both known up front) -- two things
        # depend on that: the subagent's own first internal Warrant can
        # cite it as parent (live mode only, see below), and it is what
        # the outer task-call Warrant's `parent_ids` is set to.
        #
        # An earlier version of this method instead set the outer
        # Warrant's parent_ids to `(delegation_start.id, *whatever the
        # subagent's run concluded with)`, for a richer causal trail --
        # reverted after this adapter's own replay test caught the real
        # consequence: ReplayEffectExecutor keys its cache on a Warrant's
        # *stored* parent_ids (warrant_runtime/effects.py), and the whole
        # point of replaying a delegation is to look up that cache
        # *without running the subagent* -- but the subagent's own
        # results were exactly what the richer parent set depended on.
        # That version could never produce a matching replay key by
        # construction. `(delegation_start.id,)` is stable precisely
        # because it never depends on anything the subagent produces.
        delegation_start = create_warrant(
            content={"delegating_to": args.get("subagent_type"), "task": args.get("description")},
            source_kind=SourceKind.DERIVED, epistemic=self._epistemic,
            reversibility=ReversibilityClass.PURE, parent_ids=parent_ids, origin=self._origin,
        )
        session.store.add(delegation_start)
        task_parent_ids = (delegation_start.id,)

        if session.mode == "replay":
            assert session._replay_executor is not None
            cached = session._replay_executor.invoke(tool_name, task_parent_ids, request=req_payload)
            response = cached.content
            w = create_warrant(
                content=response, source_kind=self._source_kind, epistemic=self._epistemic,
                reversibility=self._reversibility, parent_ids=task_parent_ids, origin=self._origin,
                effect_record=EffectRecord(
                    request=apply_redaction(req_payload, self._redaction),
                    response=apply_redaction(cached.raw, self._redaction),
                    source_name=tool_name, redacted=self._redaction is not None,
                ),
            )
            session.store.add(w)
            session.record_new_id(w.id)
            return _ToolMessage(content=str(response), tool_call_id=request.tool_call["id"])

        nested_session = DeepAgentsSession(store=session.store)
        nested_session._frontier = {delegation_start.id}
        token = _current_deep_agents_session.set(nested_session)
        try:
            result = handler(request)
        finally:
            _current_deep_agents_session.reset(token)

        response = _tool_result_content(result)
        w = create_warrant(
            content=response, source_kind=self._source_kind, epistemic=self._epistemic,
            reversibility=self._reversibility, parent_ids=task_parent_ids, origin=self._origin,
            effect_record=EffectRecord(
                request=apply_redaction(req_payload, self._redaction),
                response=apply_redaction(response, self._redaction),
                source_name=tool_name, redacted=self._redaction is not None,
            ),
        )
        session.store.add(w)
        session.record_new_id(w.id)
        return result


def _tool_result_content(result: Any) -> Any:
    """A `wrap_tool_call` handler returns a `ToolMessage` or a `Command`
    (verified against `langchain.agents.middleware.types.AgentMiddleware
    .wrap_tool_call`'s real return-type union). `Command` is what the
    `task` tool actually returns (`deepagents.middleware.subagents
    ._build_task_tool`'s `_return_command_with_state_update`) -- its
    `.update["messages"]` carries the `ToolMessage` a plain tool would
    have returned directly."""
    if _ToolMessage is not None and isinstance(result, _ToolMessage):
        return result.content
    if _Command is not None and isinstance(result, _Command):
        update = result.update or {}
        messages = update.get("messages") if isinstance(update, dict) else None
        if messages:
            last = messages[-1]
            return getattr(last, "content", last)
        return update
    return getattr(result, "content", result)
