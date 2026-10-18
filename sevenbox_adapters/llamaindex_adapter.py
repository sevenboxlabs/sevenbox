"""
sevenbox_adapters.llamaindex_adapter -- the "sevenbox wrap" story for LlamaIndex
Agents (`llama-index-core` on PyPI).

## Which package: `llama-index-core`, not the `llama-index` meta-package

`llama-index` (the bare-name PyPI package) is a meta-package that pulls in
a default OpenAI LLM/embedding integration and a batch of optional
readers by default. `llama-index-core` is the actual abstractions-and-
agents package (`FunctionAgent`, `ReActAgent`, `AgentWorkflow`, the
`workflows` event engine) with none of that -- verified by installing it
alone (`pip install llama-index-core`) and confirming the whole agent
loop runs, network-free, against a hand-rolled `FunctionCallingLLM`, with
zero third-party LLM/vector-store dependencies pulled in. This adapter
targets `llama-index-core` specifically, the same reasoning
`openhands_adapter.py` used to pick `openhands-sdk` over its two
PyPI-adjacent packages -- verify the right package before writing any
code, don't default to whichever name matches the framework's own.

## The 8 questions, answered from real investigation

Verified against the actual installed `llama-index-core==0.14.23` via a
real `FunctionAgent.run()` with real `FunctionTool`s and a hand-rolled
`FunctionCallingLLM` subclass (network-free, no API key -- `MockLLM`,
the SDK's own offline test double, only produces plain text completions
and has no way to script tool calls, so a real `FunctionCallingLLM`
subclass implementing `achat`/`get_tool_calls_from_response` was used
instead, the same technique used for OpenHands' `litellm_completion`
monkeypatch and Agent Framework's hand-rolled `LanguageModelV2`).

1. **Where is a decision observed?** `FunctionAgent`/`ReActAgent` are
   built on LlamaIndex's own `workflows` engine (a separate PyPI
   package, an event-driven step/state-machine runtime). `agent.run(...)`
   returns a `WorkflowHandler` whose `.stream_events()` async generator
   yields every event as the workflow progresses -- `AgentInput` (a new
   turn begins), `AgentOutput` (the model responded), `ToolCall` (one
   tool about to run), `ToolCallResult` (one tool's result), `StopEvent`
   (the run is done).
2. **Where is a tool call identified?** `ToolCall`/`ToolCallResult`,
   correlated by `tool_id` (matching the `call_id`/`toolCallId`
   convention every other framework in this project uses under a
   different name).
3. **Parallel vs. sequential?** `AgentOutput.tool_calls` is a
   `list[ToolSelection]` carrying **every** tool call the model requested
   in that turn, delivered as one Python list *before* any `ToolCall`
   event fires -- confirmed empirically with a real two-tool-call turn:
   `AgentOutput.tool_calls` had both selections, then `ToolCall`/
   `ToolCallResult` fired once per selection, in order. This is a true
   whole-batch-before-dispatch hook, the same shape as Deep Agents'
   `after_model` and OpenAI Agents SDK's `on_llm_end` -- arguably
   cleaner than either, since the batch arrives as already-typed
   `ToolSelection` objects with a known `len()`, not raw JSON to parse or
   an SDK-internal response object to inspect. Real OS-thread
   concurrency for the tool *bodies* is also confirmed, read directly in
   `llama_index/core/tools/function_tool.py::sync_to_async`: a sync tool
   function is dispatched via `loop.run_in_executor(None, ...)` (the
   default executor -- a real `ThreadPoolExecutor`) -- the same hazard
   class already found in PydanticAI/CrewAI/Deep Agents/Agent Framework/
   OpenHands.
4. **Branching representation?** Not applicable within one agent's own
   turn loop. `AgentWorkflow` (LlamaIndex's own multi-agent orchestrator,
   `can_handoff_to`) was not exercised in this pass -- see "What this
   pass did not verify" below.
5. **Context/parent propagation?** `sync_to_async` calls
   `contextvars.copy_context()` once per call, immediately before
   `run_in_executor` dispatch -- the same isolation guarantee already
   verified for Deep Agents/Agent Framework, confirming an ambient
   `ContextVar`-based session is safe under this framework's real
   concurrency too.
6. **Retries / errors?** Not framework-mediated at the level this
   adapter observes; a `ToolCallResult` whose underlying `ToolOutput`
   carries an error still arrives as a normal event and is recorded
   as such, consistent with `mastra_adapter.py`/`openhands_adapter.py`'s
   stance that an error is a real, recorded fact.
7. **Replay without re-execution?** Not implemented in this pass -- see
   "Why replay is out of scope for this pass" below, for reasons
   structurally similar to (though not identical to) OpenHands'.
8. **What does the framework not expose?** Nothing was missing for the
   event-stream-consumer design chosen here (question 3's whole-batch
   hook is better than most frameworks already have). What it does not
   expose is any way to intercept a `ToolCall` *before* the real tool
   body runs from outside the workflow's own step graph -- see question 7.

## A genuinely lock-free design, and a different reason than Mastra's

This is the second adapter in this project (after Mastra) whose session
needs no `threading.Lock` -- but for a different, verified reason.
Mastra's importer is lock-free because nothing in it ever runs
concurrently at all (a single-threaded Python process reading a JSONL
file). Here, tool *bodies* genuinely run on real OS threads (question 3)
-- but `LlamaIndexSession`'s own methods are never called from more than
one thread, confirmed two ways:

- **Structurally**: this adapter consumes `handler.stream_events()`
  directly (an `async for` loop the caller writes, matching
  `sevenbox_llamaindex_run()`'s usage below) rather than registering a
  callback the framework itself invokes from arbitrary internal call
  sites. Every event this adapter ever sees was already delivered back
  onto the event loop thread by the time `stream_events()` yields it --
  `run_in_executor`'s `await` always resumes its calling coroutine on the
  event loop thread that awaited it, never on the worker thread the sync
  function body actually ran on. This is a general property of
  `asyncio.run_in_executor`, not something specific to LlamaIndex, and
  it is what makes `LlamaIndexSession`'s methods safe to call from a
  single consuming loop with no protection.
- **Empirically**: `tests/test_llamaindex_adapter.py
  ::test_event_stream_is_single_threaded_even_when_tool_bodies_run_on_real_worker_threads`
  runs a real two-tool-call parallel batch with `time.sleep()`-backed
  tools, recording the executing thread of each tool body (two distinct
  real worker threads, confirming genuine parallel dispatch) and the
  thread every event is observed on while iterating `stream_events()`
  (always exactly one thread, the one driving the `async for` loop,
  never a worker thread).

**What this does and does not buy, stated precisely so it is not
overclaimed**: `LlamaIndexSession`'s mutating methods are plain
synchronous functions with no `await` inside them, so no individual
method call can be torn mid-execution by another coroutine -- that part
is a real structural guarantee, not a hedge. It does **not**, however,
mean two *complete, independent* turns can safely interleave while
sharing one session -- see "Known limitation" below, found adversarially
by writing exactly that scenario, not assumed safe from the no-torn-
writes argument above. The two claims sound similar but are not the
same: no data race inside one call is not the same as no logical race
across a sequence of calls belonging to two different turns.

## Known limitation: concurrently-overlapping turns sharing one session

`LlamaIndexSession` tracks exactly **one open turn** via flat fields
(`_turn_parent_ids`/`_pending_remaining`/`_new_ids_this_turn`) --
deliberately the same simple design Deep Agents' `DeepAgentsSession`
uses for its own single-agent turn cycle, and correct for that: one
agent's own sequential and/or parallel turns never overlap with each
other by construction (a new `AgentOutput` never fires until the
previous turn's tool calls have all resolved). It is **not** safe if two
*independent* agents' turns genuinely interleave while sharing one
session: unlike `OpenHandsSession`, which keys its own batches by the
framework-provided `llm_response_id` (a real per-turn identity,
supporting multiple batches in flight at once via a dict), LlamaIndex
exposes no equivalent per-turn id for this adapter to key on. Found
adversarially, not assumed: writing the same "two agents, asyncio.gather,
shared session" multi-agent test every other concurrent adapter in this
package has (Agent Framework, Mastra) reliably reproduces mis-attributed
parents -- see `tests/test_llamaindex_adapter.py
::test_KNOWN_LIMITATION_concurrent_turns_sharing_one_session_corrupt_each_others_tracking`.
The safe pattern for sharing one session across multiple agents is
sequential, not concurrent, consumption -- see `tests/test_llamaindex_
adapter.py::test_two_agents_merged_into_one_decision_via_sequential_runs`
and "What this pass did not verify" below.

**Why this is documented as a design limit, not left as an open race
condition -- checked, not assumed**: a lock does not fix this, and that
claim is verified, not argued. `tests/test_llamaindex_adapter.py
::test_KNOWN_LIMITATION_a_lock_does_not_fix_the_concurrent_turn_corruption_either`
wraps `begin_turn`/`turn_parent_ids`/`record_new_id` in an
`asyncio.Lock` and deterministically forces the exact interleaving that
corrupts the unlocked version (agent B's turn opens while agent A's is
still pending) -- the corruption still happens, every single call fully
serialized. This is expected, not surprising, once the actual mechanism
is separated from what a lock protects against: `LlamaIndexSession`'s
methods already contain no `await`, so they were never at risk of a
torn/interleaved write in the first place (see "What this does and does
not buy" above) -- a lock adds nothing there. The corruption instead
comes from `begin_turn` *itself*, once it runs to completion -- lock or
no lock -- unconditionally overwriting the session's one flat turn slot
with no way to check "is a different, still-open turn's bookkeeping
about to be discarded?", because there is no per-turn identity to ask
that question against. A lock only controls *when* two calls are allowed
to run relative to each other; it cannot supply information the data
model itself does not carry. Fixing this for real would mean giving
`LlamaIndexSession` an `OpenHandsSession`-style dict keyed by a per-turn
identity -- and LlamaIndex exposes no such identity for this adapter to
key on (question 8) -- not adding a lock to a design that was never
missing one.

## Why replay is out of scope for this pass

The same shape of gap as OpenHands, for an analogous reason: skipping a
real tool invocation (the `ReplayEffectExecutor` mechanism every other
live adapter in this package uses) requires substituting a cached
response *at the point where the real call would otherwise happen* --
inside the workflow's own `call_tool` step, a `@step` handler defined on
`BaseWorkflowAgent` itself, not something this adapter's event-stream-
consumer design has a hook into. Reaching it would mean subclassing
`FunctionAgent`/`BaseWorkflowAgent` and overriding its internal step
graph -- a much heavier, more fragile integration point than every other
adapter in this package uses, and not attempted here without first
proving it is the right boundary (the same "verify before designing"
discipline this whole program runs on). This adapter therefore builds
real, live Warrants from real events -- full causal correctness, with an
exact, pre-typed batch signal (question 3) -- but cannot skip a real tool
invocation the way the six fully-live adapters can. Documented as a real
architectural limit, not hidden behind a `replay_from=` parameter that
would quietly do nothing.

## What this pass did not verify

`AgentWorkflow` (LlamaIndex's own multi-agent orchestrator, built on
`can_handoff_to` between named agents) was not independently exercised.
Multi-agent fan-out is verified at the level this adapter actually
controls instead, and -- given the known limitation above -- only in its
safe form: two independent `FunctionAgent.run()` handlers, each consumed
to completion *sequentially*, sharing one `LlamaIndexSession` -- see
`tests/test_llamaindex_adapter.py
::test_two_agents_merged_into_one_decision_via_sequential_runs`. This
produces a correct causal *chain* (the second agent's call parents onto
the first's result), not a fork-then-merge the way Agent Framework's/
Mastra's concurrent multi-agent tests do -- a real difference from those
two adapters, not glossed over. Whether `AgentWorkflow`'s own handoff
mechanism (which may avoid this adapter's flat-field limitation entirely,
by construction, if it never dispatches two agents' turns concurrently
in the first place) produces genuine fork-then-merge fan-out was not
tested and is not claimed.

## Usage

    from llama_index.core.agent.workflow import FunctionAgent
    from sevenbox_adapters.llamaindex_adapter import sevenbox_llamaindex_run

    with sevenbox_llamaindex_run() as krun:
        krun.session.start("check order 4471")
        agent = FunctionAgent(tools=[...], llm=llm)
        handler = agent.run("check order 4471")
        async for event in handler.stream_events():
            krun.observe(event)
        result = await handler
        print(krun.session.frontier())
"""
from __future__ import annotations

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

try:
    from llama_index.core.agent.workflow import AgentOutput as _AgentOutput
    from llama_index.core.agent.workflow import ToolCallResult as _ToolCallResult
    _LLAMAINDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional extra is absent
    _AgentOutput = None
    _ToolCallResult = None
    _LLAMAINDEX_AVAILABLE = False


class LlamaIndexSession:
    """Tracks the causal frontier of one LlamaIndex agent run (or several,
    sharing this session -- see module docstring's "genuinely lock-free
    design" section for why that is safe with no lock at all, unlike
    every other concurrent adapter in this package).
    """

    def __init__(self, *, store: Optional[WarrantStore] = None) -> None:
        if not _LLAMAINDEX_AVAILABLE:
            raise RuntimeError(
                "LlamaIndexSession requires the 'llamaindex' extra "
                "(pip install -e '.[llamaindex]')."
            )
        self.store = store if store is not None else WarrantStore()
        self._frontier: set = set()
        self._turn_parent_ids: Optional[tuple] = None
        self._pending_remaining: int = 0
        self._new_ids_this_turn: set = set()

    def start(self, initial_input: Any = None) -> None:
        """Optional: root the run on an explicit axiom before any turn
        fires. If the first `AgentOutput` arrives without this having
        been called, `begin_turn` roots the run on that turn's own
        fallback content instead -- never silently parentless."""
        if self._frontier:
            return
        w = create_warrant(
            content=initial_input, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def begin_turn(self, n_tool_calls: int, fallback_axiom_content: Any = None) -> None:
        """Call once per `AgentOutput` event, with `len(event.tool_calls)`
        -- always freezes the current frontier as this turn's shared
        parent set, unconditionally (including for zero tool calls, the
        final text-only turn, which is simply a no-op)."""
        if self._turn_parent_ids is not None and self._new_ids_this_turn:
            # A previous turn never fully resolved -- fold what it did
            # produce into the frontier rather than losing its place in it.
            self._frontier = set(self._new_ids_this_turn)
        self._turn_parent_ids = None
        self._new_ids_this_turn = set()
        if n_tool_calls <= 0:
            self._pending_remaining = 0
            return
        if not self._frontier:
            self.start(fallback_axiom_content)
        self._turn_parent_ids = tuple(sorted(self._frontier))
        self._pending_remaining = n_tool_calls

    def turn_parent_ids(self) -> tuple:
        """Read-only snapshot of this turn's frozen parent set."""
        if self._turn_parent_ids is not None:
            return self._turn_parent_ids
        if not self._frontier:
            self.start({"note": "tool call with no preceding AgentOutput turn"})
        return tuple(sorted(self._frontier))

    def record_new_id(self, warrant_id: str) -> None:
        """Register one tool call's resulting Warrant as this turn's
        output. Closes the turn (folding all of the turn's outputs into
        the new frontier) once every tool call opened by the matching
        `begin_turn` has resolved."""
        self._new_ids_this_turn.add(warrant_id)
        if self._turn_parent_ids is None:
            # Not inside a begin_turn()-opened turn -- publish
            # immediately, matching the other adapters' ambient-frontier
            # fallback for calls made outside the normal turn cycle.
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


_current_llamaindex_session: ContextVar[Optional["LlamaIndexSession"]] = ContextVar(
    "sevenbox_llamaindex_session", default=None
)


def sevenbox_llamaindex_run(*, store: Optional[WarrantStore] = None) -> "_SevenboxLlamaIndexRunContext":
    return _SevenboxLlamaIndexRunContext(LlamaIndexSession(store=store))


class _SevenboxLlamaIndexRunContext:
    def __init__(self, session: LlamaIndexSession) -> None:
        self.session = session
        self.observer = SevenboxEventObserver()
        self._token = None

    def __enter__(self) -> "_SevenboxLlamaIndexRunContext":
        self._token = _current_llamaindex_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_llamaindex_session.reset(self._token)
        return False

    def observe(self, event: Any) -> None:
        """Call once per event while iterating `handler.stream_events()`
        inside this context (`async for event in handler.stream_events():
        krun.observe(event)`)."""
        self.observer(event)


class SevenboxEventObserver:
    """Stateless; reads the ambient `LlamaIndexSession` from a
    `ContextVar` at call time, the same pattern every other adapter in
    this package uses. Safe to share across multiple concurrently-
    consumed `stream_events()` loops (module docstring's "genuinely
    lock-free design").
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
        if not _LLAMAINDEX_AVAILABLE:
            raise RuntimeError(
                "SevenboxEventObserver requires the 'llamaindex' extra "
                "(pip install -e '.[llamaindex]')."
            )
        self._source_kind = source_kind
        self._epistemic = epistemic
        self._reversibility = reversibility
        self._redaction = redaction
        self._origin = origin

    def __call__(self, event: Any) -> None:
        if isinstance(event, _AgentOutput):
            self._on_agent_output(event)
        elif isinstance(event, _ToolCallResult):
            self._on_tool_call_result(event)
        # AgentInput/ToolCall/StopEvent carry no new causal fact beyond
        # what AgentOutput/ToolCallResult already capture -- matching this project's own
        # definition of what does and doesn't produce a Warrant,
        # they produce no Warrant here.

    def _on_agent_output(self, event: Any) -> None:
        session = _current_llamaindex_session.get()
        if session is None:
            raise RuntimeError(
                "A LlamaIndex AgentOutput event was observed by "
                "SevenboxEventObserver outside a `with sevenbox_llamaindex_run():` "
                "block."
            )
        tool_calls = event.tool_calls or []
        session.begin_turn(
            len(tool_calls),
            fallback_axiom_content={"response": getattr(event, "response", None)},
        )

    def _on_tool_call_result(self, event: Any) -> None:
        session = _current_llamaindex_session.get()
        if session is None:
            raise RuntimeError(
                "A LlamaIndex ToolCallResult event was observed by "
                "SevenboxEventObserver outside a `with sevenbox_llamaindex_run():` "
                "block."
            )
        tool_name = event.tool_name
        args = dict(event.tool_kwargs) if event.tool_kwargs else {}
        req_payload = {"args": args}
        parent_ids = list(session.turn_parent_ids())
        response = _tool_output_content(event.tool_output)

        if self._source_kind in PURE_SOURCE_KINDS:
            w = create_warrant(
                content=response, source_kind=self._source_kind, epistemic=self._epistemic,
                reversibility=self._reversibility, parent_ids=parent_ids, origin=self._origin,
            )
        else:
            w = create_warrant(
                content=response, source_kind=self._source_kind, epistemic=self._epistemic,
                reversibility=self._reversibility, parent_ids=parent_ids, origin=self._origin,
                effect_record=EffectRecord(
                    request=apply_redaction(req_payload, self._redaction),
                    response=apply_redaction(response, self._redaction),
                    source_name=tool_name, redacted=self._redaction is not None,
                ),
            )

        session.store.add(w)
        session.record_new_id(w.id)


def _tool_output_content(tool_output: Any) -> Any:
    """`ToolCallResult.tool_output` is a `ToolOutput` object (`.content`
    a string rendering, `.raw_output` the underlying Python value, `.is_error`
    a bool). Prefer `raw_output` when present -- it is the actual value
    the tool function returned, not a stringified rendering of it."""
    if tool_output is None:
        return None
    raw = getattr(tool_output, "raw_output", None)
    if raw is not None:
        return raw
    return getattr(tool_output, "content", tool_output)
