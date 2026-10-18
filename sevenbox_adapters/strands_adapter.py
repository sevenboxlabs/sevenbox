"""
sevenbox_adapters.strands_adapter -- the "sevenbox wrap" story for AWS Strands
Agents (`strands-agents` on PyPI).

## Verification strength, stated up front

Real Strands orchestration code runs in-process in every test in this
file, against a hand-rolled `Model` subclass emitting the real Bedrock
Converse streaming protocol (`StreamEvent`s -- `messageStart`/
`contentBlockStart`/`contentBlockDelta`/`contentBlockStop`/`messageStop`,
network-free, no API key). The same class of verification as Google ADK
and the first ten adapters, not the weaker contract-shape-only
verification `letta_adapter.py` had to settle for.

## The 8 questions, answered from real investigation

Verified against the actual installed `strands-agents==1.51.0`.

1. **Where is a decision observed?** `Agent(hooks=[...])` -- a real,
   typed hooks registry (`strands.hooks`), confirmed via
   `HookProvider`'s real `Protocol` definition (`register_hooks(registry)`
   calling `registry.add_callback(EventType, callback)`), the same shape
   `openai_agents_adapter.py`'s `SevenboxRunHooks` already uses for that
   framework's own native hook system. `agent.invoke_async(...)` (or the
   sync `agent(...)`) drives the real agent loop; hooks fire as typed
   events during execution -- confirmed via a real probe run before any
   adapter code was written, not assumed from the type names. A hook
   callback's own exception does not propagate raw: Strands' event loop
   catches it and re-raises as `strands.types.exceptions.
   EventLoopException`, with the real exception preserved on
   `.original_exception` -- confirmed empirically
   (`tests/test_strands_adapter.py
   ::test_hook_observed_outside_sevenbox_strands_run_raises_clear_error`,
   whose first version asserted a bare `RuntimeError` and failed with
   the wrapped type instead).
2. **Where is a tool call identified?** `ToolUse.toolUseId`/
   `ToolResult.toolUseId` exist and correlate 1:1 (confirmed via
   `__annotations__` on the real installed `TypedDict`s) -- but this
   adapter does not even need to correlate them across events:
   `AfterToolCallEvent` bundles `tool_use` (the call), `result` (the
   `ToolResult`), and `exception` all in ONE event, confirmed via
   `dataclasses.fields()` on the real type. The cleanest per-call
   correlation of any adapter in this package -- not an id lookup at
   all, just fields on one object.
3. **Parallel vs. sequential?** `BeforeToolsEvent.message` carries the
   COMPLETE list of `toolUse` content blocks for a step, **already
   parsed** (`input` is a real dict, not a raw JSON string needing
   parsing) -- confirmed empirically with a real two-tool-call turn: a
   true whole-batch-before-dispatch signal, the same shape as
   LlamaIndex's `AgentOutput.tool_calls`, but arriving pre-parsed rather
   than needing JSON decoding. Real OS-thread parallelism is the
   **default**, no opt-in configuration needed (unlike Google ADK, where
   the equivalent required `RunConfig.tool_thread_pool_config` and even
   then only worked for the LIVE code path) -- confirmed empirically: two
   tool bodies sleeping 0.4s/0.05s together took ~0.41s (matching
   `max()`, not the sum), running on two distinct real worker thread ids,
   both different from the main thread.
4. **Branching representation?** Not applicable within one agent's own
   step loop. `strands.hooks` also exposes
   `BeforeMultiAgentInvocationEvent`/`AfterMultiAgentInvocationEvent`/
   `BeforeNodeCallEvent`/`AfterNodeCallEvent` (a real multi-agent/graph
   orchestration story, `MultiAgentPlugin`) -- found during investigation
   but not exercised in this pass, see "What this pass did not verify"
   below.
5. **Context/parent propagation?** Real OS-thread dispatch exists for
   tool *bodies* (question 3), but this adapter's own hook callbacks
   never run on those worker threads -- confirmed empirically: recording
   the thread id inside an `AfterToolCallEvent` callback during the same
   two-parallel-tool-call run that proved genuine worker-thread dispatch
   for the tool bodies showed a single, consistent thread id (the main
   thread) for every callback invocation. The same `run_in_executor`-
   style thread-affinity argument already verified for LlamaIndex,
   applied here to hook-callback dispatch instead of an async generator.
   This is what makes `StrandsSession` safe with no lock for a single
   `invoke_async()` call's own events (see "No lock needed" below).
6. **Retries / errors?** A genuinely positive finding, the first of its
   kind across every framework studied in this package so far -- verified
   empirically, not assumed to match precedent: when one call in a
   two-call parallel batch raises, the OTHER call's result is **not**
   discarded. Confirmed with a real batch of `[ok_tool, erroring_tool]`:
   `AfterToolCallEvent` fired independently and correctly for both --
   `ok_tool`'s event carried `result.status='success'` with its real
   content, `erroring_tool`'s carried `result.status='error'` AND
   `exception=ValueError('boom: b')` populated. No exception ever
   escapes `agent.invoke_async()` itself. This is the opposite failure
   shape from smolagents'/Google ADK's shared `asyncio.gather`-without-
   `return_exceptions=True` limitation -- Strands' tool executor
   evidently isolates each call's exception into its own `ToolResult`
   rather than letting one call's failure abort the whole gather.
7. **Replay without re-execution?** Not implemented in this pass, for
   the same structural reason as LlamaIndex/OpenHands/smolagents'
   `CodeAgent`/Google ADK: this adapter observes hook events after
   Strands has already run the real tool call internally -- there is no
   live Python call site this adapter's own code controls to substitute
   a cached response at.
8. **What does the framework not expose?** Nothing missing for the
   hook-consumer design chosen here. `Agent`'s own
   `concurrent_invocation_mode` parameter (default `THROW`) suggests the
   framework itself guards against reusing ONE `Agent` object
   concurrently -- a real safety feature, but a different question from
   this adapter's own "does a shared *session* survive two independent
   Agents run concurrently" check (see "No lock needed" below); not
   relied upon by this adapter either way.

## No lock needed, for the same reason as LlamaIndex -- but the identical caller-owned-sharing hazard as every prior adapter

`StrandsSession` needs no `threading.Lock` for a single agent run's own
events, for the identical reason already verified for LlamaIndex: hook
callbacks are always invoked on the thread driving the agent's own
`await`s, never on the worker thread pool tool bodies actually run on
(question 5).

**Sharing one `StrandsSession` across genuinely independent, concurrently-invoked `Agent`s is a different, real hazard, verified rather than assumed** -- the same class of check every session-based adapter in this package now gets, run precisely because this session's shape (`begin_batch` overwriting `_turn_parent_ids`/`_pending_remaining` unconditionally, freezing a new parent set whenever no batch is currently open) is structurally identical to `LlamaIndexSession`'s, not `SmolagentsSession`'s additive-merge design. Confirmed with a real test forcing the interleaving deterministically
(`test_KNOWN_LIMITATION_concurrent_agents_sharing_one_session_corrupt_each_others_tracking`):
agent_b's batch opening while agent_a's is still pending destructively
overwrites agent_a's pending bookkeeping, the identical shape as
LlamaIndex's `begin_turn`.

**Does a lock fix it? Verified with a real `threading.Lock`, not assumed
by analogy** -- no:
`test_KNOWN_LIMITATION_a_lock_does_not_fix_the_concurrent_session_corruption_either`
wraps the session's mutating methods in a lock and forces the identical
interleaving; the same corruption still happens, every call fully
serialized. `begin_batch`'s statements already ran to completion before
the next thread's call began, lock or not -- the corruption is not a
torn write, it is that `begin_batch` has no way to ask "is a different,
still-pending batch's bookkeeping about to be discarded?", because
`StrandsSession` carries no per-agent/per-turn identity to check
against, the identical conclusion already proven for LlamaIndex and
Google ADK. Not added, for the same reason those two adapters give: it
would not actually fix what it looks like it fixes. The safe pattern is
the same as always: one `sevenbox_strands_run()` session per real agent run,
or multiple runs consumed sequentially against one shared session.

## Usage

    from strands import Agent
    from sevenbox_adapters.strands_adapter import sevenbox_strands_run

    with sevenbox_strands_run() as krun:
        krun.session.start("check order 4471")
        agent = Agent(model=model, tools=[...], hooks=[krun.observer])
        result = await agent.invoke_async("check order 4471")
        print(krun.session.frontier())

## What this pass did not verify

`BeforeMultiAgentInvocationEvent`/`AfterMultiAgentInvocationEvent`/
`BeforeNodeCallEvent`/`AfterNodeCallEvent` (Strands' own multi-agent/graph
orchestration, `MultiAgentPlugin`) and `Agent`'s `concurrent_invocation_mode`
default (`THROW` -- does reusing one `Agent` object concurrently actually
raise, under what conditions) were both found during investigation but
not independently exercised.
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
    from strands.hooks import AfterToolCallEvent as _AfterToolCallEvent
    from strands.hooks import BeforeToolsEvent as _BeforeToolsEvent
    from strands.hooks.registry import HookRegistry as _HookRegistry
    _STRANDS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional extra is absent
    _AfterToolCallEvent = None
    _BeforeToolsEvent = None
    _HookRegistry = None
    _STRANDS_AVAILABLE = False


class StrandsSession:
    """Tracks the causal frontier of one Strands agent run (or several,
    sharing this session -- see module docstring's "No lock needed"
    section for what sharing does and does not make safe).

    Mirrors `LlamaIndexSession`'s shape deliberately: `BeforeToolsEvent`
    gives the full batch size upfront (question 3), exactly like
    LlamaIndex's `AgentOutput.tool_calls`, so `begin_batch` freezes the
    parent set and a known pending count in one call, rather than
    smolagents' incremental one-at-a-time accounting.
    """

    def __init__(self, *, store: Optional[WarrantStore] = None) -> None:
        if not _STRANDS_AVAILABLE:
            raise RuntimeError(
                "StrandsSession requires the 'strands' extra "
                "(pip install -e '.[strands]')."
            )
        self.store = store if store is not None else WarrantStore()
        self._frontier: set = set()
        self._turn_parent_ids: Optional[tuple] = None
        self._pending_remaining: int = 0
        self._new_ids_this_turn: set = set()

    def start(self, initial_input: Any = None) -> None:
        """Optional: root the run on an explicit axiom before the first
        batch opens. If the first batch opens without this having been
        called, `begin_batch` roots the run on that batch's own fallback
        content instead -- never silently parentless."""
        if self._frontier:
            return
        w = create_warrant(
            content=initial_input, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def begin_batch(self, n_calls: int, fallback_axiom_content: Any = None) -> None:
        """Call once per `BeforeToolsEvent`, with the number of `toolUse`
        blocks in `event.message["content"]` -- always freezes the
        current frontier as this batch's shared parent set,
        unconditionally (including for zero calls, a no-op)."""
        if self._turn_parent_ids is not None and self._new_ids_this_turn:
            # A previous batch never fully resolved -- fold what it did
            # produce into the frontier rather than losing its place in it.
            self._frontier = set(self._new_ids_this_turn)
        self._turn_parent_ids = None
        self._new_ids_this_turn = set()
        if n_calls <= 0:
            self._pending_remaining = 0
            return
        if not self._frontier:
            self.start(fallback_axiom_content)
        self._turn_parent_ids = tuple(sorted(self._frontier))
        self._pending_remaining = n_calls

    def turn_parent_ids(self) -> tuple:
        """Read-only snapshot of this batch's frozen parent set."""
        if self._turn_parent_ids is not None:
            return self._turn_parent_ids
        if not self._frontier:
            self.start({"note": "tool call with no preceding BeforeToolsEvent batch"})
        return tuple(sorted(self._frontier))

    def record_new_id(self, warrant_id: str) -> None:
        """Register one tool call's resulting Warrant as this batch's
        output. Closes the batch (folding all of the batch's outputs
        into the new frontier) once every call opened by the matching
        `begin_batch` has resolved."""
        self._new_ids_this_turn.add(warrant_id)
        if self._turn_parent_ids is None:
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


_current_strands_session: ContextVar[Optional["StrandsSession"]] = ContextVar(
    "sevenbox_strands_session", default=None
)


def sevenbox_strands_run(*, store: Optional[WarrantStore] = None) -> "_SevenboxStrandsRunContext":
    return _SevenboxStrandsRunContext(StrandsSession(store=store))


class _SevenboxStrandsRunContext:
    def __init__(self, session: StrandsSession) -> None:
        self.session = session
        self.observer = SevenboxEventObserver()
        self._token = None

    def __enter__(self) -> "_SevenboxStrandsRunContext":
        self._token = _current_strands_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_strands_session.reset(self._token)
        return False


class SevenboxEventObserver:
    """A real `HookProvider` (per `strands.hooks.HookProvider`'s
    `Protocol`) -- pass an instance directly in `Agent(hooks=[...])`.
    Stateless; reads the ambient `StrandsSession` from a `ContextVar` at
    call time, the same pattern every other adapter in this package
    uses.
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
        if not _STRANDS_AVAILABLE:
            raise RuntimeError(
                "SevenboxEventObserver requires the 'strands' extra "
                "(pip install -e '.[strands]')."
            )
        self._source_kind = source_kind
        self._epistemic = epistemic
        self._reversibility = reversibility
        self._redaction = redaction
        self._origin = origin

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        registry.add_callback(_BeforeToolsEvent, self._on_before_tools)
        registry.add_callback(_AfterToolCallEvent, self._on_after_tool_call)

    def _session(self, event_name: str) -> "StrandsSession":
        session = _current_strands_session.get()
        if session is None:
            raise RuntimeError(
                f"A Strands {event_name} was observed by SevenboxEventObserver "
                "outside a `with sevenbox_strands_run():` block."
            )
        return session

    def _on_before_tools(self, event: Any) -> None:
        session = self._session("BeforeToolsEvent")
        content = event.message.get("content", []) if event.message else []
        n_calls = sum(1 for block in content if "toolUse" in block)
        session.begin_batch(n_calls, fallback_axiom_content={"message": event.message})

    def _on_after_tool_call(self, event: Any) -> None:
        session = self._session("AfterToolCallEvent")
        tool_use = event.tool_use
        result = event.result
        parent_ids = list(session.turn_parent_ids())
        response = _tool_result_content(result)

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
                    request=apply_redaction({"args": tool_use.get("input")}, self._redaction),
                    response=apply_redaction(
                        {"content": response, "status": result.get("status") if result else None},
                        self._redaction,
                    ),
                    source_name=tool_use.get("name"), redacted=self._redaction is not None,
                ),
            )

        session.store.add(w)
        session.record_new_id(w.id)


def _tool_result_content(result: Any) -> Any:
    """`ToolResult.content` is `list[ToolResultContent]` -- typically
    `[{"text": "..."}]`. Prefer the plain text of a single text block
    (the common case); fall back to the raw content list otherwise."""
    if not result:
        return None
    content = result.get("content") or []
    if len(content) == 1 and isinstance(content[0], dict) and "text" in content[0]:
        return content[0]["text"]
    return content
