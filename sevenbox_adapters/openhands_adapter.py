"""
sevenbox_adapters.openhands_adapter -- the "sevenbox wrap" story for the OpenHands
SDK (`openhands-sdk` on PyPI, the actively-developed programmatic core of
the OpenHands project -- github.com/OpenHands/software-agent-sdk).

## A real discovery before any design: which package is "OpenHands"

PyPI has three OpenHands-adjacent packages, verified by downloading and
reading their real metadata before choosing one, not by name-guessing:
`openhands-ai` (legacy monolithic app, "Code Less, Make More", last
released at 1.11.0, dozens of heavyweight runtime/sandbox dependencies),
`openhands` (1.16.0+, "OpenHands CLI - Terminal User Interface", a thin
CLI wrapping the SDK, not a programmatic library), and `openhands-sdk`
(1.17.0 at investigation time, 1.41.0 verified compatible in this
project's own venv -- "OpenHands SDK - Core functionality for building
AI agents", by far the most actively released of the three). This
adapter targets `openhands-sdk` specifically -- the other two are not
what a Sevenbox integration needs, and using either would have been a wrong
guess dressed up as a reasonable default.

## The 8 questions, answered from real investigation

Verified against the actual installed `openhands-sdk` (1.17.0 and,
re-confirmed identical, 1.41.0) via a real `Conversation.run()` with real
`ToolDefinition`/`ToolExecutor` subclasses and litellm's own completion
entry point monkeypatched at `openhands.sdk.llm.llm.litellm_completion`
(module-level name, patched where it is *used*, not where it is defined
-- the standard technique for a name imported via `from x import y as z`)
to return real `litellm.types.utils.ModelResponse` objects with real
`tool_calls`, no network, no API key.

1. **Where is a decision observed?** `Conversation(..., callbacks=[...])`
   -- each callback is `Callable[[Event], None]`, invoked once per event
   as the conversation's own event log grows. This is an event-sourced
   architecture, unlike every other adapter in this package: there is no
   single "here is a whole turn" hook to intercept; instead, a tool call
   is two separate events arriving as two separate callback invocations
   (see question 2).
2. **Where is a tool call identified?** Two paired event types,
   `ActionEvent` (the decision to call a tool: `tool_name`, `tool_call_id`,
   `action` -- the parsed, tool-specific input -- and `llm_response_id`)
   and `ObservationEvent` (the result: `tool_name`, `tool_call_id`,
   `observation`, and `action_id` -- an explicit pointer back to the
   `ActionEvent.id` that produced it). **This is the cleanest per-call
   correlation of any framework investigated in this project's
   adversarial-validation program**: every other adapter has had to
   *derive* request/response pairing from a shared `call_id`/`toolCallId`
   convention; here the framework hands over an explicit foreign key
   (`action_id`) in addition to the shared `tool_call_id`, needing no
   inference at all.
3. **Parallel vs. sequential?** `ActionEvent.llm_response_id` is
   **identical** across every action requested in the same model
   response and different across separate responses -- confirmed
   empirically with a real two-tool-call turn (both actions carried
   `llm_response_id="resp1"`, the following turn's `MessageEvent` carried
   `"resp2"`). This is an *exact* batch-identity signal, not a heuristic
   -- unlike PydanticAI's `run_step` (which needed `run_id` added
   alongside it after an adversarial bug) or Agent Framework's in-flight
   counter (a real-concurrency approximation with a documented gap),
   `llm_response_id` alone is sufficient to group siblings correctly with
   zero timing dependence. Real concurrent dispatch of TOOL BODIES is
   also confirmed, read directly in `openhands/sdk/agent/parallel_
   executor.py`: `ParallelToolExecutor.execute_batch` runs each action's
   tool call through a real `concurrent.futures.ThreadPoolExecutor`
   whenever `Agent(tool_concurrency_limit=N)` is set above its default of
   `1`. **This does not, however, mean `OpenHandsSession`'s own
   callback-facing methods are ever called from more than one thread --
   verified, not assumed, and this is a materially different situation
   from PydanticAI/CrewAI/Deep Agents/Agent Framework, not the same
   hazard restated.** `_execute_action_event` (which runs ON the worker
   thread) says so directly in its own docstring in the installed SDK:
   "Events are NOT emitted here -- the caller is responsible for emitting
   them in order." `_ActionBatch.emit()` is a plain sequential `for`
   loop, called exactly once, from the thread that called `Conversation
   .run()`, only *after* `execute_batch`'s `with ThreadPoolExecutor(...)
   as executor:` block has already exited (which blocks until every
   worker has finished, before any `future.result()` is read) -- so by
   the time any event reaches `SevenboxConversationCallback.__call__`, every
   worker thread for that batch has already completed and been joined.
   Confirmed empirically, not just by reading source:
   `tests/test_openhands_adapter.py::test_event_emission_is_single_
   threaded_even_when_tool_bodies_run_on_real_worker_threads` runs a real
   two-tool-call parallel batch under `tool_concurrency_limit=2`,
   recording the executing thread of each tool body (two distinct real
   worker threads, confirming real parallel dispatch genuinely happened)
   *and* the thread every callback invocation lands on (always exactly
   one thread, always the one that called `Conversation.run()`, never a
   worker thread). **So why does `OpenHandsSession` still carry a
   `threading.Lock`?** Not to guard against `ParallelToolExecutor`'s
   workers reaching the session -- that is structurally impossible by
   the above. The lock is defensive against a different, plausible but
   *not independently verified in this pass* scenario: multiple
   `Conversation`s sharing one `OpenHandsSession` via a common `store=`
   (the same shared-session pattern this adapter's multi-agent fan-out
   note, below, already flags as unverified) -- if two `Conversation`s
   each ran on their own thread and both happened to be mid-batch at the
   same time, *their own* single-threaded emission loops could still
   collide with each other inside one shared session. Kept as cheap,
   still-plausible insurance for that case; not kept because of a hazard
   found inside a single `Conversation`'s own execution, because there
   isn't one.
4. **Branching representation?** Not applicable at the adapter level --
   `Conversation`'s loop is a fixed action/observation cycle, not a
   caller-authored branching graph. Multi-agent fan-out was not
   independently verified in this pass (see "What this pass did not
   verify" below).
5. **Context/parent propagation?** Entirely within `OpenHandsSession`,
   protected by its lock: a new `llm_response_id` freezes the current
   frontier as that batch's shared parent set; every `ActionEvent`
   sharing the previous `llm_response_id` reuses it; the batch closes
   (folding its own outputs into the new frontier) once every action
   opened for that response has been resolved by a matching
   `ObservationEvent` (or `AgentErrorEvent`, see question 6).
6. **Retries / errors?** `AgentErrorEvent` (`tool_name`, `tool_call_id`,
   `error`) is handled the same way `ObservationEvent` is -- it still
   closes out the pending action and still produces a Warrant, with
   `content=None` and the error text preserved in `effect_record
   .response`, consistent with `sevenbox_adapters.mastra_adapter`'s stance
   that an error is a real, recorded fact, not a silently dropped one.
   Not separately adversarially stress-tested against a real forced
   tool failure in this pass -- flagged as untested, not claimed as
   verified, the same honesty standard `mastra_adapter.py` holds itself
   to for its own untested error path.
7. **Replay without re-execution?** **Not implemented in this pass --
   a real, verified scope limit, not an oversight.** See "Why replay is
   out of scope for this pass" below.
8. **What does the framework not expose?** The one genuine gap found:
   `ToolExecutor.__call__(self, action, conversation=None)` receives the
   parsed `Action` object and the live `Conversation`, but *not* the
   action's own `id`, `tool_call_id`, or `llm_response_id` -- confirmed
   directly against `Action`'s own schema (`Action.model_fields` is
   empty; only tool-specific subclass fields exist). The executor layer
   and the event-callback layer are two independently-registered hook
   points with no framework-provided way to correlate "this executor
   invocation" with "this ActionEvent" from inside the executor alone.
   This is exactly why question 7's answer is what it is.

## Why replay is out of scope for this pass

Every live adapter in this package (LangGraph, OpenAI Agents SDK,
PydanticAI, CrewAI, Deep Agents, Agent Framework) implements replay the
same way: a `ReplayEffectExecutor`, keyed on `(source_name, parent_ids)`,
sits *at the tool-call site itself* and can substitute a cached response
*before* the real tool body ever runs. That requires knowing a call's
`parent_ids` (hence its batch identity) at the exact point where the
real invocation would otherwise happen -- inside `ToolExecutor.__call__`
here. Question 8 is why that is not available: `ToolExecutor.__call__`
has no visibility into `llm_response_id`/`tool_call_id` at all, and the
one place that information does exist (`ActionEvent`, delivered via the
separate `callbacks=` hook) cannot pass it to the executor through any
mechanism this SDK exposes. A thread-local or contextvar bridge between
the two hooks was considered and rejected *unverified* -- fabricating
that coordination without empirical proof it survives real concurrent
dispatch (`tool_concurrency_limit > 1`, question 3) would be exactly the
kind of "force it without verifying" this project's discipline warns
against. This
adapter therefore builds real, live Warrants from real events -- full
causal correctness, with the cleanest per-call identity of any framework
studied so far (question 2) -- but cannot skip a real tool invocation the
way the other six adapters can. Recorded here as a real, honest
architectural limit of this integration, not hidden behind a
`replay_from=` parameter that would quietly do nothing.

## What this pass did not verify

Multi-agent fan-out (the `Agent -> Agent A / Agent B -> Decision` shape
exercised for Agent Framework and Mastra) was not independently built or
tested for OpenHands in this pass -- `OpenHandsSession` accepts a shared
`store=` exactly like every other adapter's session, so importing two
independent `Conversation` runs' events into one store and citing both
`.frontier()` results as a Decision's parents should work by the same
construction already proven elsewhere in this package, but that claim is
not backed by a real test here and is not presented with the same
confidence as the verified findings above.

## A tool-authoring gotcha, verified against the real SDK, worth flagging
## for anyone registering their own tools alongside this adapter

`ToolDefinition.name` is a `ClassVar[str]`, auto-derived by
`__init_subclass__` from the tool class's own name (`FooTool` ->
`_camel_to_snake("FooTool")` -> `"foo_tool"` -> `.removesuffix("_tool")`
-> `"foo"`) *unless* the subclass sets `name` explicitly in its own class
body. Passing `name=...` to the constructor inside `create()` does
**not** override it, because a `ClassVar` is not a pydantic model field
at all -- confirmed the hard way, not by reading docs: a test tool class
named `BrokenTool` silently registered and resolved as `"broken"`, not
the intended `"broken_tool"`, because `_camel_to_snake` only strips one
trailing `"_tool"` and `"BrokenTool"` -> `"broken_tool"` -> stripped ->
`"broken"`. Nothing to do with this adapter specifically -- recorded here
because it will bite the first person who writes a tool class whose name
does not happen to already end in a word that survives the strip (as
`LookupOrderTool` -> `"lookup_order"` does, by naming-convention luck).

## Usage

    from openhands.sdk import Agent, Conversation, LLM
    from sevenbox_adapters.openhands_adapter import sevenbox_openhands_run

    with sevenbox_openhands_run() as krun:
        krun.session.start("check order 4471")
        conversation = Conversation(agent=agent, callbacks=[krun.callback])
        conversation.send_message("check order 4471")
        conversation.run()
        print(krun.session.frontier())
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

try:
    from openhands.sdk.event import ActionEvent as _ActionEvent
    from openhands.sdk.event import AgentErrorEvent as _AgentErrorEvent
    from openhands.sdk.event import ObservationEvent as _ObservationEvent
    _OPENHANDS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional extra is absent
    _ActionEvent = None
    _AgentErrorEvent = None
    _ObservationEvent = None
    _OPENHANDS_AVAILABLE = False


class OpenHandsSession:
    """Tracks the causal frontier of one OpenHands `Conversation`.

    Carries a `threading.Lock`, but -- verified empirically, not assumed,
    see module docstring question 3 -- **not** because `ParallelToolExecutor`
    reaches this session from multiple threads; it never does, by the
    SDK's own construction (`_ActionBatch.emit()` calls back into this
    session's methods sequentially, from a single thread, only after
    every worker thread for a batch has already been joined). Batch
    *grouping* also needs no heuristic (`ActionEvent.llm_response_id` is
    an exact signal, unlike every other concurrent adapter in this
    package). The lock is kept as cheap insurance for a different,
    plausible-but-unverified scenario instead: multiple `Conversation`s
    sharing one `OpenHandsSession` (a common `store=`) concurrently --
    see module docstring question 3 for the full analysis.
    """

    def __init__(self, *, store: Optional[WarrantStore] = None) -> None:
        self.store = store if store is not None else WarrantStore()
        self._lock = threading.Lock()
        self._frontier: set = set()
        self._last_llm_response_id: Optional[str] = None
        self._batch_parent_ids: Optional[tuple] = None
        self._open_in_batch: int = 0
        self._new_ids_this_batch: set = set()
        self._pending: dict = {}  # tool_call_id -> (parent_ids, tool_name, args)

    def start(self, initial_input: Any = None) -> None:
        """Optional: root the run on an explicit axiom before any action
        fires. If the first action arrives without this having been
        called, `on_action` roots the run on that action's own fallback
        content instead -- never silently parentless."""
        if self._frontier:
            return
        w = create_warrant(
            content=initial_input, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def on_action(
        self, *, tool_call_id: str, tool_name: str, args: Any,
        llm_response_id: Optional[str], fallback_axiom_content: Any = None,
    ) -> None:
        """Call once per `ActionEvent`, in the order the framework
        delivers them (before that action's own tool body has
        necessarily finished -- see module docstring question 7 for why
        this is observation-only, not a short-circuit point)."""
        with self._lock:
            if self._batch_parent_ids is None or llm_response_id != self._last_llm_response_id:
                if not self._frontier:
                    self.start(fallback_axiom_content)
                self._batch_parent_ids = tuple(sorted(self._frontier))
                self._last_llm_response_id = llm_response_id
                self._open_in_batch = 0
                self._new_ids_this_batch = set()
            self._pending[tool_call_id] = (self._batch_parent_ids, tool_name, args)
            self._open_in_batch += 1

    def resolve_pending(self, tool_call_id: str, fallback_tool_name: str) -> tuple:
        """Call once per `ObservationEvent`/`AgentErrorEvent`. Returns
        `(parent_ids, tool_name, args)` for the matching `on_action`
        call; falls back to the current frontier (with `args=None`) if
        no matching pending action is found -- e.g. an observation
        delivered outside this session's own `on_action` bookkeeping --
        so a call is never silently dropped."""
        with self._lock:
            entry = self._pending.pop(tool_call_id, None)
            if entry is not None:
                return entry
            return tuple(sorted(self._frontier)), fallback_tool_name, None

    def close_call(self, warrant_id: str) -> None:
        """Call once the Warrant for a resolved action has been created
        and stored. Closes the batch (folding its outputs into the new
        frontier) once every action opened for the current
        `llm_response_id` has been resolved."""
        with self._lock:
            self._new_ids_this_batch.add(warrant_id)
            self._open_in_batch -= 1
            if self._open_in_batch <= 0:
                self._frontier = set(self._new_ids_this_batch)
                self._batch_parent_ids = None
                self._new_ids_this_batch = set()
                self._open_in_batch = 0

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


_current_openhands_session: ContextVar[Optional["OpenHandsSession"]] = ContextVar(
    "sevenbox_openhands_session", default=None
)


def sevenbox_openhands_run(*, store: Optional[WarrantStore] = None) -> "_SevenboxOpenHandsRunContext":
    return _SevenboxOpenHandsRunContext(OpenHandsSession(store=store))


class _SevenboxOpenHandsRunContext:
    def __init__(self, session: OpenHandsSession) -> None:
        self.session = session
        self.callback = SevenboxConversationCallback()
        self._token = None

    def __enter__(self) -> "_SevenboxOpenHandsRunContext":
        self._token = _current_openhands_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_openhands_session.reset(self._token)
        return False


class SevenboxConversationCallback:
    """Pass as one of `Conversation(callbacks=[...])`. Stateless and safe
    to share across multiple `Conversation`s -- it reads the ambient
    `OpenHandsSession` from a `ContextVar` at call time, the same pattern
    every other adapter in this package uses.
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
        if not _OPENHANDS_AVAILABLE:
            raise RuntimeError(
                "SevenboxConversationCallback requires the 'openhands' extra "
                "(pip install -e '.[openhands]')."
            )
        self._source_kind = source_kind
        self._epistemic = epistemic
        self._reversibility = reversibility
        self._redaction = redaction
        self._origin = origin

    def __call__(self, event: Any) -> None:
        if isinstance(event, _ActionEvent):
            self._on_action_event(event)
        elif isinstance(event, _ObservationEvent):
            self._on_resolution(
                tool_call_id=event.tool_call_id, tool_name=event.tool_name,
                response=_dump(event.observation),
            )
        elif isinstance(event, _AgentErrorEvent):
            self._on_resolution(
                tool_call_id=event.tool_call_id, tool_name=event.tool_name,
                response=None, error=event.error,
            )
        # SystemPromptEvent, MessageEvent, and everything else are not
        # causal tool-call facts -- matching this project's own definition
        # of what does and doesn't produce a Warrant (chatter/bookkeeping),
        # they produce no Warrant here.

    def _on_action_event(self, event: Any) -> None:
        session = _current_openhands_session.get()
        if session is None:
            raise RuntimeError(
                "An OpenHands ActionEvent was intercepted by "
                "SevenboxConversationCallback outside a `with sevenbox_openhands_run():` "
                "block."
            )
        session.on_action(
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            args=_dump(event.action),
            llm_response_id=event.llm_response_id,
            fallback_axiom_content={"tool": event.tool_name},
        )

    def _on_resolution(
        self, *, tool_call_id: str, tool_name: str, response: Any, error: Optional[str] = None,
    ) -> None:
        session = _current_openhands_session.get()
        if session is None:
            raise RuntimeError(
                "An OpenHands Observation/Error event was intercepted by "
                "SevenboxConversationCallback outside a `with sevenbox_openhands_run():` "
                "block."
            )
        parent_ids, recorded_tool_name, args = session.resolve_pending(tool_call_id, tool_name)
        content = response if error is None else None
        req_payload = {"args": args}
        effective_response = response if error is None else {"error": error}

        if self._source_kind in PURE_SOURCE_KINDS:
            w = create_warrant(
                content=content, source_kind=self._source_kind, epistemic=self._epistemic,
                reversibility=self._reversibility, parent_ids=parent_ids, origin=self._origin,
            )
        else:
            w = create_warrant(
                content=content, source_kind=self._source_kind, epistemic=self._epistemic,
                reversibility=self._reversibility, parent_ids=parent_ids, origin=self._origin,
                effect_record=EffectRecord(
                    request=apply_redaction(req_payload, self._redaction),
                    response=apply_redaction(effective_response, self._redaction),
                    source_name=recorded_tool_name or tool_name,
                    redacted=self._redaction is not None,
                ),
            )

        session.store.add(w)
        session.close_call(w.id)


def _dump(model: Any) -> Any:
    """`event.action`/`event.observation` are pydantic models (tool-
    specific `Action`/`Observation` subclasses) -- `.model_dump()` turns
    them into plain, JSON-shaped content for a Warrant."""
    if model is None:
        return None
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model
