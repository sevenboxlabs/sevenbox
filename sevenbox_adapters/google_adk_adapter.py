"""
sevenbox_adapters.google_adk_adapter -- the "sevenbox wrap" story for Google's
Agent Development Kit (`google-adk` on PyPI).

## Verification strength, stated up front

Real ADK orchestration code runs in-process in every test in this file,
against a hand-rolled `BaseLlm` subclass (network-free, no API key) --
the same class of verification as the first ten adapters in this
package, NOT the weaker, contract-shape-only verification `letta_adapter.py`
had to settle for (Letta's real boundary is HTTP against a server
requiring Postgres, so its own orchestration code never ran in that
adapter's tests). Here, `Runner`/`LlmAgent`'s real dispatch code --
including the real `asyncio.gather` fan-out and the real per-call
correlation logic -- executes genuinely, confirmed by reading tracebacks
that surface real ADK internals (`google/adk/flows/llm_flows/functions.py`,
`google/adk/workflow/_node_runner.py`) when a scripted tool raises.

## The 8 questions, answered from real investigation

Verified against the actual installed `google-adk==2.6.3`.

1. **Where is a decision observed?** `Runner.run(user_id=, session_id=,
   new_message=) -> Generator[Event, None, None]` -- a real, synchronous
   generator (confirmed via `inspect.signature`, not assumed from the
   name). Internally ADK bridges its own async orchestration onto a
   background thread and an internal asyncio loop (a real traceback
   surfaced `Thread-2 (_asyncio_thread_main)` during investigation), but
   this adapter never touches that directly -- it only ever consumes
   `Event` objects via the synchronous `for event in runner.run(...):`
   loop, on whichever thread the caller drives that loop from. `Event`
   is a real, richly-typed Pydantic model (`Event.model_fields`) --
   `content` (a `google.genai.types.Content`, itself a list of `Part`s),
   `actions` (`EventActions` -- `state_delta`/`transfer_to_agent`/...),
   `author`, `invocation_id`, `branch`, `error_code`/`error_message`,
   `id`.
2. **Where is a tool call identified?** `Part.function_call`/
   `Part.function_response`, each carrying its own `.id` -- correlated
   1:1 (`FunctionCall.id` == the matching `FunctionResponse.id`),
   confirmed via `model_fields` on the real installed types, not assumed
   from the name.
3. **Parallel vs. sequential?** Confirmed empirically with a real
   two-tool-call turn: **both** `FunctionCall`s arrive as separate
   `Part`s within a **single** `Event.content.parts` list -- a true
   whole-batch-before-dispatch signal, the same shape as LlamaIndex's
   `AgentOutput.tool_calls`/Letta's `ToolCallMessage.tool_calls`. Their
   results are symmetric: `google/adk/flows/llm_flows/functions.py
   ::merge_parallel_function_response_events` merges every parallel
   call's `FunctionResponse` into a **single** following `Event` too --
   confirmed by reading that function's real source, not assumed from
   its name.

   **A real, adversarially-significant finding that a naive reading of
   the framework would miss**: this whole-batch signal does NOT mean the
   underlying tool bodies actually run in OS-thread parallel. Read
   directly in `functions.py::_execute_single_function_call_async` (the
   function the regular, non-live `run()`/`run_async()` path actually
   uses): each call becomes a real `asyncio.create_task`, and
   `asyncio.gather(*tasks)` awaits them together -- genuine concurrency
   IF the tool itself is an `async def` using non-blocking internal
   `await`s (confirmed empirically: two async tools sleeping 0.4s/0.05s
   via `asyncio.sleep` together took ~0.44s total, matching `max()`, not
   the sum). But for an ordinary **synchronous** tool function -- the
   common case -- `FunctionTool._invoke_callable` calls it directly,
   `return target(**args_to_call)`, with **no thread-pool wrapping at
   all** in this code path (confirmed by reading the source, then
   confirmed empirically: two sync tools sleeping 0.4s/0.05s together
   took ~0.49s, matching the *sum*, and both ran on the *same* thread
   id). `RunConfig.tool_thread_pool_config` exists and does wrap sync
   tools in a real `ThreadPoolExecutor` via `loop.run_in_executor` (with
   `contextvars.copy_context()` captured per call, the same isolation
   guarantee already verified for six other adapters) -- **but it only
   wires into the LIVE/voice-streaming code path**
   (`_execute_single_function_call_live`), never into
   `_execute_single_function_call_async`, the function the regular
   `run()`/`run_async()` path actually calls -- confirmed by setting
   `RunConfig(tool_thread_pool_config=ToolThreadPoolConfig())` and
   re-running the same two-sync-tool probe: identical ~0.49s/same-thread
   result, config or no config. A reasonable person reading
   `RunConfig.tool_thread_pool_config`'s name would assume it applies
   generally; it does not, for the regular agent flow every text-based
   ADK application uses. Documented here because it directly bears on
   this adapter's own concurrency story (see "No lock needed, for a
   fourth reason again" below) and because verifying rather than assuming applies just as much to a framework's own config
   surface as to its documented behavior.
4. **Branching representation?** `Event.branch` and
   `EventActions.transfer_to_agent` exist (sub-agent delegation) but
   were not independently exercised in this pass -- see "What this pass
   did not verify" below.
5. **Context/parent propagation?** Not a concern the way it is for
   adapters with real OS-thread dispatch inside the code path they
   observe: per question 3, the regular (non-live) flow this adapter
   targets never dispatches a sync tool onto a worker thread at all, and
   async-tool concurrency is single-threaded cooperative multitasking
   (one event loop, one thread) -- so there is no framework-internal
   thread boundary for this adapter to isolate via `copy_context()` in
   the first place. This adapter's own event consumption happens
   entirely on whichever single thread drives
   `for event in runner.run(...):`, matching smolagents'/LlamaIndex's
   single-consumer-thread argument, for the same underlying reason (see
   below).
6. **Retries / errors?** A real, precisely-shaped, verified-not-assumed
   behavior: when a tool call raises, `Runner.run()`'s generator does
   **not** propagate the exception to the caller -- it completes
   normally, having yielded an `Event` with `error_code`/`error_message`
   set directly on the event (`error_code='ValueError'`,
   `error_message='boom: hi'` in the probe) and **empty** `content`
   (no `function_response` `Part` at all for the failed call).
   Adversarially extended to the parallel case, mirroring
   `smolagents_adapter.py`'s own discovery process: when one call in a
   two-call batch raises, the OTHER call's result -- which genuinely
   computed successfully -- is **also** silently discarded, never
   appearing in any event. Root cause, read directly rather than
   inferred: `asyncio.gather(*tasks)` (no `return_exceptions=True`) at
   the fan-out site raises on the first task exception, discarding
   whatever the other gathered tasks already returned. The identical
   failure shape smolagents has, for a related but distinct underlying
   reason. Separately, a framework-internal quirk (not a Sevenbox bug, and
   not a sign of lost data -- verified directly, not assumed): ADK's own
   background thread (`Runner.run()`'s sync wrapper spins one up to
   drive its internal asyncio loop, visible in tracebacks as
   `_asyncio_thread_main`) re-raises the same tool exception during its
   own cleanup path (`_cleanup_root_task`/`_drive_root_node`) *after*
   this adapter has already received the correct, complete `Event`
   sequence -- Python's default thread-exception hook then reports it as
   a `PytestUnhandledThreadExceptionWarning`. Confirmed benign by
   directly asserting on this adapter's own observed output in
   `tests/test_google_adk_adapter.py
   ::test_error_produces_no_warrant_and_does_not_crash_the_run` (exactly
   one Warrant -- the axiom -- ends up in the store, nothing leaks and
   nothing is silently lost) rather than assumed safe from the warning's
   absence -- the same "verify, don't let a swallowed warning hide a
   real bug" discipline `letta_adapter.py`'s `ContextVar` finding used,
   just landing on "actually benign" this time instead of "actually a
   bug."
7. **Replay without re-execution?** Not implemented in this pass, for
   the same structural reason as LlamaIndex/OpenHands/smolagents'
   `CodeAgent`: this adapter observes `Event`s after ADK has already run
   the real tool call internally -- there is no live Python call site
   this adapter's own code controls to substitute a cached response at.
   Reaching one would mean wrapping tool functions before registering
   them with `LlmAgent(tools=[...])`, a heavier, different-shaped
   integration not attempted here without first proving it is the right
   boundary.
8. **What does the framework not expose?** Nothing missing for the
   event-stream-consumer design chosen here. What a naive read of the
   framework's own surface WOULD get wrong is `tool_thread_pool_config`'s
   apparent generality (question 3) -- documented precisely so a future
   session does not have to rediscover it.

## No lock needed, for a fourth reason again -- but the same caller-owned-sharing hazard as every prior adapter

`Runner.run()` is a synchronous generator; this adapter's own session
mutation happens exactly once per `Event`, entirely on whichever single
thread drives the `for event in runner.run(...):` loop -- the same
"generator/consumer thread-affinity" argument already verified for
smolagents (a plain sync generator) and LlamaIndex (`asyncio`
thread-affinity), just for ADK's own reason: the regular flow this
adapter targets never dispatches concurrent OS-thread work into code
this adapter observes (question 3). No `threading.Lock` is needed for a
single `runner.run()` call's own events.

**Sharing one `GoogleAdkSession` across genuinely independent, concurrently-driven `runner.run()`/`runner.run_async()` calls is a different, real hazard, verified rather than assumed** -- the same class of check every session-based adapter in this package now gets. `_open_batch` unconditionally overwrites `self._pending_calls`/`self._batch_parent_ids` when a new calls-`Event` arrives, the same destructive-overwrite shape as LlamaIndex's `begin_turn`, not smolagents' additive merge. If agent_b's calls-`Event` arrives on a shared session while agent_a's own batch is still awaiting its response-`Event`, agent_b's overwrite discards agent_a's `_pending_calls` -- when agent_a's real response-`Event` finally arrives, none of its response ids match anything in the (now agent_b-owned) pending dict, so **agent_a's genuinely successful results are silently discarded entirely**, not merely mis-parented. Confirmed with a real test forcing the interleaving deterministically
(`test_KNOWN_LIMITATION_concurrent_sessions_sharing_one_store_lose_the_first_agents_results`).
**Does a lock fix it? Verified with a real `threading.Lock`, not assumed
by analogy to LlamaIndex** -- no: `test_KNOWN_LIMITATION_a_lock_does_not_fix_the_concurrent_session_data_loss_either`
wraps `on_event` in a lock and forces the identical interleaving; the
same data loss still happens, every call fully serialized. The
mechanism is identical to LlamaIndex's: `on_event`'s statements already
ran to completion before the next thread's call began, lock or not --
the corruption is not a torn write, it is that `_open_batch` has no way
to ask "is a different, still-pending batch's bookkeeping about to be
discarded?", because `GoogleAdkSession` carries no per-agent/per-turn
identity to check against. Not added, for the same reason every other
adapter in this package with this exact shape of limitation gives: it
would not actually fix what it looks like it fixes. The safe pattern is
the same as always: one `sevenbox_google_adk_run()` session per real agent
run, or multiple runs consumed sequentially against one shared session.

## Usage

    from google.adk import Runner
    from google.adk.agents import LlmAgent
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from sevenbox_adapters.google_adk_adapter import sevenbox_google_adk_run

    agent = LlmAgent(name="support_agent", model=llm, tools=[...])
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(app_name="app", user_id="u1")
    runner = Runner(app_name="app", agent=agent, session_service=session_service)

    with sevenbox_google_adk_run() as krun:
        krun.session.start("check order 4471")
        for event in runner.run(
            user_id="u1", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="check order 4471")]),
        ):
            krun.observe(event)
        print(krun.session.frontier())

## What this pass did not verify

Sub-agent delegation/branching (`Event.branch`, `EventActions.
transfer_to_agent`) and ADK's own internal `workflow`/node-graph engine
(visible in tracebacks as `_node_runner.py`/`_llm_agent_wrapper.py` --
apparently the real implementation underneath `LlmAgent` as of 2.6.3)
were both found during investigation but not independently exercised.
`Runner.run_async()` (the async-generator sibling of `run()`, usable
with `asyncio.gather` for concurrent multi-agent scenarios within a
single event loop) was not separately tested beyond the caller-owned-
sharing hazard above, which was verified by driving `GoogleAdkSession`
directly rather than through two real concurrent `run_async()` calls.
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
    from google.adk import Event as _Event
    _GOOGLE_ADK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional extra is absent
    _Event = None
    _GOOGLE_ADK_AVAILABLE = False


class GoogleAdkSession:
    """Tracks the causal frontier across one or more `Event`s from a
    Google ADK `Runner.run()`/`run_async()` stream (or several, sharing
    this session -- see module docstring's "No lock needed" section for
    what sharing does and does not make safe).

    Each `Event` carrying `function_call` parts opens a new batch,
    unconditionally overwriting any still-pending batch's bookkeeping
    (see module docstring's known limitation). Each `Event` carrying
    `function_response` parts resolves whatever of the currently-pending
    batch it can correlate by id; ids with no match (a call this session
    never opened a batch for) are ignored defensively rather than
    crashing.
    """

    def __init__(self, *, store: Optional[WarrantStore] = None) -> None:
        if not _GOOGLE_ADK_AVAILABLE:
            raise RuntimeError(
                "GoogleAdkSession requires the 'google-adk' extra "
                "(pip install -e '.[google-adk]')."
            )
        self.store = store if store is not None else WarrantStore()
        self._frontier: set = set()
        self._pending_calls: dict = {}
        self._batch_parent_ids: Optional[tuple] = None

    def start(self, initial_input: Any = None) -> None:
        """Optional: root the run on an explicit axiom before the first
        event is observed. If the first calls-batch arrives without this
        having been called, `_open_batch` roots the run on a fallback
        axiom instead -- never silently parentless."""
        if self._frontier:
            return
        w = create_warrant(
            content=initial_input, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def open_batch(self, calls: list, fallback_axiom_content: Any = None) -> None:
        if not self._frontier:
            self.start(fallback_axiom_content)
        self._batch_parent_ids = tuple(sorted(self._frontier))
        self._pending_calls = {c.id: c for c in calls}

    def resolve_batch(self, responses: list, builder) -> None:
        new_ids = set()
        for response in responses:
            call = self._pending_calls.pop(response.id, None)
            if call is None:
                # A response with no matching call this session opened a
                # batch for -- Sevenbox cannot fabricate provenance for a
                # call it was never shown opening (module docstring
                # question 6/known limitation).
                continue
            w = builder(call, response, self._batch_parent_ids or ())
            self.store.add(w)
            new_ids.add(w.id)
        if new_ids:
            self._frontier = new_ids
        self._pending_calls = {}
        self._batch_parent_ids = None

    def abandon_batch(self) -> None:
        """Call when an error `Event` arrives -- discards whatever was
        pending rather than leaving it stuck forever waiting for a
        response that (per module docstring question 6) will never
        arrive for a failed batch."""
        self._pending_calls = {}
        self._batch_parent_ids = None

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


_current_google_adk_session: ContextVar[Optional["GoogleAdkSession"]] = ContextVar(
    "sevenbox_google_adk_session", default=None
)


def sevenbox_google_adk_run(*, store: Optional[WarrantStore] = None) -> "_SevenboxGoogleAdkRunContext":
    return _SevenboxGoogleAdkRunContext(GoogleAdkSession(store=store))


class _SevenboxGoogleAdkRunContext:
    def __init__(self, session: GoogleAdkSession) -> None:
        self.session = session
        self.observer = SevenboxEventObserver()
        self._token = None

    def __enter__(self) -> "_SevenboxGoogleAdkRunContext":
        self._token = _current_google_adk_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_google_adk_session.reset(self._token)
        return False

    def observe(self, event: Any) -> None:
        """Call once per `Event` while iterating
        `runner.run(...)`/`runner.run_async(...)` inside this context."""
        self.observer.on_event(event)


class SevenboxEventObserver:
    """Stateless; reads the ambient `GoogleAdkSession` from a
    `ContextVar` at call time, the same pattern every other adapter in
    this package uses.
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
        if not _GOOGLE_ADK_AVAILABLE:
            raise RuntimeError(
                "SevenboxEventObserver requires the 'google-adk' extra "
                "(pip install -e '.[google-adk]')."
            )
        self._source_kind = source_kind
        self._epistemic = epistemic
        self._reversibility = reversibility
        self._redaction = redaction
        self._origin = origin

    def on_event(self, event: Any) -> None:
        session = _current_google_adk_session.get()
        if session is None:
            raise RuntimeError(
                "A Google ADK Event was observed by SevenboxEventObserver "
                "outside a `with sevenbox_google_adk_run():` block."
            )
        content = getattr(event, "content", None)
        parts = list(content.parts) if content is not None and content.parts else []
        calls = [p.function_call for p in parts if getattr(p, "function_call", None) is not None]
        responses = [p.function_response for p in parts if getattr(p, "function_response", None) is not None]

        if calls:
            session.open_batch(calls, fallback_axiom_content={"author": getattr(event, "author", None)})
        if responses:
            session.resolve_batch(responses, self._build_warrant)
        if getattr(event, "error_code", None) is not None and not calls and not responses:
            session.abandon_batch()

    def _build_warrant(self, call: Any, response: Any, parent_ids: tuple):
        args = dict(call.args) if getattr(call, "args", None) else {}
        req_payload = {"args": args}
        response_content = _function_response_content(response)

        if self._source_kind in PURE_SOURCE_KINDS:
            return create_warrant(
                content=response_content, source_kind=self._source_kind,
                epistemic=self._epistemic, reversibility=self._reversibility,
                parent_ids=list(parent_ids), origin=self._origin,
            )
        return create_warrant(
            content=response_content, source_kind=self._source_kind,
            epistemic=self._epistemic, reversibility=self._reversibility,
            parent_ids=list(parent_ids), origin=self._origin,
            effect_record=EffectRecord(
                request=apply_redaction(req_payload, self._redaction),
                response=apply_redaction(response_content, self._redaction),
                source_name=call.name, redacted=self._redaction is not None,
            ),
        )


def _function_response_content(response: Any) -> Any:
    """`FunctionResponse.response` is typically `{"result": <value>}`
    (confirmed empirically -- ADK wraps a non-dict return value in a
    `result` key before this adapter ever sees it); unwrap that
    convention when present so the Warrant carries the tool's own
    return value directly rather than an extra wrapper dict, otherwise
    pass the response through as-is."""
    resp = getattr(response, "response", None)
    if isinstance(resp, dict) and set(resp.keys()) == {"result"}:
        return resp["result"]
    return resp
