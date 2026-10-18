"""
sevenbox_adapters.smolagents_adapter -- the "sevenbox wrap" story for Smolagents
(`smolagents` on PyPI, Hugging Face's agent framework).

## The 8 questions, answered from real investigation

Verified against the actual installed `smolagents==1.26.0` via a real
`ToolCallingAgent`/`CodeAgent` with real `@tool`-decorated functions and a
hand-rolled `Model` subclass (network-free, no API key -- the same
technique every prior adapter in this package uses; smolagents ships no
offline scripted-response test double of its own).

1. **Where is a decision observed?** `MultiStepAgent.run(task, stream=True)`
   returns a generator; iterating it yields fine-grained events as a step
   executes -- `ToolCall` (one call about to dispatch), `ToolOutput` (one
   call resolved), `ActionOutput`, and the fully-resolved `ActionStep`
   itself at the end of each step. `stream=False` (the default) hides all
   of this and returns only the final answer -- confirmed by reading
   `MultiStepAgent.run`'s docstring and `_run_stream`'s implementation,
   not assumed from the name.
2. **Where is a tool call identified?** `ToolCall.id` / `ToolOutput.id`,
   correlated 1:1 -- `ToolOutput.tool_call` is a direct back-reference to
   the originating `ToolCall` object (no separate id-matching dict
   needed, unlike OpenHands' `action_id`/`llm_response_id` pair). Verified
   `ToolOutput`'s actual fields via `dataclasses.fields()` against the
   installed package rather than assumed from the name: `id`, `output`
   (the **raw, non-stringified** Python return value), `is_final_answer`,
   `observation` (the stringified rendering fed back to the model),
   `tool_call`.
3. **Parallel vs. sequential?** Confirmed empirically with a real
   two-tool-call turn (`probe_stream.py` during investigation): **both**
   `ToolCall` events for a step are yielded before **either** `ToolOutput`
   arrives -- a true whole-batch-before-dispatch signal, the same shape
   Deep Agents/OpenAI Agents SDK/LlamaIndex already have, except here it
   arrives incrementally (one `ToolCall` at a time) rather than as a
   single pre-counted list, so this adapter's session tracks "batch is
   open" as a boolean rather than a known `n_tool_calls` count (see
   `SmolagentsSession` below). Real OS-thread parallelism for the tool
   *bodies* is also confirmed, read directly in
   `smolagents/agents.py::ToolCallingAgent.process_tool_calls`: when
   `len(parallel_calls) > 1`, dispatch goes through
   `with ThreadPoolExecutor(self.max_tool_threads) as executor:` with
   `copy_context()` captured per call before `executor.submit(ctx.run, ...)`
   -- verified empirically too (`probe_stream.py` recorded two distinct
   real OS thread ids for two concurrently-running tool bodies, both
   different from the main thread).
4. **Branching representation?** Not applicable within one agent's own
   step loop -- no multi-agent orchestration primitive was exercised in
   this pass (smolagents supports agent-as-tool composition via
   `managed_agents`, not independently investigated here).
5. **Context/parent propagation?** `copy_context()` per call, immediately
   before `ThreadPoolExecutor.submit` -- the same isolation guarantee
   already verified for Deep Agents/Agent Framework/LlamaIndex, confirming
   an ambient `ContextVar`-based session is safe under this framework's
   real concurrency too.
6. **Retries / errors?** A real, adversarially-discovered framework
   behavior, not a Sevenbox bug: when any call in a parallel batch raises, the
   framework's own `process_tool_calls` generator raises out of the
   `as_completed(futures)` loop *before* yielding a `ToolOutput` for
   **any** call in that batch -- confirmed empirically
   (`probe_partial_error.py`) with a batch of `[slow_ok_tool, fast_fail_tool]`:
   `slow_ok_tool` genuinely ran to completion (its `time.sleep(0.5)` shows
   up in the step's 0.51s duration) but its result is silently discarded
   by the framework itself -- no `ToolOutput` for it ever reaches this
   adapter's stream. See "Known limitation: partial-batch failures" below.
   Separately, a framework-internal quirk (not a Sevenbox bug) surfaces when a
   consumer abandons the `stream=True` loop by letting an exception
   propagate out of it (as `tests/test_smolagents_adapter.py
   ::test_tool_called_outside_sevenbox_smolagents_run_raises_clear_error`
   deliberately does): Python's GC later sends `GeneratorExit` into the
   abandoned generator to close it, and smolagents' own generator ignores
   it (a `PytestUnraisableExceptionWarning`, not a test failure) -- noted
   here because it was observed during this adapter's own testing, not
   because this adapter can do anything about it.
7. **Replay without re-execution?** Not implemented in this pass -- see
   "Why replay is out of scope for this pass" below.
8. **What does the framework not expose?** `CodeAgent` (writes and
   executes Python code as its tool-invocation mechanism, rather than
   discrete JSON tool calls) exposes only one synthetic `ToolCall` per
   step -- `name="python_interpreter"`, `arguments=<the raw code string>`
   -- and **no `ToolOutput` at all**; confirmed empirically
   (`probe_codeagent.py`). Any real tool functions the generated code
   calls execute *inside* the sandboxed interpreter, invisible to the
   event stream this adapter observes. `CodeAgent` is therefore out of
   scope for per-real-tool-call Warrant construction -- see "CodeAgent is
   out of scope" below, the same honest-limit discipline as `CodeAgent`
   not exposing a boundary this adapter's design can reach, matching how
   Mastra/OpenHands' own structural gaps were handled.

## No `@sevenbox_tool` wrapper needed -- a first among this package's adapters

Every prior adapter in this package needs a decorator or middleware
wrapped around the *tool function itself* to capture its exact inputs and
raw return value at the call site. This one does not: `ToolOutput.output`
already **is** the tool's raw Python return value, confirmed empirically
(`probe_rawtype.py`, a tool returning a real `dict` came back as
`ToolOutput.output` with `type(...) is dict`, while `.observation` held
the stringified rendering) -- so a pure stream-observer design loses
nothing relative to wrapping the function, and needs no `sevenbox_tool()`
export at all. Kept anyway is `sevenbox_smolagents_run()`, matching every
other adapter's naming convention for the run-scoped context manager.

## A genuinely lock-free design, and a third distinct reason (after Mastra, LlamaIndex)

Mastra's importer is lock-free because nothing in it ever runs
concurrently (single-threaded log replay). LlamaIndex's is lock-free
because `run_in_executor`'s `await` always resumes on the event-loop
thread, never a worker thread. This adapter's reason is simpler than
either: `agent.run(task, stream=True)` returns a **plain synchronous
generator**, not an async one -- `_step_stream`'s `yield` statements only
ever execute on whichever thread calls `next()` on the generator (a basic
property of Python generators, not specific to asyncio at all). The
`ThreadPoolExecutor` workers inside `process_tool_calls` compute tool
results, but the `yield future.result()` that hands each `ToolOutput`
back to this adapter's consuming `for` loop always happens on the single
thread driving that loop. Verified empirically, the same way LlamaIndex's
claim was verified rather than assumed:
`tests/test_smolagents_adapter.py
::test_event_stream_is_single_threaded_even_when_tool_bodies_run_on_real_worker_threads`
records the executing thread of each tool body (two distinct real worker
threads) and the thread every stream event is observed on (always exactly
the one driving the `for event in ... stream=True` loop).

## Known limitation: partial-batch failures silently drop successful siblings

Documented as an **inherited** framework limitation, not a Sevenbox bug --
verified in the module's own event stream, not assumed. The same failure
class as the OpenAI Agents SDK's `on_llm_end` over-counting bug (an
in-flight counter expecting N results that never all arrive): here the
risk would be `SmolagentsSession` waiting forever for a `ToolOutput` that
`record_new_id` never sees. It does not, verified directly on the
session's own internal fields (`_batch_open`, `_pending_remaining`), not
just inferred from downstream behavior -- when any call in a parallel
batch raises, `SmolagentsSession.close_step()` (invoked once per
`ActionStep`, success or error) flushes whatever calls *did* resolve
before the exception (possibly zero, as in the adversarial case above)
into the frontier and discards the open-but-abandoned pending count --
this correctly avoids leaving the session stuck waiting forever for a
`ToolOutput` that the framework itself will never deliver, but it cannot
recover a result the framework already discarded. See
`tests/test_smolagents_adapter.py
::test_KNOWN_LIMITATION_a_parallel_sibling_that_actually_succeeded_is_still_lost_when_another_sibling_errors`.
No `ToolOutput`, no Warrant -- Sevenbox cannot fabricate provenance for a
result it was never shown, and does not try to.

## Known limitation: concurrently-overlapping runs sharing one session

Same underlying cause as LlamaIndex's documented limit -- `SmolagentsSession`
tracks exactly one open batch via flat fields, with no per-turn identity
to key a dict on (unlike `OpenHandsSession`'s `llm_response_id`) -- but it
manifests differently, checked rather than assumed to match LlamaIndex's
shape: `on_tool_call` is *additive* when a batch is already open (it just
increments the pending count, it never re-freezes the parent set), so two
independent agents' overlapping immediate warrants still each correctly
cite the shared frontier they opened against -- no destructive overwrite
the way LlamaIndex's `begin_turn` has. What actually breaks is the
*close*: the two agents' overlapping batches get silently merged into one
shared close, so the next call from either agent inherits a false
dependency on the *other* agent's unrelated result -- confirmed
empirically, and only after an initial, more direct hypothesis (mirroring
LlamaIndex's immediate-mis-attribution shape) was written as a test and
found to pass instead of fail, i.e. was false --
the same "test before trusting a claim" discipline this project follows throughout, not a claim taken on faith. See
`tests/test_smolagents_adapter.py
::test_KNOWN_LIMITATION_concurrent_threads_sharing_one_session_corrupt_each_others_tracking`
for the corrected, verified failure shape. This is also a strictly worse
hazard than LlamaIndex's in one respect: LlamaIndex consumes its event
stream from a single asyncio task, so two independent runs sharing a
session there could only interleave *logically*, never race at the
Python-object level; here, `stream=True` is a plain synchronous
generator with no asyncio involved at all, so two real OS threads really
can call `SmolagentsSession`'s methods at overlapping wall-clock moments.
**Does a lock fix it? Verified with a real `threading.Lock`, not argued
by analogy to LlamaIndex.** `tests/test_smolagents_adapter.py
::test_KNOWN_LIMITATION_a_lock_does_not_fix_the_additive_merge_corruption_either`
wraps `on_tool_call`/`turn_parent_ids`/`record_new_id` in a
`threading.Lock` and forces the identical interleaving that corrupts the
unlocked version -- the false dependency still appears, every call fully
serialized. A lock removes the data-race dimension (no call can be torn
mid-execution by another thread) but does not touch the false-dependency
merge, because that corruption was never caused by a torn write in the
first place -- `on_tool_call`'s statements already ran to completion
before the next thread's call began, lock or not; what is wrong is the
*sequence* of complete calls itself, which has no per-agent/per-turn
identity to tell "agent_a's own second call" apart from "agent_b's
unrelated first call" against. A lock only controls *when* calls are
allowed to run relative to each other; it cannot supply information the
data model itself does not carry -- the identical conclusion, verified
the identical way, as LlamaIndex's own lock test. Not added, for the
same reason LlamaIndex's docstring gives: it would not actually fix the
limitation it looks like it fixes. The safe pattern is the same as every
other adapter in this package: one `sevenbox_smolagents_run()` session per
real agent run, or multiple runs consumed *sequentially* against one
shared session.

## CodeAgent is out of scope for per-call Warrant construction

Confirmed empirically (`probe_codeagent.py`), not assumed from
`CodeAgent`'s docstring: streaming a `CodeAgent` run yields one `ToolCall`
per step named `python_interpreter` whose `arguments` is the raw code
string, and **no `ToolOutput`** for it at all -- any real tool functions
the generated code invokes run inside `LocalPythonExecutor` (or a
sandboxed alternative), a boundary this adapter's stream-observer design
has no visibility into. `SevenboxEventObserver` still processes whatever
`ToolCall`/`ToolOutput` events a `CodeAgent` run happens to produce (there
are none of the latter), so a `CodeAgent` run produces no Warrants at all
under this adapter rather than silently mis-attributing one Warrant per
step to the whole code block -- a deliberate choice to fail loud (nothing
recorded) over fail quiet (a plausible-looking but false decomposition).
Reaching real per-call granularity for `CodeAgent` would require
instrumenting `LocalPythonExecutor` itself, a materially different and
heavier integration not attempted in this pass -- matching the same
"verify before designing" discipline OpenHands'/LlamaIndex's own
structural gaps were handled with, not silently forced.

## Why replay is out of scope for this pass

The same shape of gap as OpenHands/LlamaIndex, for an analogous reason:
skipping a real tool invocation (the `ReplayEffectExecutor` mechanism
every fully-live adapter in this package uses) requires substituting a
cached response *at the point where the real call would otherwise
happen* -- inside `process_tool_calls`'s own dispatch, not something this
adapter's event-stream-consumer design has a hook into from outside.
Reaching it would mean subclassing `ToolCallingAgent`/`CodeAgent` and
overriding their internal step methods, a much heavier integration point
than every other adapter in this package uses, and not attempted here
without first proving it is the right boundary. This adapter therefore
builds real, live Warrants from real events -- full causal correctness,
with an exact, pre-typed batch signal (question 3) -- but cannot skip a
real tool invocation the way the six fully-live adapters can.

## Usage

    from smolagents import ToolCallingAgent
    from sevenbox_adapters.smolagents_adapter import sevenbox_smolagents_run

    with sevenbox_smolagents_run() as krun:
        krun.session.start("check order 4471")
        agent = ToolCallingAgent(tools=[...], model=model)
        for event in agent.run("check order 4471", stream=True):
            krun.observe(event)
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
    from smolagents import ActionStep as _ActionStep
    from smolagents import ToolCall as _ToolCall
    from smolagents import ToolOutput as _ToolOutput
    _SMOLAGENTS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional extra is absent
    _ActionStep = None
    _ToolCall = None
    _ToolOutput = None
    _SMOLAGENTS_AVAILABLE = False


class SmolagentsSession:
    """Tracks the causal frontier of one Smolagents `stream=True` run (or
    several, consumed *sequentially* -- see module docstring's "Known
    limitation: concurrently-overlapping runs" for why simultaneous,
    multi-threaded sharing is unsafe).

    Unlike `LlamaIndexSession` (which learns a batch's full size upfront
    from a single `AgentOutput.tool_calls` list), this session learns a
    batch's size incrementally: `ToolCall` events arrive one at a time,
    all before any `ToolOutput` (module docstring question 3), so
    `on_tool_call` just tracks "is a batch currently open" as a boolean
    rather than a pre-known count.
    """

    def __init__(self, *, store: Optional[WarrantStore] = None) -> None:
        if not _SMOLAGENTS_AVAILABLE:
            raise RuntimeError(
                "SmolagentsSession requires the 'smolagents' extra "
                "(pip install -e '.[smolagents]')."
            )
        self.store = store if store is not None else WarrantStore()
        self._frontier: set = set()
        self._batch_open: bool = False
        self._turn_parent_ids: Optional[tuple] = None
        self._pending_remaining: int = 0
        self._new_ids_this_turn: set = set()

    def start(self, initial_input: Any = None) -> None:
        """Optional: root the run on an explicit axiom before any tool
        call fires. If the first `ToolCall` arrives without this having
        been called, `on_tool_call` roots the run on a fallback axiom
        instead -- never silently parentless."""
        if self._frontier:
            return
        w = create_warrant(
            content=initial_input, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def on_tool_call(self, fallback_axiom_content: Any = None) -> None:
        """Call once per `ToolCall` event. Opens a new batch (freezing
        the current frontier as this batch's shared parent set) only if
        no batch is currently open; a `ToolCall` arriving while a batch
        is already open just means this step has more than one parallel
        call, and joins the same batch."""
        if not self._batch_open:
            if not self._frontier:
                self.start(fallback_axiom_content)
            self._turn_parent_ids = tuple(sorted(self._frontier))
            self._batch_open = True
            self._pending_remaining = 0
            self._new_ids_this_turn = set()
        self._pending_remaining += 1

    def turn_parent_ids(self) -> tuple:
        """Read-only snapshot of the currently-open batch's frozen parent
        set."""
        if self._turn_parent_ids is not None:
            return self._turn_parent_ids
        if not self._frontier:
            self.start({"note": "tool output with no preceding ToolCall batch"})
        return tuple(sorted(self._frontier))

    def record_new_id(self, warrant_id: str) -> None:
        """Register one tool call's resulting Warrant as this batch's
        output. Closes the batch (folding all of the batch's outputs
        into the new frontier) once every call opened via `on_tool_call`
        has resolved."""
        self._new_ids_this_turn.add(warrant_id)
        if not self._batch_open:
            # A ToolOutput with no matching on_tool_call -- publish
            # immediately, matching the other adapters' ambient-frontier
            # fallback for calls made outside the normal batch cycle.
            self._frontier = {warrant_id}
            return
        self._pending_remaining -= 1
        if self._pending_remaining <= 0:
            self._close_batch()

    def close_step(self) -> None:
        """Call once per `ActionStep` event, success or error alike --
        the definitive step boundary. Guarantees an open batch never
        stays open past its own step even when some `ToolCall`s never
        got a matching `ToolOutput` (module docstring's "Known
        limitation: partial-batch failures"). A no-op if the batch
        already closed itself via `record_new_id` reaching zero."""
        if self._batch_open:
            self._close_batch()

    def _close_batch(self) -> None:
        if self._new_ids_this_turn:
            self._frontier = set(self._new_ids_this_turn)
        self._turn_parent_ids = None
        self._batch_open = False
        self._pending_remaining = 0
        self._new_ids_this_turn = set()

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


_current_smolagents_session: ContextVar[Optional["SmolagentsSession"]] = ContextVar(
    "sevenbox_smolagents_session", default=None
)


def sevenbox_smolagents_run(*, store: Optional[WarrantStore] = None) -> "_SevenboxSmolagentsRunContext":
    return _SevenboxSmolagentsRunContext(SmolagentsSession(store=store))


class _SevenboxSmolagentsRunContext:
    def __init__(self, session: SmolagentsSession) -> None:
        self.session = session
        self.observer = SevenboxEventObserver()
        self._token = None

    def __enter__(self) -> "_SevenboxSmolagentsRunContext":
        self._token = _current_smolagents_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_smolagents_session.reset(self._token)
        return False

    def observe(self, event: Any) -> None:
        """Call once per event while iterating
        `agent.run(task, stream=True)` inside this context
        (`for event in agent.run(task, stream=True): krun.observe(event)`)."""
        self.observer(event)


class SevenboxEventObserver:
    """Stateless; reads the ambient `SmolagentsSession` from a
    `ContextVar` at call time, the same pattern every other adapter in
    this package uses. Safe to share across multiple *sequentially*
    consumed `stream=True` loops (module docstring's "genuinely lock-free
    design") -- not safe across *concurrent* ones (module docstring's
    "Known limitation: concurrently-overlapping runs").
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
        if not _SMOLAGENTS_AVAILABLE:
            raise RuntimeError(
                "SevenboxEventObserver requires the 'smolagents' extra "
                "(pip install -e '.[smolagents]')."
            )
        self._source_kind = source_kind
        self._epistemic = epistemic
        self._reversibility = reversibility
        self._redaction = redaction
        self._origin = origin

    def __call__(self, event: Any) -> None:
        if isinstance(event, _ToolCall):
            self._on_tool_call(event)
        elif isinstance(event, _ToolOutput):
            self._on_tool_output(event)
        elif isinstance(event, _ActionStep):
            self._on_action_step(event)
        # ChatMessageStreamDelta/ActionOutput/PlanningStep/FinalAnswerStep
        # carry no new causal fact beyond what ToolCall/ToolOutput already
        # capture -- matching this project's own definition of what does and doesn't produce a Warrant
        # (bookkeeping/chatter), they produce no Warrant here.

    def _on_tool_call(self, event: Any) -> None:
        session = _current_smolagents_session.get()
        if session is None:
            raise RuntimeError(
                "A Smolagents ToolCall event was observed by "
                "SevenboxEventObserver outside a `with sevenbox_smolagents_run():` "
                "block."
            )
        session.on_tool_call(fallback_axiom_content={"tool_call_id": event.id})

    def _on_action_step(self, event: Any) -> None:
        session = _current_smolagents_session.get()
        if session is None:
            raise RuntimeError(
                "A Smolagents ActionStep event was observed by "
                "SevenboxEventObserver outside a `with sevenbox_smolagents_run():` "
                "block."
            )
        session.close_step()

    def _on_tool_output(self, event: Any) -> None:
        session = _current_smolagents_session.get()
        if session is None:
            raise RuntimeError(
                "A Smolagents ToolOutput event was observed by "
                "SevenboxEventObserver outside a `with sevenbox_smolagents_run():` "
                "block."
            )
        tool_call = event.tool_call
        tool_name = tool_call.name if tool_call is not None else None
        raw_args = tool_call.arguments if tool_call is not None else None
        args = dict(raw_args) if isinstance(raw_args, dict) else raw_args
        req_payload = {"args": args}
        parent_ids = list(session.turn_parent_ids())
        response = event.output if event.output is not None else event.observation

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
