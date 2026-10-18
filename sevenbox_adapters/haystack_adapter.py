"""
sevenbox_adapters.haystack_adapter -- the "sevenbox wrap" story for Haystack
Agents (`haystack-ai` on PyPI, deepset's LLM/pipeline framework).

## Verification strength, stated up front

Real Haystack orchestration code runs in-process in every test in this
file, against a hand-rolled `@component`-decorated `ChatGenerator`
(network-free, no API key -- Haystack ships no offline scripted-response
test double of its own). The same class of verification as Strands,
Google ADK, and the first ten adapters, not the weaker contract-shape-
only verification `letta_adapter.py` had to settle for.

## Which package: `haystack-ai`, and note the 3.0.0 major version

`haystack-ai` is the real package name (the bare `haystack` name on PyPI
is an unrelated, abandoned project) -- verified before installing
anything, the same discipline used to pick `llama-index-core`/
`openhands-sdk`/`letta-client` over their own PyPI-adjacent look-alikes.
Verified against `haystack-ai==3.0.0`, a very recent major-version
release at the time of this integration -- the `Agent`/hooks surface
used here matched what the installed package actually exposes, checked
directly rather than assumed to still match older Haystack 2.x
documentation that may be findable by search.

## The 8 questions, answered from real investigation

1. **Where is a decision observed?** `Agent(hooks={"before_tool": [...],
   "after_tool": [...], ...})` -- a real, dict-keyed hooks system
   (`haystack.hooks`), structurally different from every other hooks-
   based adapter in this package: a hook is not handed a typed, per-call
   event object (unlike Strands' `AfterToolCallEvent` or OpenAI Agents
   SDK's `RunHooks`) -- it receives the Agent's entire live, mutable
   `State` (confirmed via `Hook`'s real `Protocol` definition: `def
   run(self, state: State) -> None`), and must read `state.data[
   "messages"]` itself to find out what happened. Validated at
   `Agent.__init__` via `callable(getattr(h, "run", None))` -- a plain
   function is NOT accepted directly (confirmed empirically: passing one
   raised `TypeError` telling you to wrap it with the `@hook` decorator,
   or hand it a `Hook`-shaped object), so this adapter exposes its hooks
   as small objects implementing `run(self, state)` rather than bare
   functions.
2. **Where is a tool call identified?** `ToolCallResult.origin` is
   literally the originating `ToolCall` object (confirmed via
   `dataclasses.fields()` on the real installed type) -- the same
   caliber of correlation as Strands' bundled `AfterToolCallEvent`
   (`tool_use`+`result` on one object), just reached differently: here
   the call and its result live on two *different* `ChatMessage`s (an
   assistant message with `.tool_calls`, followed later by tool-role
   messages each with one `.tool_call_results` entry), but `.origin`
   still means no id-matching dict is needed -- each result already
   carries its own call.
3. **Parallel vs. sequential?** `BEFORE_TOOL` fires once per step, with
   `state.data["messages"][-1].tool_calls` already holding the model's
   COMPLETE requested batch -- confirmed empirically with a real
   two-tool-call turn: a true whole-batch-before-dispatch signal, the
   same shape as LlamaIndex's `AgentOutput.tool_calls`/Strands'
   `BeforeToolsEvent.message`. `AFTER_TOOL` also fires once per step,
   after every result for that step is already appended to
   `state.data["messages"]` -- this adapter identifies exactly which
   messages belong to the just-finished batch by snapshotting
   `len(state.data["messages"])` at `BEFORE_TOOL` and slicing the tail
   at `AFTER_TOOL`, rather than counting individual call/result events
   the way smolagents does. Real OS-thread parallelism is the default,
   same as Strands and unlike Google ADK: read directly in
   `haystack/components/agents/tool_calling.py::_run_tool`, a real
   `ThreadPoolExecutor(max_workers=4)` (the same `tool_concurrency_limit`
   `Agent.__init__` parameter controls) -- confirmed empirically too, two
   tool bodies sleeping 0.3s each together took ~0.31s (matching `max()`,
   not the ~0.6s sum), on two distinct real worker thread ids. `contextvars.
   copy_context()` is captured per call before `executor.submit(...)`
   (`_make_context_bound_invoke`, confirmed by reading its own docstring:
   "preserves tracing spans and other context-local state across
   thread-pool boundaries") -- the same isolation guarantee already
   verified for six other adapters.
4. **Branching representation?** Not applicable within one agent's own
   step loop; Haystack's own `Pipeline`/multi-agent composition (a
   separate, real orchestration story built on the same `@component`
   system) was not exercised in this pass.
5. **Context/parent propagation?** Hook callbacks never run on the
   `ThreadPoolExecutor` worker threads tool bodies run on -- confirmed
   empirically: recording the thread id inside an `after_tool` hook
   during the same two-parallel-tool-call run that proved genuine
   worker-thread dispatch for the tool bodies showed a single,
   consistent thread id (the main thread) for the callback. The same
   `run_in_executor`-style thread-affinity argument already verified for
   LlamaIndex and Strands, applied here to Haystack's dict-based hook
   dispatch. This is what makes `HaystackSession` safe with no lock for
   a single agent run's own events (see "No lock needed" below).
6. **Retries / errors?** A second, independently-verified instance of
   the genuinely positive finding first confirmed for Strands (and, per
   that adapter's own docstring, the OPPOSITE of smolagents'/Google
   ADK's shared limitation) -- verified empirically, not assumed to
   match either precedent: when one call in a two-call parallel batch
   raises, the OTHER call's result is **not** discarded. Confirmed with a
   real batch of `[ok_tool, erroring_tool]`: both calls' `ToolCallResult`s
   appeared in `state.data["messages"]` after the step -- `ok_tool`'s
   with `error=False` and its real result string, `erroring_tool`'s with
   `error=True` and a descriptive error string as `.result`
   ("Failed to invoke Tool erroring_tool with parameters {'x': 'b'}.
   Error: boom: b"). No exception ever escapes `agent.run()` itself --
   confirmed directly in `_make_context_bound_invoke`'s own docstring:
   "The callable returns a ToolInvocationError instead of raising so
   that parallel executions can collect failures without aborting the
   whole batch." `Agent.__init__`'s `raise_on_tool_invocation_failure`
   parameter (default `False`) is the real, named, documented switch
   controlling this -- not an accident of this specific test.
7. **Replay without re-execution?** Not implemented in this pass, for
   the same structural reason as every hook/stream-consumer adapter in
   this package (LlamaIndex/OpenHands/smolagents' `CodeAgent`/Google
   ADK/Strands): this adapter observes `State` after Haystack has
   already run the real tool call internally -- there is no live Python
   call site this adapter's own code controls to substitute a cached
   response at.
8. **What does the framework not expose?** Nothing missing for the
   hook-consumer design chosen here. `_schedule_tool_calls` groups a
   step's tool calls into sub-batches that respect read-after-write
   dependencies on `State` -- meaning not every call in one `BEFORE_TOOL`/
   `AFTER_TOOL` pair is necessarily dispatched concurrently with every
   other -- but this adapter does not need to model that internal
   scheduling: it treats the whole step's batch as one causal unit
   either way, which is correct regardless of how Haystack internally
   sequenced the dispatch.

## No lock needed, for the same reason as LlamaIndex/Strands -- but the identical caller-owned-sharing hazard as every prior adapter

`HaystackSession` needs no `threading.Lock` for a single agent run's own
events, for the identical reason already verified for LlamaIndex and
Strands: hook callbacks always run on the thread driving the agent's own
loop, never on the `ThreadPoolExecutor` worker threads tool bodies
actually run on (question 5).

**Sharing one `HaystackSession` across genuinely independent, concurrently-run `Agent`s is a different, real hazard, verified rather than assumed** -- the same class of check every session-based adapter in this package now gets. A first attempt at the test opened both agents' batches while neither had resolved yet and found no corruption -- `begin_batch` overwriting `_turn_parent_ids` with the *same* still-fresh axiom value twice is a no-op in effect, not a bug; the corruption needs a third step. Exactly the "assert -> test -> discovered false -> correct" pattern this project's "test before trusting a claim" discipline, caught while writing this adapter's own test, not left for a later session. The real shape, confirmed with a test forcing the precise interleaving deterministically
(`test_KNOWN_LIMITATION_concurrent_agents_sharing_one_session_corrupt_each_others_tracking`):
agent_a's batch opens (freezing the axiom as its parent set), agent_b's
batch opens too while agent_a's is still pending (overwriting
`_turn_parent_ids` with the same axiom value -- still harmless), then
agent_a fully *resolves* -- which resets `_turn_parent_ids` to `None`
and moves `_frontier` forward to agent_a's own result. When agent_b
finally reads its own (supposedly frozen) parent set, `_turn_parent_ids`
is `None` again, so `turn_parent_ids()` falls through to "no batch open,
use the current frontier" -- which is now agent_a's result, not the
axiom agent_b's batch actually opened against. agent_b's warrant ends up
parented on agent_a's, a false dependency between two logically
unrelated agents.
**Does a lock fix it? Verified with a real `threading.Lock`, not assumed
by analogy** --
`test_KNOWN_LIMITATION_a_lock_does_not_fix_the_concurrent_session_corruption_either`
confirms no, by the identical mechanism already proven for LlamaIndex,
Google ADK, and Strands: the corruption is not a torn write, it is that
`HaystackSession` has no per-agent/per-turn identity for `turn_parent_ids`
to check against before falling back to "whatever the frontier currently
is." Not added, for the same reason those three adapters give.
The safe pattern is the same as always: one `sevenbox_haystack_run()`
session per real agent run, or multiple runs consumed sequentially
against one shared session.

## Usage

    from haystack.components.agents import Agent
    from haystack.dataclasses import ChatMessage
    from sevenbox_adapters.haystack_adapter import sevenbox_haystack_run

    with sevenbox_haystack_run() as krun:
        krun.session.start("check order 4471")
        agent = Agent(chat_generator=generator, tools=[...], hooks=krun.observer.hooks)
        result = agent.run(messages=[ChatMessage.from_user("check order 4471")])
        print(krun.session.frontier())

## What this pass did not verify

Haystack's own `Pipeline`/multi-agent composition and its async run path
(`Agent.run_async`, which uses `asyncio.Semaphore`-bounded concurrency
instead of `ThreadPoolExecutor` -- read in `_run_tool_async`, not
independently exercised) were both found during investigation but not
tested against this adapter.
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
    import haystack  # noqa: F401
    _HAYSTACK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional extra is absent
    _HAYSTACK_AVAILABLE = False


class HaystackSession:
    """Tracks the causal frontier of one Haystack `Agent.run()` (or
    several, sharing this session -- see module docstring's "No lock
    needed" section for what sharing does and does not make safe).

    Unlike every other session in this package, batch membership is not
    known via an explicit count or a bundled per-call event -- it is
    inferred from the length of `state.data["messages"]` before and
    after one `BEFORE_TOOL`/`AFTER_TOOL` pair (module docstring
    question 3).
    """

    def __init__(self, *, store: Optional[WarrantStore] = None) -> None:
        if not _HAYSTACK_AVAILABLE:
            raise RuntimeError(
                "HaystackSession requires the 'haystack' extra "
                "(pip install -e '.[haystack]')."
            )
        self.store = store if store is not None else WarrantStore()
        self._frontier: set = set()
        self._turn_parent_ids: Optional[tuple] = None
        self.message_count_before_batch: Optional[int] = None

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

    def begin_batch(self, message_count_before: int, fallback_axiom_content: Any = None) -> None:
        """Call once per `BEFORE_TOOL` hook invocation, with
        `len(state.data["messages"])` at that moment -- always freezes
        the current frontier as this batch's shared parent set."""
        if not self._frontier:
            self.start(fallback_axiom_content)
        self._turn_parent_ids = tuple(sorted(self._frontier))
        self.message_count_before_batch = message_count_before

    def turn_parent_ids(self) -> tuple:
        """Read-only snapshot of this batch's frozen parent set."""
        if self._turn_parent_ids is not None:
            return self._turn_parent_ids
        if not self._frontier:
            self.start({"note": "tool result with no preceding BEFORE_TOOL batch"})
        return tuple(sorted(self._frontier))

    def resolve_batch(self, results: list, builder) -> None:
        """Call once per `AFTER_TOOL` hook invocation, with the list of
        `ToolCallResult`s newly appended since the matching
        `begin_batch`. Closes the batch (folding all of the batch's
        outputs into the new frontier)."""
        new_ids = set()
        parent_ids = self.turn_parent_ids()
        for result in results:
            w = builder(result, parent_ids)
            self.store.add(w)
            new_ids.add(w.id)
        if new_ids:
            self._frontier = new_ids
        self._turn_parent_ids = None
        self.message_count_before_batch = None

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


_current_haystack_session: ContextVar[Optional["HaystackSession"]] = ContextVar(
    "sevenbox_haystack_session", default=None
)


def sevenbox_haystack_run(*, store: Optional[WarrantStore] = None) -> "_SevenboxHaystackRunContext":
    return _SevenboxHaystackRunContext(HaystackSession(store=store))


class _SevenboxHaystackRunContext:
    def __init__(self, session: HaystackSession) -> None:
        self.session = session
        self.observer = SevenboxEventObserver()
        self._token = None

    def __enter__(self) -> "_SevenboxHaystackRunContext":
        self._token = _current_haystack_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_haystack_session.reset(self._token)
        return False


class _BeforeToolHook:
    """A real `Hook` (per `haystack.hooks.protocol.Hook`'s `Protocol` --
    validated at `Agent.__init__` via `callable(getattr(h, "run", None))`,
    confirmed empirically that a bare function is rejected)."""

    def __init__(self, observer: "SevenboxEventObserver") -> None:
        self._observer = observer

    def run(self, state: Any) -> None:
        self._observer._on_before_tool(state)


class _AfterToolHook:
    def __init__(self, observer: "SevenboxEventObserver") -> None:
        self._observer = observer

    def run(self, state: Any) -> None:
        self._observer._on_after_tool(state)


class SevenboxEventObserver:
    """Stateless; reads the ambient `HaystackSession` from a `ContextVar`
    at call time, the same pattern every other adapter in this package
    uses. Expose `.hooks` directly to `Agent(hooks=...)`.
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
        if not _HAYSTACK_AVAILABLE:
            raise RuntimeError(
                "SevenboxEventObserver requires the 'haystack' extra "
                "(pip install -e '.[haystack]')."
            )
        self._source_kind = source_kind
        self._epistemic = epistemic
        self._reversibility = reversibility
        self._redaction = redaction
        self._origin = origin
        self.hooks = {"before_tool": [_BeforeToolHook(self)], "after_tool": [_AfterToolHook(self)]}

    def _session(self, hook_name: str) -> "HaystackSession":
        session = _current_haystack_session.get()
        if session is None:
            raise RuntimeError(
                f"A Haystack {hook_name} hook was invoked by SevenboxEventObserver "
                "outside a `with sevenbox_haystack_run():` block."
            )
        return session

    def _on_before_tool(self, state: Any) -> None:
        session = self._session("before_tool")
        messages = state.data["messages"]
        n_before = len(messages)
        last = messages[-1] if messages else None
        fallback = {"tool_calls": [tc.tool_name for tc in last.tool_calls]} if last and last.tool_calls else None
        session.begin_batch(n_before, fallback_axiom_content=fallback)

    def _on_after_tool(self, state: Any) -> None:
        session = self._session("after_tool")
        n_before = session.message_count_before_batch or 0
        new_messages = state.data["messages"][n_before:]
        results = []
        for message in new_messages:
            if message.tool_call_results:
                results.extend(message.tool_call_results)
        session.resolve_batch(results, self._build_warrant)

    def _build_warrant(self, result: Any, parent_ids: tuple):
        call = result.origin
        args = dict(call.arguments) if call.arguments else {}
        req_payload = {"args": args}
        response_content = _tool_result_content(result)

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
                response=apply_redaction(
                    {"content": response_content, "error": result.error}, self._redaction,
                ),
                source_name=call.tool_name, redacted=self._redaction is not None,
            ),
        )


def _tool_result_content(result: Any) -> Any:
    """`ToolCallResult.result` is `str | Sequence[TextContent | ImageContent
    | FileContent]`. Prefer the plain string (the common case); for a
    structured sequence, prefer a single text part's `.text`, otherwise
    pass the sequence through as-is."""
    value = result.result
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and hasattr(value[0], "text"):
        return value[0].text
    return value
