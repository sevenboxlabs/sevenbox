"""
sevenbox_adapters.crewai_adapter -- the "sevenbox wrap" story for CrewAI
(`crewai` package on PyPI).

Verified against the actual installed crewai==1.15.13 package, not
training-data assumptions about its API. Verification includes real
offline runs -- a custom BaseLLM subclass driving real Agent/Task/Crew
objects through crewai.kickoff(), both without tools (Task-level layer)
and with genuinely parallel native tool calls dispatched by CrewAI's own
ThreadPoolExecutor (tool-call layer) -- see tests/test_crewai_adapter.py.

## Design: two layers, because CrewAI exposes two different, honest
## levels of causal precision -- this adapter does not force one grain
## onto information the framework doesn't give it at that grain.

### Layer 1 (primary, exact): one Warrant per completed Task

CrewAI already thinks in DAG terms at the Task level: `Task.context`
is a first-class field --- literally "the other Tasks this task's
output depends on" --- verified against the framework's own
`Crew._get_context` to have three distinct states, not two:

  1. `context` left unset (the `NOT_SPECIFIED` sentinel, Task's actual
     default): the task implicitly receives EVERY previously-completed
     task's output as context. `CrewSession.record_task_output` mirrors
     this: parent_ids = every Task Warrant produced so far (the
     accumulating frontier).
  2. `context=[task_a, task_b, ...]` (explicit list): parent_ids =
     EXACTLY those tasks' Warrant ids, verified present in the session
     before the task in question ran (a task declaring context on one
     that has not completed yet raises, rather than silently omitting
     it).
  3. `context=None` / `context=[]` (explicit, falsy): CrewAI's own
     `_get_context` returns "" for this case -- no prior task
     contributes to the prompt at all -- so parent_ids = the axiom only,
     not the accumulated frontier.

No inference, no guessing, no "ambient" chaining at this layer: every
number above is read directly from a field CrewAI's own DAG scheduling
already depends on, so being wrong here would break CrewAI's own
execution, not just Sevenbox's bookkeeping.

### Layer 2 (secondary, honestly approximate): tool calls within a Task

Verified against crewai.agents.crew_agent_executor.CrewAgentExecutor
directly (not documentation, which is stale here -- see below):
CrewAI's own hooks (`before_tool_call`/`after_tool_call`,
`ToolCallHookContext`) fire once PER INDIVIDUAL tool call, with no
batch/turn identifier exposed alongside them -- unlike
sevenbox_adapters.pydantic_ai_adapter's `RunContext.run_step` or
sevenbox_adapters.openai_agents_adapter's `on_llm_end` (which hands back
every tool call in a turn before any execute), there is no public
signal here that says "these N calls are siblings." This is registered
here as a genuine, current limitation of CrewAI's public API surface,
not approximated away: this layer uses the same "ambient" per-task
chaining sevenbox_adapters.langgraph_adapter.GraphSession documents --
correct for the common case (one tool call at a time within a task) and
imprecise for a true parallel batch, where each call in the batch is
chained onto whichever sibling happened to be recorded first rather
than sharing the task's own parent set.

A real, verified discrepancy worth flagging directly: the docstring of
`CrewAgentExecutor._handle_native_tool_calls` claims it "Executes only
the FIRST tool call" -- but the actual code, for a batch of >1 native
tool calls with no `result_as_answer`/`max_usage_count` tool in it,
dispatches the WHOLE batch through a real `ThreadPoolExecutor`
(`concurrent.futures`, genuine OS threads, not async concurrency) and
collects results via `as_completed`. This adapter's own tests exercise
that real thread pool directly, not a simulation of it -- see
tests/test_crewai_adapter.py's parallel-batch test.

### Real thread-safety, not just reactive bookkeeping

Because CrewAI's native parallel tool-call batch is dispatched via
`ThreadPoolExecutor` (true OS threads, unlike every other framework this
project has integrated, which only ever reached the adapter through a
single thread or cooperative async concurrency), `CrewSession` guards
its mutable state (`_frontier`, `_task_id_to_warrant_id`, the per-task
tool-call frontier) with a real `threading.RLock`. This is new to this
adapter specifically -- verified necessary, not precautionary, given
`_handle_native_tool_calls`'s own use of `contextvars.copy_context().run(...)`
per submitted worker (each thread gets an independent COPY of context
vars, so `_current_crew_session.get()` still correctly reads the active
session from any worker thread, but mutating that session's shared
Python objects from multiple real threads without a lock would be a
genuine race, not a hypothetical one).

### Replay: real for Layer 2 (tool calls), NOT possible for Layer 1 (Tasks)

Earlier verification against `CrewAgentExecutor._execute_single_native_tool_call`
confirmed `before_tool_call`/`after_tool_call` cannot substitute a custom
cached result for a real tool call -- only fully block it with a fixed,
generic message. That remains true. But the SAME method also checks
`self.tools_handler.cache` -- CrewAI's own native tool-result cache,
keyed on `(tool_name, json.dumps(args))` -- BEFORE calling the real tool
body, and `Agent.tools_handler` is a plain, directly-settable field
(`agent.tools_handler.cache = ...`), not something locked away inside
executor-internal state. Verified empirically (see
tests/test_crewai_adapter.py's replay tests): pre-populating that cache
from a reference `WarrantStore`'s captured `effect_record`s, before
`kickoff()`, makes CrewAI's own executor skip the real tool call
entirely and use the cached value -- genuine non-execution replay for
Layer 2, using a mechanism the framework already has for an unrelated
purpose (performance caching), not a new interception point invented
here.

This is real but partial, and the boundary is exact, not approximate:
**Layer 2 tool calls replay for real; Layer 1 Task/LLM calls do not.**
There is no analogous cache CrewAI consults before invoking an agent's
LLM for a Task -- `krun.wrap(..., replay_from=store)` therefore still
re-invokes the real (or, in every test/example here, a deterministic
scripted) LLM for every Task. If that LLM is deterministic, the
resulting Task Warrants converge to byte-identical ids by construction
(content-addressing + idempotent construction) -- but that is a
weaker guarantee than "did not execute," and this module does not
conflate the two (see test_two_deterministic_live_runs_produce_
identical_warrant_ids_but_both_really_execute for the same distinction
drawn explicitly for the no-replay case this replaces).

One more honest, verified difference from the other three adapters'
replay semantics: `sevenbox_adapters.warrant_runtime.ReplayEffectExecutor`
(used by every other adapter) raises `KeyError` loudly if a live call
has no matching cached entry -- replay there is all-or-nothing by
design. CrewAI's own native cache instead falls through to REAL
execution silently on a miss (`cached_result is None` -> call the real
tool). This adapter does not paper over that difference by adding a
stricter pre-flight check on top of CrewAI's native behavior; a
replay run with incomplete reference data will partially replay and
partially re-execute for real, and this is stated here rather than
left for a user to discover by surprise.

### Tool-call hooks are process-global, not scoped per run

`crewai.hooks.register_before_tool_call_hook` / `register_after_tool_call_hook`
register a hook for every tool call in the process, with no per-Crew or
per-run scoping in CrewAI's own API. This adapter registers its hook
pair exactly once, at first use, and the hook body itself reads the
same `_current_crew_session` ContextVar every other adapter in this
project uses to find "is there an active `sevenbox_agent_run()` session
right now" -- correctly a no-op for tool calls from any Crew running
outside one.

## Integration surface

    from sevenbox_adapters.crewai_adapter import sevenbox_task, sevenbox_agent_run
    from crewai import Agent, Task, Crew, Process
    from warrant_core import SourceKind

    agent = Agent(role="Analyst", goal="...", backstory="...")
    task1 = Task(description="Gather data", expected_output="...", agent=agent)
    task2 = Task(description="Summarize", expected_output="...", agent=agent, context=[task1])
    crew = Crew(agents=[agent], tasks=[task1, task2], process=Process.sequential)

    with sevenbox_agent_run() as krun:
        krun.session.start("Analyze Q1 revenue")
        krun.wrap(crew)  # registers task_callback; wraps any existing one
        result = crew.kickoff()
        print(krun.session.frontier().live)

Two changes to an existing CrewAI app: `krun.wrap(crew)` before
`crew.kickoff()`, the `sevenbox_agent_run()` block around the whole run.
Task and tool bodies untouched. Tool-call capture (Layer 2, approximate)
is opt-in per tool via `@sevenbox_tool`, exactly like the other adapters --
Layer 1 (Task-level, exact) requires no per-tool annotation at all.

`source_kind` for a Task Warrant defaults to `SourceKind.ORACLE`
(effectful): a task's output is authored by the agent's LLM, not a
deterministic external system call, and the adapter cannot know by
inspection whether an agent's tools made it otherwise.
"""
from __future__ import annotations

import json
import threading
from contextvars import ContextVar
from typing import Any, Callable, Optional
from uuid import UUID

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
    from crewai.agents.cache.cache_handler import CacheHandler as _CrewCacheHandler
    from crewai.hooks import register_after_tool_call_hook, register_before_tool_call_hook
    from crewai.utilities.constants import NOT_SPECIFIED as _CREW_NOT_SPECIFIED
    _CREWAI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when crewai is absent
    _CrewCacheHandler = None
    register_before_tool_call_hook = None
    register_after_tool_call_hook = None
    _CREW_NOT_SPECIFIED = object()
    _CREWAI_AVAILABLE = False


class CrewSession:
    """Tracks the causal graph of one CrewAI run across both layers. See
    module docstring's "Design" section."""

    def __init__(self, *, store: Optional[WarrantStore] = None) -> None:
        """No `replay_from` parameter -- see module docstring's "Replay:
        NOT supported by this adapter" section for why this is a
        deliberate omission, not an oversight."""
        self.store = store if store is not None else WarrantStore()
        self._lock = threading.RLock()
        # Layer 1 state
        self._axiom_id: Optional[str] = None
        self._frontier: set = set()  # accumulates every Task Warrant so far
        self._task_id_to_warrant_id: "dict[UUID, str]" = {}
        # Layer 2 state: per-task ambient frontier for tool calls
        self._tool_frontier_by_task: "dict[UUID, set]" = {}

    def start(self, initial_input: Any) -> None:
        """Root the run on an explicit axiom (e.g. the crew's kickoff
        input) before any task completes. Idempotent: a second call is a
        no-op, matching every other adapter's start()."""
        with self._lock:
            if self._axiom_id is not None:
                return
            w = create_warrant(
                content=initial_input, source_kind=SourceKind.AXIOM,
                epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            )
            self.store.add(w)
            self._axiom_id = w.id
            self._frontier = {w.id}

    def _ensure_started(self, fallback_content: Any) -> None:
        if self._axiom_id is None:
            self.start(fallback_content)

    # --- Layer 1: Task-level, exact -------------------------------------

    def record_task_output(
        self,
        *,
        task: Any,  # crewai.Task
        output: Any,  # crewai.tasks.task_output.TaskOutput
        source_kind: SourceKind = SourceKind.ORACLE,
        epistemic: EpistemicClass = EpistemicClass.INFERRED,
        reversibility: ReversibilityClass = ReversibilityClass.PURE,
        redaction: Optional[RedactionPolicy] = None,
        origin: Optional[Origin] = None,
    ):
        """Call once per completed Task -- normally via the task_callback
        `sevenbox_agent_run().wrap(crew)` installs, not called directly by
        application code. Returns the created Warrant."""
        with self._lock:
            self._ensure_started({"description": task.description})

            if task.context is _CREW_NOT_SPECIFIED:
                parent_ids = sorted(self._frontier)
            elif not task.context:
                parent_ids = [self._axiom_id]
            else:
                missing = [t for t in task.context if t.id not in self._task_id_to_warrant_id]
                if missing:
                    raise ValueError(
                        f"Task {task.description!r} declares context on "
                        f"{len(missing)} task(s) that have not completed yet in this "
                        f"session: {[t.description for t in missing]}. CrewAI itself "
                        f"would also be missing that output in the prompt -- this is "
                        f"almost always a task-ordering bug in the crew definition, "
                        f"not something the adapter should paper over."
                    )
                parent_ids = sorted({self._task_id_to_warrant_id[t.id] for t in task.context})

            content = {
                "description": task.description,
                "raw": getattr(output, "raw", output),
            }
            request = {"description": task.description, "expected_output": getattr(task, "expected_output", None)}

            if source_kind in PURE_SOURCE_KINDS:
                w = create_warrant(
                    content=content, source_kind=source_kind, epistemic=epistemic,
                    reversibility=reversibility, parent_ids=parent_ids, origin=origin,
                )
            else:
                w = create_warrant(
                    content=content, source_kind=source_kind, epistemic=epistemic,
                    reversibility=reversibility, parent_ids=parent_ids, origin=origin,
                    effect_record=EffectRecord(
                        request=apply_redaction(request, redaction),
                        response=apply_redaction(getattr(output, "raw", output), redaction),
                        source_name=getattr(task, "name", None) or task.description[:60],
                        redacted=redaction is not None,
                    ),
                )

            self.store.add(w)
            self._task_id_to_warrant_id[task.id] = w.id
            if self._frontier == {self._axiom_id}:
                # The bare axiom is scaffolding for "nothing has happened
                # yet" -- the first real Task Warrant already causally
                # contains "justified starting from the axiom" via its
                # OWN parent_ids, so it replaces the axiom in the
                # frontier rather than sitting alongside it.
                self._frontier = {w.id}
            else:
                self._frontier.add(w.id)
            return w

    # --- Layer 2: tool-call, honestly approximate ------------------------

    def record_tool_call(
        self,
        *,
        task_id: Optional[UUID],
        tool_name: str,
        args: dict,
        result: Any,
        source_kind: SourceKind = SourceKind.TOOL,
        epistemic: EpistemicClass = EpistemicClass.VERIFIED,
        reversibility: ReversibilityClass = ReversibilityClass.PURE,
        redaction: Optional[RedactionPolicy] = None,
        origin: Optional[Origin] = None,
    ):
        """Called from the global after_tool_call hook -- see module
        docstring's Layer 2 section for exactly what precision this
        does and does not have."""
        with self._lock:
            self._ensure_started({"tool": tool_name})
            if task_id is not None and task_id in self._task_id_to_warrant_id:
                base_frontier = {self._task_id_to_warrant_id[task_id]}
            else:
                base_frontier = {self._axiom_id}
            current = self._tool_frontier_by_task.get(task_id, base_frontier)
            parent_ids = sorted(current)

            request = {"args": args}
            w = create_warrant(
                content=result, source_kind=source_kind, epistemic=epistemic,
                reversibility=reversibility, parent_ids=parent_ids, origin=origin,
                effect_record=EffectRecord(
                    request=apply_redaction(request, redaction),
                    response=apply_redaction(result, redaction),
                    source_name=tool_name, redacted=redaction is not None,
                ),
            )
            self.store.add(w)
            self._tool_frontier_by_task[task_id] = {w.id}
            return w

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


_current_crew_session: ContextVar[Optional[CrewSession]] = ContextVar(
    "sevenbox_crewai_session", default=None
)

_global_hooks_registered = False
_global_hooks_lock = threading.Lock()


def _global_before_tool_call_hook(context) -> None:
    # Layer 2 is opt-in per tool (only tools wrapped with @sevenbox_tool
    # record anything); this hook currently only needs to exist so
    # after_tool_call has a matching pair registered -- kept symmetric
    # and empty rather than omitted, in case a future version needs a
    # before-call checkpoint (e.g. redaction of tool_input in place).
    return None


def _global_after_tool_call_hook(context) -> None:
    session = _current_crew_session.get()
    if session is None:
        return
    tool_name = getattr(context, "tool_name", None)
    registration = _SEVENBOX_TOOL_REGISTRY.get(tool_name)
    if registration is None:
        return  # not a @sevenbox_tool-wrapped tool -- Layer 2 is opt-in
    task = getattr(context, "task", None)
    task_id = getattr(task, "id", None)
    session.record_tool_call(
        task_id=task_id, tool_name=tool_name,
        args=dict(getattr(context, "tool_input", {}) or {}),
        result=getattr(context, "raw_tool_result", None),
        source_kind=registration["source_kind"], epistemic=registration["epistemic"],
        reversibility=registration["reversibility"], redaction=registration["redaction"],
        origin=registration["origin"],
    )


def _ensure_global_hooks_registered() -> None:
    global _global_hooks_registered
    if _global_hooks_registered:
        return
    with _global_hooks_lock:
        if _global_hooks_registered:
            return
        if not _CREWAI_AVAILABLE:
            raise RuntimeError(
                "sevenbox_adapters.crewai_adapter requires the 'crewai' package. "
                "Install the 'crewai' extra."
            )
        register_before_tool_call_hook(_global_before_tool_call_hook)
        register_after_tool_call_hook(_global_after_tool_call_hook)
        _global_hooks_registered = True


# name -> {source_kind, epistemic, reversibility, redaction, origin}
_SEVENBOX_TOOL_REGISTRY: "dict[str, dict]" = {}


def _reject_if_chromadb_backed_features_enabled(crew: Any) -> None:
    """See `_SevenboxAgentRunContext.wrap`'s docstring for the full CVE
    context. Checks crew-level and per-agent `knowledge`/
    `knowledge_sources` ONLY -- verified against the installed package
    (crewai 1.15.x), not assumed: `crewai/knowledge/storage/
    knowledge_storage.py` imports `crewai.rag.chromadb` unconditionally,
    with no alternative backend, so enabling Knowledge always reaches
    chromadb. `memory=True` was checked too and deliberately left OUT of
    this guard -- `crewai/memory/unified_memory.py`'s `Memory.storage`
    defaults to `"lancedb"`, and its only reference to `chromadb` is a
    `TYPE_CHECKING`-only import (never executed) -- so flagging bare
    `memory=True` here would be a real false positive, not caution. If a
    future crewai version adds a chromadb-backed memory storage option,
    or Memory is constructed with one explicitly, this check does not
    catch that -- it checks the feature that is unconditionally wired to
    chromadb today, not every feature that could theoretically be
    reconfigured to use it."""
    culprits = []
    if getattr(crew, "knowledge_sources", None):
        culprits.append("Crew.knowledge_sources")
    for agent in getattr(crew, "agents", None) or []:
        if getattr(agent, "knowledge", None) or getattr(agent, "knowledge_sources", None):
            culprits.append(f"Agent({getattr(agent, 'role', '?')!r}).knowledge")
    if culprits:
        raise RuntimeError(
            "sevenbox_adapters.crewai_adapter: this Crew has Knowledge "
            f"enabled ({', '.join(culprits)}), which crewai backs "
            "unconditionally with chromadb. chromadb currently carries an "
            "unpatched CVSS-9.3 pre-auth RCE in its HTTP server "
            "(CVE-2026-45829) -- see docs/security-review.md section 1 "
            "before proceeding. If you've read it and the residual risk is "
            "acceptable for your deployment (or a patch has since shipped "
            "-- re-check `pip-audit`'s fix_versions for chromadb first), "
            "pass acknowledge_chromadb_risk=True to wrap()."
        )


def sevenbox_agent_run(*, store: Optional[WarrantStore] = None) -> "_SevenboxAgentRunContext":
    """No `replay_from` parameter -- see sevenbox_adapters.crewai_adapter's
    module docstring, "Replay: NOT supported by this adapter"."""
    _ensure_global_hooks_registered()
    return _SevenboxAgentRunContext(CrewSession(store=store))


class _SevenboxAgentRunContext:
    def __init__(self, session: CrewSession) -> None:
        self.session = session
        self._token = None
        self._wrapped_crews: list = []

    def wrap(
        self,
        crew: Any,
        *,
        source_kind: SourceKind = SourceKind.ORACLE,
        epistemic: EpistemicClass = EpistemicClass.INFERRED,
        reversibility: ReversibilityClass = ReversibilityClass.PURE,
        replay_from: Optional[WarrantStore] = None,
        acknowledge_chromadb_risk: bool = False,
    ) -> Any:
        """Install a task_callback on `crew` that records each completed
        Task as a Layer-1 Warrant. If `crew` already has a task_callback,
        it is preserved and called too (Sevenbox's callback runs first).

        `replay_from`, if given, gives REAL non-execution replay for
        Layer 2 (tool calls) only -- see module docstring's "Replay"
        section for the exact, verified scope and its honest limits
        (Layer 1 Task/LLM calls still execute for real even in replay).

        Raises if `crew` (or any of its agents) has Knowledge
        (`knowledge`/`knowledge_sources`) enabled, unless
        `acknowledge_chromadb_risk=True` is passed explicitly --
        `crewai.knowledge.storage.knowledge_storage.KnowledgeStorage`
        imports `crewai.rag.chromadb` unconditionally, no alternative
        backend. `memory=True` alone does NOT trigger this (verified
        against the installed crewai: its default storage is `lancedb`,
        not chromadb -- see `_reject_if_chromadb_backed_features_enabled`'s
        docstring for the check that ruled this out rather than assumed
        it). chromadb currently carries an unpatched, CVSS-9.3 pre-auth
        RCE (CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c) in its HTTP server --
        see docs/security-review.md section 1. That CVE's actual trigger
        is chromadb's server being network-reachable, which Knowledge
        alone does not do either (it uses chromadb embedded, not as a
        server) -- this check exists so that decision gets made
        consciously, with the CVE in view, by whoever first turns
        Knowledge on for a real Crew, instead of silently, by someone who
        never knew this warning existed. Before setting this flag:
        re-run `pip-audit` and check whether chromadb's advisory now
        lists a `fix_versions` entry -- if it does, upgrade instead of
        acknowledging the risk."""
        if not acknowledge_chromadb_risk:
            _reject_if_chromadb_backed_features_enabled(crew)
        session = self.session
        previous_callback = crew.task_callback

        def _combined_task_callback(output):
            task = self._find_task_for_output(crew, output)
            if task is not None:
                session.record_task_output(
                    task=task, output=output, source_kind=source_kind,
                    epistemic=epistemic, reversibility=reversibility,
                )
            if previous_callback is not None:
                previous_callback(output)

        crew.task_callback = _combined_task_callback
        if replay_from is not None:
            self._install_replay_cache(crew, replay_from)
        self._wrapped_crews.append(crew)
        return crew

    @staticmethod
    def _install_replay_cache(crew: Any, replay_from: WarrantStore) -> None:
        """Pre-populate every agent's native tool-result cache from a
        reference store's captured effect_records, matching CrewAI's own
        cache-key format (tool_name, json.dumps(args)) exactly -- see
        module docstring's "Replay" section for why this achieves real
        non-execution replay without any new interception point."""
        if not _CREWAI_AVAILABLE:
            raise RuntimeError("sevenbox_adapters.crewai_adapter requires the 'crewai' package.")
        cache = _CrewCacheHandler()
        populated = 0
        for w in replay_from.all():
            if w.effect_record is None:
                continue
            request = w.effect_record.request
            if not isinstance(request, dict) or "args" not in request:
                continue  # a Layer-1 (Task) effect_record, not Layer 2 -- no native cache for those
            tool_name = w.effect_record.source_name
            input_str = json.dumps(request["args"])
            cache.add(tool=tool_name, input=input_str, output=w.effect_record.response)
            populated += 1
        for agent in crew.agents:
            agent.tools_handler.cache = cache

    @staticmethod
    def _find_task_for_output(crew: Any, output: Any) -> Any:
        for t in crew.tasks:
            if t.output is output:
                return t
        return None

    def __enter__(self) -> "_SevenboxAgentRunContext":
        self._token = _current_crew_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_crew_session.reset(self._token)
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
    """Opt a CrewAI tool function into Layer 2 (tool-call-level, honestly
    approximate) capture. Unlike the other three adapters, this does NOT
    wrap the function's execution directly -- CrewAI's own tool-calling
    machinery already runs it; this only REGISTERS the tool's name so
    the global after_tool_call hook (see module docstring) knows to
    record a Warrant for it. The function itself is returned unmodified,
    so `@sevenbox_tool` can be stacked with `@tool` from crewai.tools in
    either order.
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or getattr(f, "__qualname__", getattr(f, "__name__", "tool"))
        _SEVENBOX_TOOL_REGISTRY[tool_name] = {
            "source_kind": source_kind, "epistemic": epistemic,
            "reversibility": reversibility, "redaction": redaction, "origin": origin,
        }
        return f

    return decorator(fn) if fn is not None else decorator
