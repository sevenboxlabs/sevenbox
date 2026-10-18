"""
sevenbox_adapters.mastra_adapter -- the Python-side half of Sevenbox's Mastra
integration (`@mastra/core` on npm, a TypeScript-native agent framework
built directly on Vercel AI SDK v5's loop primitives).

## Why this adapter looks structurally different from the other six

Every other adapter in this package instruments a framework running in
the same Python process: a decorator wraps a tool function, a middleware
class hooks into the framework's own call chain, all in-process. Mastra
has no Python SDK at all -- it is TypeScript-only, running in Node. There
is no in-process boundary to attach to from Python.

**The decision, made explicitly before writing any code, was a log
importer, not a live adapter**: the TS side
(`mastra_bridge/sevenboxStepLogger.mjs`) observes a real Mastra `Agent` run
via its native `onStepFinish` hook and writes a structured JSONL log of
raw facts -- tool name, arguments, result, step grouping -- computing NO
Warrant ids, no hashes, nothing calculus-shaped at all. This module reads
that log afterward and constructs the real Warrants. Two alternatives
were considered and rejected, for reasons worth recording:

- **A TS-native Warrant construction path** (porting `compute_id()`'s
  content-addressing hash faithfully to TypeScript, so Node could build
  real Warrants and write them to a store shared with Python via HTTP or
  SQLite) would give full "live adapter" parity with the other six, but
  commits the project to a *second, permanent implementation* of
  content-addressing that has to stay bit-for-bit identical to the Python one forever -- a
  concrete, ongoing risk of silent divergence for a framework that was
  #3 in the adversarial-validation priority list, not #1.
- **A synchronous cross-process bridge** (Mastra calling out to a local
  Python process per tool call, so Python remains the only place a
  Warrant is ever computed, but Sevenbox can influence the run live -- e.g.
  return a cached replay response) gives the most complete parity, but
  requires inventing a synchronous cross-language RPC protocol that does
  not exist anywhere else in this project, for a comparatively low-
  priority framework.

The log-importer approach avoids both costs: `compute_id()` is evaluated
exactly once, in Python, exactly as for every other adapter. The
explicit, accepted cost -- stated here rather than discovered by a future
reader the hard way -- is that this is **not a live adapter**: there is
no in-run distinction between "live" and "replay" the way the other six
adapters have it (see "Replay is different here, not absent" below), and
Sevenbox cannot influence a Mastra run while it is happening (no compensable-
decision gating mid-run, no cached-response short-circuiting of a real
tool call). Both of those would require one of the two rejected
approaches.

## The 8 questions, answered from real investigation

Verified against the actual installed `@mastra/core@1.57.0` and `ai@5.0.228`
packages (Node 24.15.0, npm 11.12.1 -- both confirmed present on this
machine before any of this was attempted), via a real `Agent.generate()`
run with real `createTool()` tools and a hand-rolled object implementing
`@ai-sdk/provider`'s `LanguageModelV2` interface directly (network-free;
the officially-exported `MockLanguageModelV2` from `ai/test` requires
`vitest`/`msw` as peer dependencies purely for its own test harness --
confirmed by trying it first and hitting `ERR_MODULE_NOT_FOUND` for both
in sequence -- so a plain object satisfying the same interface was used
instead, avoiding two dependencies this project has no other use for).
See `mastra_bridge/examples/kyc_scenario_parallel.mjs` and
`kyc_scenario_sequential.mjs` for the exact scripts that produced this
module's test fixtures.

1. **Where is a decision observed?** `Agent.generate()`'s `onStepFinish`
   callback, once per step, after that step's tool calls have ALL already
   executed and resolved -- structurally the AI SDK's own per-step hook,
   which Mastra does not alter the timing of. This is the one hook needed;
   nothing analogous to `wrap_tool_call`/`FunctionMiddleware` is required
   because the step is already complete by the time Sevenbox sees it.
2. **Where is a tool call identified?** `step.toolCalls` and
   `step.toolResults`, matched by `toolCallId`. **Real, verified finding**:
   `step.content` (the AI-SDK-level content-block array for that step) is
   NOT a reliable complete record for a step with more than one tool
   call -- in a real two-parallel-call step, `step.content` carried only
   the second call's tool-call/tool-result pair, not the first, even
   though both calls genuinely executed and both appear correctly in
   `step.toolCalls`/`step.toolResults`. Confirmed this was not a bug in
   the probe's own tools (ruled out by first fixing an unrelated
   `execute()` signature mistake in the same probe and re-running).
   `sevenboxStepLogger.mjs` reads only `toolCalls`/`toolResults`, never
   `content`, because of this.
3. **Parallel vs. sequential?** Both scenarios verified with real runs
   (`kyc_scenario_parallel.mjs`: one step, two tool calls; `kyc_scenario_
   sequential.mjs`: two steps, one call each). Unlike every Python
   adapter in this package, **there is no concurrency hazard to guard
   against on the ingestion side at all**: Node.js tool execution is
   single-threaded/event-loop-cooperative (no `asyncio.to_thread`-style
   real-OS-thread offload the way PydanticAI/CrewAI/Deep Agents/Agent
   Framework all turned out to have), and `onStepFinish` reports a whole,
   already-resolved step atomically in one JS callback invocation with no
   possibility of interleaving with another step's callback. The Python
   side then reads the resulting JSONL file sequentially, single-threaded,
   with no threads or async dispatch of its own -- so unlike every other
   adapter, `import_mastra_log` needed no lock, no in-flight counter, and
   no adversarial concurrency test at all. This is a genuine, positive
   architectural difference, not an oversight -- recorded here with the
   same weight this project gives a negative finding.
4. **Branching representation?** Not applicable within one agent run
   (Mastra's step loop is linear, like every other framework's). Multi-
   agent fan-out (`Agent -> Agent A / Agent B -> Decision`) is supported
   by importing two independent logs into the SAME `WarrantStore` (pass
   the same `store=` to two `import_mastra_log()` calls) and citing both
   calls' returned `.frontier` as the parent set of a manually-constructed
   Decision Warrant -- see `tests/test_mastra_adapter.py
   ::test_two_independent_agent_logs_can_be_merged_into_one_decision`.
   Not verified: any of Mastra's own multi-agent orchestration primitives
   (agent networks, workflows) -- out of scope for this pass, since the
   log-importer boundary makes Sevenbox indifferent to how the log was
   produced, only to its shape.
5. **Context/parent propagation?** Entirely within the importer, in
   Python, single-threaded: `_MastraLogImporter` tracks one `frontier`
   set, frozen as `parent_ids` for every call in a step, then replaced by
   that step's own output ids once the step is done -- the same fold-in
   pattern every other adapter uses for its per-turn frontier, just
   without any concurrency protection because there is provably nothing
   racing here (see question 3).
6. **Retries / errors?** A tool call whose `toolCallId` never appears in
   `step.toolResults` (visible in the log as `"ok": false`) still produces
   a Warrant here, with `content=None` and that fact preserved in
   `effect_record` -- consistent with this project's general stance that
   errors are real, recorded facts, not silently dropped. Not separately
   adversarially tested in this pass (no real Mastra tool failure was
   forced during investigation); flagged as untested, not claimed as
   verified.
7. **Replay without re-execution?** Structurally different from the other
   six adapters, not absent -- see "Replay is different here, not absent"
   below.
8. **What does the framework not expose?** Nothing was missing for the
   log-importer boundary chosen here (`onStepFinish` gives a complete,
   already-resolved step). What Mastra does not expose, and what this
   adapter therefore cannot reach, is anything *inside* one step's
   execution while it is happening -- by design, since this integration
   never runs live inside the same process as the Mastra agent at all.

## Replay is different here, not absent -- and not a stronger version of
## the other adapters' guarantee, a different one

Every other adapter distinguishes a "live" mode (real tool execution,
building an `effect_record`) from a "replay" mode (a `ReplayEffectExecutor`
returns a cached response, keyed on `(source_name, parent_ids)`, without
re-running the real call, WHILE Python is driving a real, live run of the
agent through the real framework -- the interception happens inside that
live loop). `import_mastra_log` cannot offer that guarantee at all,
because Python is never in a Mastra run's loop in the first place (module
docstring, "Why this adapter looks structurally different"): there is no
live Mastra execution for Python to intercept or short-circuit, live or
replayed. What `import_mastra_log` *does* offer is narrower and comes
into play strictly after the fact: the real tool bodies already ran,
inside Node, before Python ever sees a single byte of the log, and
calling `import_mastra_log` a second time on the identical, already-
captured file, into a fresh `WarrantStore`, produces Warrants with
**identical ids** to the first call (content-addressing + idempotent
construction) -- verified by `tests/test_mastra_adapter.py
::test_importing_the_same_log_twice_is_idempotent`. That is
reprocessing-idempotence, not replay-without-execution -- the same
distinction `crewai_adapter.py` draws explicitly for its own Layer 1
(two deterministic live runs converging to byte-identical ids is a
weaker guarantee than "did not execute," and that module does not
conflate the two; see its `test_two_deterministic_live_runs_produce_
identical_warrant_ids_but_both_really_execute`). This module does not
conflate them either: reprocessing a log is always safe and always free
of side effects, but that says nothing about avoiding a live Mastra run
-- there is no live Mastra run this adapter is ever in a position to
avoid, replay, or otherwise influence.

## Usage

TypeScript side (`mastra_bridge/`):

    import { createSevenboxStepLogger } from "./sevenboxStepLogger.mjs";
    const sevenboxLog = createSevenboxStepLogger({ write: (line) => ... });
    sevenboxLog.recordRunStart(userInput);
    const result = await agent.generate(userInput, { onStepFinish: sevenboxLog.onStepFinish });
    sevenboxLog.recordRunFinish(result.text);

Python side:

    from sevenbox_adapters.mastra_adapter import import_mastra_log
    result = import_mastra_log("path/to/log.jsonl")
    print(result.store.all())
    print(result.frontier)  # ids to parent a downstream Decision Warrant on
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Iterable, Optional, TextIO, Union

from warrant_core import (
    EpistemicClass,
    Origin,
    RedactionPolicy,
    ReversibilityClass,
    SourceKind,
    WarrantStore,
    apply_redaction,
    create_warrant,
)
from warrant_core import EffectRecord

LogSource = Union[str, Path, TextIO, Iterable[str]]


@dataclass(frozen=True)
class MastraImportResult:
    """Result of one `import_mastra_log()` call. `frontier` is this
    log's own final causal frontier -- the ids a caller should cite as
    `parent_ids` when chaining a downstream Warrant (e.g. a Decision
    merging two agents' logs) onto what this import produced."""

    store: WarrantStore
    frontier: FrozenSet[str]
    run_axiom_id: Optional[str]


def import_mastra_log(
    source: LogSource,
    *,
    store: Optional[WarrantStore] = None,
    source_kind: SourceKind = SourceKind.TOOL,
    epistemic: EpistemicClass = EpistemicClass.VERIFIED,
    reversibility: ReversibilityClass = ReversibilityClass.PURE,
    redaction: Optional[RedactionPolicy] = None,
    origin: Optional[Origin] = None,
) -> MastraImportResult:
    """Read a JSONL log produced by `mastra_bridge/sevenboxStepLogger.mjs`
    and construct the real Warrants it describes. Never executes
    anything -- see module docstring's "Replay is different here, not
    absent". `source` may be a path, an open text file, or any iterable
    of JSON-line strings (e.g. `log_text.splitlines()`, useful in tests).
    """
    target_store = store if store is not None else WarrantStore()
    frontier: set = set()
    run_axiom_id: Optional[str] = None

    for line in _iter_lines(source):
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        event_type = event.get("type")

        if event_type == "run_start":
            axiom = create_warrant(
                content=event.get("input"),
                source_kind=SourceKind.AXIOM,
                epistemic=EpistemicClass.VERIFIED,
                reversibility=ReversibilityClass.PURE,
                origin=origin,
            )
            target_store.add(axiom)
            frontier = {axiom.id}
            run_axiom_id = axiom.id

        elif event_type == "step":
            if not frontier:
                # A step with no preceding run_start -- root lazily rather
                # than silently dropping the step's parentage, matching
                # every other adapter's fallback-axiom convention.
                axiom = create_warrant(
                    content={"note": "mastra step with no preceding run_start"},
                    source_kind=SourceKind.AXIOM,
                    epistemic=EpistemicClass.VERIFIED,
                    reversibility=ReversibilityClass.PURE,
                    origin=origin,
                )
                target_store.add(axiom)
                frontier = {axiom.id}
                run_axiom_id = axiom.id

            parent_ids = tuple(sorted(frontier))
            new_ids: set = set()
            for call in event.get("calls", []):
                req_payload = {"args": call.get("args")}
                response = call.get("result") if call.get("ok") else None
                w = create_warrant(
                    content=response,
                    source_kind=source_kind,
                    epistemic=epistemic,
                    reversibility=reversibility,
                    parent_ids=parent_ids,
                    origin=origin,
                    effect_record=EffectRecord(
                        request=apply_redaction(req_payload, redaction),
                        response=apply_redaction(response, redaction),
                        source_name=call.get("name"),
                        redacted=redaction is not None,
                    ),
                )
                target_store.add(w)
                new_ids.add(w.id)
            if new_ids:
                frontier = new_ids

        # "run_finish" is informational only (final text output) -- not a
        # Warrant, matching this project's own definition of what does and doesn't produce a Warrant
        # (there is no tool call, no decision, nothing causal about it).

    return MastraImportResult(store=target_store, frontier=frozenset(frontier), run_axiom_id=run_axiom_id)


def _iter_lines(source: LogSource) -> Iterable[str]:
    if isinstance(source, (str, Path)) and _looks_like_path(source):
        with open(source, "r", encoding="utf-8") as f:
            yield from f.readlines()
        return
    if isinstance(source, str):
        yield from source.splitlines()
        return
    if hasattr(source, "read"):
        yield from source.read().splitlines()
        return
    yield from source


def _looks_like_path(source: Union[str, Path]) -> bool:
    if isinstance(source, Path):
        return True
    # A bare JSON-lines string will never parse as an existing path;
    # Path.exists() is a safe, side-effect-free way to disambiguate a
    # path argument from raw log text passed directly (used in tests).
    try:
        return Path(source).exists()
    except OSError:
        return False
