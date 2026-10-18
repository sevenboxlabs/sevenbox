"""
sevenbox_adapters.letta_adapter -- the "sevenbox wrap" story for Letta
(`letta-client` on PyPI, the official Python client for the Letta API).

## Which package: `letta-client`, not `letta`

`letta` (the bare-name PyPI package) is the full Letta *server* --
FastAPI app, SQLAlchemy ORM, Alembic migrations, and (as of 0.16.8) an
import-time dependency on `asyncpg` even when running on SQLite. It also
ships THREE coexisting internal agent-loop implementations in the same
version (`letta.agents.letta_agent_v2`/`letta_agent_v3`, plus separate
ephemeral/voice/sleeptime variants) -- verified by installing it and
reading `letta/agents/` directly, not assumed. That is a strong, real
signal of internal churn, the same kind of instability
`llamaindex_adapter.py`'s and `openhands_adapter.py`'s own docstrings
warn about for *their* unexported internals. `letta-client` is a
separate, much lighter, Fern-generated REST SDK (`pip install
letta-client`, no server/ORM/DB dependencies at all) -- the same stable,
public, versioned boundary every real Letta application already talks
to, whether the server is self-hosted or Letta Cloud. This adapter
targets `letta-client` specifically, for the same reason
`llamaindex_adapter.py` chose `llama-index-core` over the `llama-index`
meta-package and `openhands_adapter.py` chose `openhands-sdk` over its
PyPI-adjacent alternatives: verify which package is the real, intended
integration point before writing any code, never default to whichever
name matches the framework's own.

## The 8 questions, answered from real investigation

Verified against the actual installed `letta-client==1.12.1` -- but,
unlike every other adapter in this package, network-free testing here
does NOT mean a hand-rolled in-process model double. Letta's real
boundary is an HTTP API (`POST /v1/agents/{agent_id}/messages`), so
"network-free" here means injecting a custom `httpx.Client` backed by
`httpx.MockTransport` into the real `Letta` client (`http_client=`, a
real, public constructor parameter) and returning scripted, schema-
accurate JSON -- confirmed to parse into real, correctly-typed
`LettaResponse`/`ToolCallMessage`/`ToolReturnMessage` Pydantic objects
via a real probe run before any adapter code was written. This exercises
the exact same request/response Pydantic validation a real server's
response would go through; only the transport is faked.

**What this verification method does and does not establish, stated
precisely so it is not overclaimed relative to the other ten adapters**:
every other adapter in this package fakes only the LLM -- the
framework's own real dispatch/orchestration code still runs in-process
against that fake, which is exactly how real framework bugs were found
(CrewAI's `ThreadPoolExecutor` handling, Agent Framework's
`ChatMiddlewareLayer` silently dropping `FunctionMiddleware`, both found
because the real code path executed). Here, mocking the HTTP transport
means Letta's own server-side code -- the real agent loop, the real
tool-dispatch logic, whatever orchestration decisions produce a
`LettaResponse` in the first place -- never executes, not once, in any
test in this file. What is verified is narrower: that this adapter's own
parsing and Warrant-construction logic correctly consumes response
shapes matching the real, installed Pydantic schema. It does NOT verify
that this adapter (or a caller using it) survives Letta's real server
behavior -- if a bug exists in Letta's own real dispatch logic, the
class of bug this project found in CrewAI/Agent Framework's real code,
this testing approach structurally cannot find it, because the server
never runs. This is a real, deliberate trade-off (the alternative was
either a live server requiring Postgres, or reverse-engineering the
unstable internals described above), not a limitation to leave implicit.

1. **Where is a decision observed?** `client.agents.messages.create(
   agent_id, input=...)` returns one `LettaResponse` per call, whose
   `.messages` list already contains the FULL, resolved sequence of
   everything the server-side agent loop did for that turn --
   `SystemMessage`, `UserMessage`, `ReasoningMessage`,
   `ToolCallMessage`, `ToolReturnMessage`, `AssistantMessage`, and a few
   others (`ApprovalRequestMessage`/`ApprovalResponseMessage`/
   `SummaryMessage`/`EventMessage`/`HiddenReasoningMessage`), typed via a
   `message_type`-discriminated union -- confirmed via
   `LettaResponse.model_fields`, not assumed from the name. A
   `.stream()` variant also exists, but was not needed: the
   non-streaming response already carries an exact, pre-resolved batch
   signal (question 3), so this adapter processes one complete
   `LettaResponse` per observation rather than consuming a live event
   stream -- the first adapter in this package where "the whole batch,
   already resolved" is the *simplest* correct design, not a fallback.
2. **Where is a tool call identified?** `ToolCall.tool_call_id` /
   `ToolReturn.tool_call_id`, correlated 1:1 -- confirmed via
   `model_fields` on the real installed Pydantic types, not assumed from
   the name.
3. **Parallel vs. sequential?** Every message type -- `ToolCallMessage`,
   `ToolReturnMessage`, `ReasoningMessage`, `AssistantMessage` -- carries
   its own `step_id`, confirmed via `model_fields`: a real, exact
   per-step identity, the same caliber of signal as OpenHands'
   `llm_response_id` (the cleanest correlation found in this package so
   far), not a heuristic. `ToolCallMessage.tool_calls` is
   `Optional[List[ToolCall]]` -- a single message can carry every
   parallel call from one step as one already-typed list (the same
   whole-batch-before-dispatch shape as LlamaIndex's
   `AgentOutput.tool_calls`), and `ToolReturnMessage.tool_returns` is
   `Optional[List[ToolReturn]]` symmetrically. `AgentState`'s own
   `parallel_tool_calls: Optional[bool]` field, read directly off
   `agents.create()`'s real signature, confirms the framework models
   parallel tool calling as a first-class, named concept server-side,
   not an inferred behavior.
4. **Branching representation?** Not applicable within one agent's own
   step loop, matching every other adapter's stance where the answer is
   "no." A separate `client.conversations` resource (with its own
   `.fork()` method) and `client.runs` resource (for the `background=
   True` async job pattern) both exist but were not exercised in this
   pass -- see "What this pass did not verify" below.
5. **Context/parent propagation?** Not applicable the way it is for
   every in-process adapter in this package -- and this is a different
   KIND of answer, not just a different value: there is no client-side
   `ThreadPoolExecutor`/`run_in_executor` dispatch to isolate via
   `copy_context()` at all, because tool execution happens entirely
   server-side -- the Python client never runs a tool body, and (per the
   verification-method caveat above) this pass never observed Letta's
   real server-side dispatch directly, so no framework-internal
   concurrency hazard was found here, unlike PydanticAI/CrewAI/Deep
   Agents/Agent Framework/OpenHands/LlamaIndex/smolagents, all of which
   had one. The only concurrency this adapter's own code can be exposed
   to is the *calling application* choosing to drive multiple
   `messages.create()` calls from its own threads while sharing one
   session -- a defense against caller misuse, not a framework-dispatch
   safety property (see "A lock that defends against caller-owned
   threads, not framework-internal dispatch" below -- a different
   *question* from every prior adapter's concurrency story, not merely a
   different answer to the same one).
6. **Retries / errors?** `ToolReturn.status` is a real, non-optional
   `Literal['success', 'error']` field (confirmed via `model_fields`,
   not `Optional`) -- Letta's own data model represents a failed tool
   call as a normal, always-present return with `status='error'`, not as
   a missing/dropped message the way smolagents silently discards a
   failed parallel batch's siblings. This adapter records that status
   directly in the Warrant's `effect_record`, matching
   `mastra_adapter.py`/`openhands_adapter.py`'s stance that an error is a
   real, recorded fact -- not inferred to be safer than smolagents'
   behavior in general, just verified to be *this framework's* behavior
   specifically, confirmed with a scripted error response, not assumed
   from the schema alone.
7. **Replay without re-execution?** A narrower property than the six
   fully-live adapters' actual replay support, and described precisely
   here so it is not mistaken for the same guarantee under a different
   name -- the same overclaim risk this project already caught once with
   Mastra's replay story. The six fully-live adapters
   (LangGraph/OpenAI Agents SDK/PydanticAI/CrewAI/Deep Agents/Agent
   Framework) wire a real `ReplayEffectExecutor` into the `Runtime`:
   running the SAME Sevenbox program in "replay" mode transparently
   substitutes a cached response for what would otherwise be a real,
   live tool/model call -- the real call is genuinely skipped. This
   adapter has no equivalent. What is true, and verified (not assumed):
   `SevenboxEventObserver.observe_response()` takes a plain `LettaResponse`
   object and has no idea whether it came from a live HTTP call or from
   deserializing a previously-stored one -- confirmed with a real test
   (`test_observe_response_does_not_distinguish_a_live_response_from_a_stored_one`)
   that constructs a `LettaResponse` from stored JSON and confirms it
   produces Warrant ids identical to the live-equivalent run, via
   `compute_id()`'s own content-addressing. But this only
   demonstrates that the *consumer* is provenance-agnostic -- given a
   response, regardless of where it came from. It does NOT demonstrate
   "avoids a real re-execution," because no real execution happened in
   the first place in any test in this file (the verification-method
   caveat above): there is no live HTTP call being skipped, no
   `ReplayEffectExecutor`-equivalent deciding "call the real server or
   fetch a cached response instead," and no such decision point exists
   in this adapter at all -- a caller who wants that must build and own
   the cache-or-call decision entirely outside this adapter. The
   practical benefit is real but modest: building a replay layer on top
   of this design would face no impedance mismatch (unlike
   LlamaIndex/OpenHands/smolagents' `CodeAgent`, where reaching the real
   substitution point requires subclassing unstable internals) -- but no
   such layer has been built, and "no impedance mismatch if one is built
   later" is not the same claim as "replay is supported."
8. **What does the framework not expose?** Nothing was missing for the
   whole-response-consumer design chosen here. What it does not expose
   to this adapter is any visibility into the tool call itself running
   (question 5) -- by the time this adapter sees anything, the server
   has already finished the entire turn, so this adapter cannot, even in
   principle, observe a tool call "about to run" the way every in-process
   adapter in this package can.

## A lock that defends against caller-owned threads, not framework-internal dispatch

**This is a different question from Mastra's/LlamaIndex's/smolagents'
own "does this adapter need a lock" sections, not another answer to the
same one** -- worth stating plainly, since lining it up as a fourth
entry in that sequence would misrepresent what was actually tested.
Mastra/LlamaIndex/smolagents were all answering "is this adapter safe
against the FRAMEWORK's own internal concurrent tool dispatch (a real
`ThreadPoolExecutor`/`run_in_executor` the framework itself spawns
within one turn)?" -- a question this pass could not even ask for Letta,
because (per the verification-method caveat above) no framework-internal
dispatch was ever observed running at all; Letta's tool execution is
server-side and entirely invisible to this adapter. The question this
section actually answers is narrower and different in kind: "if the
*calling application* chooses, on its own, to drive multiple
`messages.create()` calls from multiple of its own threads while sharing
one `LettaSession`, does that corrupt state?" -- a defense against caller
misuse of a `ContextVar`-based session design, not a claim about
Letta's own concurrency behavior, which remains unverified by this pass
either way.

With that scoped correctly: Letta is the first adapter in this package
where a lock is added *and* proven (against the caller-misuse scenario
just described) to genuinely fix a real hazard, not merely fail to make
things worse. The reason is structural:
`observe_response()` is not a handful of short, single-statement session
methods the way `LlamaIndexSession`/`SmolagentsSession`'s mutating
methods are -- it loops over potentially many steps, reading and
updating `self._frontier` once per step as it walks the response's
`.messages` list. Without a lock, two threads sharing one `LettaSession`
and each processing their own multi-step `LettaResponse` concurrently
can genuinely interleave mid-loop (CPython's GIL time-slices between
bytecode instructions, it does not protect a multi-statement loop body),
producing a torn frontier that reflects neither response's steps
correctly -- confirmed with a real test forcing the interleaving
deterministically
(`test_KNOWN_LIMITATION_unlocked_concurrent_responses_can_tear_the_frontier_mid_processing`).
With the lock `LettaSession` actually ships (verified in the paired test
`test_locked_concurrent_responses_do_not_tear_the_frontier`), each
response's steps are applied atomically relative to the other thread's
-- the torn-frontier corruption is gone. **This does not mean two
independent agents sharing one session get correct causal isolation**:
if agent_a's and agent_b's responses are both processed against the same
shared session, whichever is processed second still causally chains onto
whichever finished first, same as every other adapter's documented
"safe pattern is one session per agent, or sequential sharing" -- that is
a usage question, not a bug, and unaffected by the lock. What the lock
fixes is real, low-level data corruption (a frontier reflecting a
half-applied response); what it does not and cannot fix is the
higher-level question of whether two logically-independent turns should
share causal lineage at all -- the same distinction
`smolagents_adapter.py`'s own lock test draws, just landing on the
opposite answer for the low-level question because this adapter's
`observe_response()` is shaped differently (a multi-step loop, not a
handful of atomic single-purpose methods).

## Context propagation across caller-owned threads is the caller's job

A real bug, found adversarially while writing the concurrency tests
above, not designed for in advance: a first version of
`test_locked_concurrent_responses_do_not_tear_the_frontier` drove two
plain `threading.Thread(target=krun.observe, args=(response,))` threads
and got a silent `RuntimeError` ("... outside a `with sevenbox_letta_run():`
block") from inside ONE of the two child threads -- swallowed by pytest
as a background-thread warning, not a test failure, caught only because
the resulting causal chain came out suspiciously short (i.e. by checking
the actual result, not by trusting a green test run). Root cause,
confirmed as a general Python behavior and not specific to this adapter:
`_current_letta_session` is a `contextvars.ContextVar`, and
`ContextVar.set()` on one thread does not propagate to a plain new
`threading.Thread` -- each starts with its own default `Context`. Every
other adapter in this package never hit this because their real
concurrency was always internal to the framework's own dispatch
(`ThreadPoolExecutor.submit(copy_context().run, ...)`, verified case by
case for Deep Agents/Agent Framework/LlamaIndex/smolagents). Here, tool
execution is entirely server-side (question 5), so the only concurrency
this adapter's own code can ever be exposed to is the *calling
application* choosing to drive multiple `messages.create()` calls from
multiple threads itself -- meaning propagating context across those
threads is the caller's responsibility, not something this adapter's
`ContextVar`-based design can do on the caller's behalf. The fix used in
the test (and the pattern any multi-threaded caller needs):
`ctx = contextvars.copy_context()`, then
`threading.Thread(target=ctx.run, args=(fn, arg))` -- and a `Context`
object cannot be entered concurrently even from different threads
(`RuntimeError: cannot enter context: ... is already entered`, also
found the same way, from an initial attempt that reused one
`copy_context()` result for both threads), so each thread needs its own
copy, not a shared one.

## Usage

    from letta_client import Letta
    from sevenbox_adapters.letta_adapter import sevenbox_letta_run

    client = Letta(base_url="http://localhost:8283")

    with sevenbox_letta_run() as krun:
        krun.session.start("check order 4471")
        response = client.agents.messages.create(
            agent_id=agent_id, input="check order 4471",
        )
        krun.observe(response)
        print(krun.session.frontier())

## What this pass did not verify

`client.conversations` (with its own `.fork()` -- possibly Letta's real
branching primitive) and `client.runs`/`background=True` (the async job
pattern) were both found during investigation but not exercised.

More importantly, restated once more because it is the central caveat
of this whole adapter, not a footnote: **no test in this file ever runs
Letta's own real server-side code**. Every other adapter in this package
fakes only the LLM and lets the framework's real dispatch/orchestration
logic execute in-process -- the class of testing that found real bugs in
CrewAI's `ThreadPoolExecutor` handling and Agent Framework's
`ChatMiddlewareLayer`. Here, `httpx.MockTransport` replaces the entire
HTTP round trip, so Letta's real agent loop, real tool dispatch, and
real error handling never execute at all, not once, anywhere in this
pass. What was verified is that this adapter correctly consumes response
shapes matching the real, installed Pydantic schema -- not that this
adapter (or a caller using it) survives Letta's actual server behavior.
A real bug in Letta's own dispatch logic would not be caught by anything
in this file.
"""
from __future__ import annotations

import json
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
    from letta_client.types.agents.tool_call_message import ToolCallMessage as _ToolCallMessage
    from letta_client.types.tool_return_message import ToolReturnMessage as _ToolReturnMessage
    _LETTA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional extra is absent
    _ToolCallMessage = None
    _ToolReturnMessage = None
    _LETTA_AVAILABLE = False


class LettaSession:
    """Tracks the causal frontier across one or more
    `client.agents.messages.create(...)` responses for one Letta agent
    (or several, sharing this session -- see module docstring's "A lock
    that actually earns its keep" for what sharing does and does not make
    safe).

    Unlike every other session in this package, there is no cross-call
    "open batch" state: each `observe_response()` call is self-contained
    -- it reads the frontier once at the start, walks the response's
    already-resolved `.messages` list step by step, and writes the
    frontier once at the end. That shape is what makes a `threading.Lock`
    genuinely sufficient here (module docstring), unlike
    `LlamaIndexSession`/`SmolagentsSession`.
    """

    def __init__(self, *, store: Optional[WarrantStore] = None) -> None:
        if not _LETTA_AVAILABLE:
            raise RuntimeError(
                "LettaSession requires the 'letta' extra "
                "(pip install -e '.[letta]')."
            )
        self.store = store if store is not None else WarrantStore()
        self._frontier: set = set()
        self._lock = threading.Lock()

    def start(self, initial_input: Any = None) -> None:
        """Optional: root the run on an explicit axiom before the first
        response is observed. If the first response arrives without this
        having been called, the first step roots the run on a fallback
        axiom instead -- never silently parentless."""
        if self._frontier:
            return
        w = create_warrant(
            content=initial_input, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


_current_letta_session: ContextVar[Optional["LettaSession"]] = ContextVar(
    "sevenbox_letta_session", default=None
)


def sevenbox_letta_run(*, store: Optional[WarrantStore] = None) -> "_SevenboxLettaRunContext":
    return _SevenboxLettaRunContext(LettaSession(store=store))


class _SevenboxLettaRunContext:
    def __init__(self, session: LettaSession) -> None:
        self.session = session
        self.observer = SevenboxEventObserver()
        self._token = None

    def __enter__(self) -> "_SevenboxLettaRunContext":
        self._token = _current_letta_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_letta_session.reset(self._token)
        return False

    def observe(self, response: Any) -> None:
        """Call once per `LettaResponse` returned by
        `client.agents.messages.create(...)` (live or, for replay,
        reconstructed from stored JSON -- module docstring question 7)."""
        self.observer.observe_response(response)


class SevenboxEventObserver:
    """Stateless; reads the ambient `LettaSession` from a `ContextVar` at
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
        if not _LETTA_AVAILABLE:
            raise RuntimeError(
                "SevenboxEventObserver requires the 'letta' extra "
                "(pip install -e '.[letta]')."
            )
        self._source_kind = source_kind
        self._epistemic = epistemic
        self._reversibility = reversibility
        self._redaction = redaction
        self._origin = origin

    def observe_response(self, response: Any) -> None:
        session = _current_letta_session.get()
        if session is None:
            raise RuntimeError(
                "A Letta response was observed by SevenboxEventObserver "
                "outside a `with sevenbox_letta_run():` block."
            )
        messages = list(getattr(response, "messages", None) or [])
        with session._lock:
            self._apply_response_locked(session, messages)

    def _apply_response_locked(self, session: "LettaSession", messages: list) -> None:
        steps: "dict[Any, dict]" = {}
        step_order: list = []
        for m in messages:
            step_id = getattr(m, "step_id", None)
            if step_id not in steps:
                steps[step_id] = {"calls": {}, "returns": {}}
                step_order.append(step_id)
            bucket = steps[step_id]
            if _ToolCallMessage is not None and isinstance(m, _ToolCallMessage):
                for tc in _tool_calls_of(m):
                    bucket["calls"][tc.tool_call_id] = tc
            elif _ToolReturnMessage is not None and isinstance(m, _ToolReturnMessage):
                for tr in _tool_returns_of(m):
                    bucket["returns"][tr.tool_call_id] = tr
            # SystemMessage/UserMessage/ReasoningMessage/AssistantMessage/
            # ApprovalRequestMessage/ApprovalResponseMessage/SummaryMessage/
            # EventMessage/HiddenReasoningMessage carry no new causal fact
            # beyond what ToolCallMessage/ToolReturnMessage already
            # capture -- matching this project's own definition of what does and doesn't produce a Warrant
            # (bookkeeping/chatter), they produce no Warrant here.

        if not session._frontier and step_order:
            session.start({"note": "Letta response with no preceding axiom"})

        for step_id in step_order:
            bucket = steps[step_id]
            calls, returns = bucket["calls"], bucket["returns"]
            if not calls:
                continue
            parent_ids = list(session._frontier) if session._frontier else []
            new_ids = set()
            for tool_call_id, tc in calls.items():
                tr = returns.get(tool_call_id)
                if tr is None:
                    # A call this adapter was shown with no matching
                    # return -- Sevenbox cannot fabricate a result it was
                    # never given (module docstring question 6).
                    continue
                w = self._build_warrant(tc, tr, parent_ids)
                session.store.add(w)
                new_ids.add(w.id)
            if new_ids:
                session._frontier = new_ids

    def _build_warrant(self, tc: Any, tr: Any, parent_ids: list):
        args = _parse_arguments(tc.arguments)
        req_payload = {"args": args}
        response_content = _tool_return_content(tr)
        is_error = getattr(tr, "status", None) == "error" or bool(getattr(tr, "is_err", False))

        if self._source_kind in PURE_SOURCE_KINDS:
            return create_warrant(
                content=response_content, source_kind=self._source_kind,
                epistemic=self._epistemic, reversibility=self._reversibility,
                parent_ids=parent_ids, origin=self._origin,
            )
        return create_warrant(
            content=response_content, source_kind=self._source_kind,
            epistemic=self._epistemic, reversibility=self._reversibility,
            parent_ids=parent_ids, origin=self._origin,
            effect_record=EffectRecord(
                request=apply_redaction(req_payload, self._redaction),
                response=apply_redaction(
                    {"content": response_content, "status": "error" if is_error else "success"},
                    self._redaction,
                ),
                source_name=tc.name, redacted=self._redaction is not None,
            ),
        )


def _tool_calls_of(message: Any) -> list:
    calls = getattr(message, "tool_calls", None)
    if calls:
        return list(calls)
    single = getattr(message, "tool_call", None)
    return [single] if single is not None else []


def _tool_returns_of(message: Any) -> list:
    returns = getattr(message, "tool_returns", None)
    if returns:
        return list(returns)
    single_id = getattr(message, "tool_call_id", None)
    if single_id is None:
        return []
    # Single-return shape: the fields live directly on the ToolReturnMessage
    # itself (`tool_return`/`status`), not nested in a `ToolReturn` object.
    return [message]


def _parse_arguments(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    return raw


def _tool_return_content(tr: Any) -> Any:
    value = getattr(tr, "tool_return", None)
    if isinstance(value, list):
        # List[TextContent | ImageContent] -- prefer plain text content.
        texts = [getattr(part, "text", None) for part in value]
        texts = [t for t in texts if t is not None]
        return "\n".join(texts) if texts else value
    return value
