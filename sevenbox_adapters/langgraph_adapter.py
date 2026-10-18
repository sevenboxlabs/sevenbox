"""
sevenbox_adapters.langgraph_adapter -- the "sevenbox wrap" story for LangGraph.

Verified against the actual installed `langgraph==1.2.10` package, not against
training-data assumptions about its API -- LangGraph's internals move fast.

Integration surface, deliberately small:

    from sevenbox_adapters.langgraph_adapter import sevenbox_node, sevenbox_run
    from warrant_core import SourceKind, ReversibilityClass

    graph.add_node("infer_intent", sevenbox_node(infer_intent, source_kind=SourceKind.ORACLE))
    graph.add_node("lookup_order", sevenbox_node(lookup_order, source_kind=SourceKind.TOOL))
    graph.add_node("refund", sevenbox_node(refund, reversibility=ReversibilityClass.IRREVERSIBLE))

    compiled = graph.compile()

    with sevenbox_run() as krun:
        result = compiled.invoke(initial_state)
        print(krun.session.frontier().live)

Two changes to an existing LangGraph app: wrap each `add_node` call, wrap the
`.invoke()` call site in `sevenbox_run()`. Node bodies are untouched.

## Design, and its honest limitations

This is a linear-chain model: each wrapped node's output becomes a new Warrant
whose single parent is "whatever the most recently produced Warrant in this run
was." This matches the common sequential agent shape (read state -> call model ->
call tool -> update state) and is what's tested here. It does NOT correctly
attribute parallel/fan-out node execution -- if LangGraph runs two nodes
concurrently from the same state, both will be chained onto the same "latest"
pointer in whatever order they happen to execute, which is a simplification, not
a faithful causal graph of a branching run. Fixing that properly needs per-branch
state tracking, which is future work, not this slice -- flagged here rather than
silently mismodeled.

`source_kind` defaults to `SourceKind.TOOL` (effectful), not `SourceKind.DERIVED`
(pure) -- deliberately conservative. A wrapped node might secretly call an LLM
inside its body; the adapter cannot know this by inspection, and assuming "pure"
by default would silently break the confluence/replay guarantees the moment a
node does anything non-deterministic. Developers who know a node is genuinely
pure can opt in with `source_kind=SourceKind.DERIVED`.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable, Optional

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

# Reserved key a pure (DERIVED) node's returned dict can include to declare
# EXTRA causal parents beyond the session's current frontier -- see
# GraphSession.record_node()'s docstring for why this exists and why it's
# deliberately pure-node-only.
ADDITIONAL_PARENTS_KEY = "__sevenbox_additional_parents__"


def _extract_additional_parents(output: Optional[dict]) -> "tuple[Optional[dict], list]":
    if isinstance(output, dict) and ADDITIONAL_PARENTS_KEY in output:
        output = dict(output)  # never mutate the caller's returned dict
        extra = list(output.pop(ADDITIONAL_PARENTS_KEY))
        return output, extra
    return output, []


class GraphSession:
    """Tracks the causal frontier of a single graph invocation.

    Proposal E (docs/design-review-organizations.md): this used to track a
    single scalar `_latest_warrant_id`, chaining every node's output onto
    whatever the previous node produced. That's correct for a strictly
    sequential graph and silently wrong the moment two sibling nodes run
    from the same prior state -- both would chain onto whichever one
    happened to update the shared pointer first, making one sibling a
    (false) causal parent of the other.

    What changed, and what honestly didn't: `_latest_warrant_id` is now
    `_frontier`, a set. For a sequential graph -- the case every existing
    test and example here actually exercises -- the frontier is always
    exactly one element, so behavior is unchanged. What this does NOT do is
    automatically detect true LangGraph-level fan-out/fan-in: a wrapped node
    function only ever receives LangGraph's already-merged `state` dict, with
    no breadcrumb identifying which prior Warrant(s) contributed to it --
    and getting that information would mean reaching into LangGraph's
    private graph-topology internals, which `sevenbox_node`'s public-API-only
    design deliberately avoids (see the module docstring's "why not fully
    zero-touch" section for the identical trade-off made in Phase 4).

    Instead: a node can EXPLICITLY declare extra causal parents by including
    `ADDITIONAL_PARENTS_KEY` (a list of Warrant ids) in its returned dict --
    real, correct multi-parent attribution for a node that knows it's
    joining two branches, without guessing at LangGraph's topology. This
    only works for pure (DERIVED) nodes: an effectful node's real function
    body is never called during replay (that's the entire point of replay),
    so there is no way to *discover* its additional parents without
    defeating replay -- attempting this on an effectful node raises loudly
    at call time instead of silently losing the extra parents during replay.
    """

    def __init__(
        self,
        *,
        store: Optional[WarrantStore] = None,
        replay_from: Optional[WarrantStore] = None,
    ) -> None:
        self.store = store if store is not None else WarrantStore()
        self.mode = "replay" if replay_from is not None else "live"
        self._replay_executor = (
            ReplayEffectExecutor(reference_store=replay_from) if replay_from is not None else None
        )
        self._frontier: set = set()

    def _ensure_started(self, initial_state: Any) -> None:
        if self._frontier:
            return
        w = create_warrant(
            content=initial_state, source_kind=SourceKind.AXIOM,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        )
        self.store.add(w)
        self._frontier = {w.id}

    def record_node(
        self,
        *,
        node_name: str,
        source_kind: SourceKind,
        epistemic: EpistemicClass,
        reversibility: ReversibilityClass,
        input_state: Any,
        real_fn: Callable[[Any], Optional[dict]],
        redaction: Optional[RedactionPolicy] = None,
        origin: Optional[Origin] = None,
    ) -> Optional[dict]:
        self._ensure_started(input_state)
        parent_ids = sorted(self._frontier)

        if source_kind in PURE_SOURCE_KINDS:
            output = real_fn(input_state)
            if output is None:
                return None
            output, extra_parents = _extract_additional_parents(output)
            if extra_parents:
                unknown = [pid for pid in extra_parents if pid not in self.store]
                if unknown:
                    raise ValueError(
                        f"Node {node_name!r} declared additional parent id(s) not present "
                        f"in the store: {unknown}."
                    )
                parent_ids = sorted(set(parent_ids) | set(extra_parents))
            w = create_warrant(
                content=output, source_kind=source_kind,
                epistemic=epistemic, reversibility=reversibility, parent_ids=parent_ids,
                origin=origin,
            )
        else:
            if self.mode == "live":
                output = real_fn(input_state)
                if output is None:
                    return None
                if isinstance(output, dict) and ADDITIONAL_PARENTS_KEY in output:
                    raise ValueError(
                        f"Node {node_name!r} is effectful (source_kind={source_kind.value}) "
                        f"and returned {ADDITIONAL_PARENTS_KEY!r} -- additional-parent "
                        f"declaration is only supported for pure (DERIVED) nodes, because "
                        f"an effectful node's real function body is never called during "
                        f"replay, so its declared extra parents could never be recovered "
                        f"there. Restructure so the extra-parent decision happens in a "
                        f"downstream DERIVED node instead."
                    )
                raw_response = output
            else:
                assert self._replay_executor is not None
                cached = self._replay_executor.invoke(node_name, parent_ids, request=input_state)
                output = cached.content
                raw_response = cached.raw
            w = create_warrant(
                content=output, source_kind=source_kind,
                epistemic=epistemic, reversibility=reversibility,
                parent_ids=parent_ids, origin=origin,
                effect_record=EffectRecord(
                    request=apply_redaction(input_state, redaction),
                    response=apply_redaction(raw_response, redaction),
                    source_name=node_name,
                    redacted=redaction is not None,
                ),
            )

        self.store.add(w)
        self._frontier = {w.id}
        return output

    def frontier(self):
        return compute_frontier(self.store)

    def obligations(self):
        return find_transitive_obligations(self.store)


_current_graph_session: ContextVar[Optional[GraphSession]] = ContextVar(
    "sevenbox_graph_session", default=None
)


def sevenbox_run(
    *, store: Optional[WarrantStore] = None, replay_from: Optional[WarrantStore] = None
) -> "_SevenboxRunContext":
    return _SevenboxRunContext(GraphSession(store=store, replay_from=replay_from))


class _SevenboxRunContext:
    def __init__(self, session: GraphSession) -> None:
        self.session = session
        self._token = None

    def __enter__(self) -> "_SevenboxRunContext":
        self._token = _current_graph_session.set(self.session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_graph_session.reset(self._token)
        return False


def sevenbox_node(
    fn: Callable[[Any], Optional[dict]],
    *,
    source_kind: SourceKind = SourceKind.TOOL,
    epistemic: EpistemicClass = EpistemicClass.VERIFIED,
    reversibility: ReversibilityClass = ReversibilityClass.PURE,
    name: Optional[str] = None,
    redaction: Optional[RedactionPolicy] = None,
    origin: Optional[Origin] = None,
) -> Callable[[Any], Optional[dict]]:
    node_name = name or getattr(fn, "__qualname__", getattr(fn, "__name__", "node"))

    def wrapped(state: Any) -> Optional[dict]:
        session = _current_graph_session.get()
        if session is None:
            raise RuntimeError(
                f"Node {node_name!r} was called outside a `with sevenbox_run():` block. "
                f"Wrap your graph.invoke(...) call site in `with sevenbox_run() as krun:`."
            )
        return session.record_node(
            node_name=node_name, source_kind=source_kind, epistemic=epistemic,
            reversibility=reversibility, input_state=state, real_fn=fn,
            redaction=redaction, origin=origin,
        )

    wrapped.__name__ = f"sevenbox_wrapped_{node_name}"
    wrapped.__qualname__ = wrapped.__name__
    return wrapped
