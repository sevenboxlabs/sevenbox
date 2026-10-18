"""
sevenbox_demo.visualize -- pure rendering of warrant_core's domain objects
(DecisionGraph, Explanation) into the JSON shape sevenbox_demo's frontend
consumes.

ARCHITECTURAL BOUNDARY, enforced deliberately, not just by convention: no
function in this module accepts a WarrantStore, and no function here calls
inspect(), explain(), compute_frontier(), compensation_for(),
compute_metrics(), compute_health(), compute_timeline(), or any other
warrant_core query. Every semantic fact -- what's live, what's owed, what's
replayable, what was redacted, even a Warrant's causal depth, whether the
system is healthy, or what *kind* of event a Warrant represents on a
timeline -- is computed exactly once, in warrant_core, by the caller (today,
sevenbox_demo.server; tomorrow, potentially a CLI, a VS Code extension, or a
REST API). This module only decides presentation: truncating an id for
display, shaping a dataclass into a JSON-safe dict. Graph-layout "layer"
used to be computed here by re-walking causal_relationships -- moved to
DecisionNode.causal_relationships.depth in warrant_core once
DecisionMetrics needed the identical computation for average_causal_depth.
Two consumers needing the same fact was the signal that it was never a
rendering choice to begin with. `render_health_summary()` follows the same
rule for the Operations Center home view: it shapes an already-computed
DecisionMetrics/DecisionHealth pair, never decides which numbers count as
"healthy" -- that categorical judgment is `health.overall`/`signal.status`,
computed once in warrant_core.decision_health.

This is the whole point of sevenbox_demo not being "the demo" anymore. It is the
first *client* of warrant_core.DecisionGraph -- proof that the model is
genuinely renderer-agnostic, not a description of one. A second renderer
(CLI, REST API, ...) should be able to sit next to this one, call the exact
same warrant_core.inspect()/explain(), and never need to duplicate a single
fact this module computes -- because this module doesn't compute any facts,
it only formats ones it was handed. If a future change here ever needs to
answer a question about the graph that isn't already a field on
DecisionGraph/Explanation, that's a sign the answer belongs in warrant_core,
not in a conditional bolted onto a render function.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from warrant_core import (
    CompensationObligation,
    DecisionGraph,
    DecisionHealth,
    DecisionMetrics,
    DecisionNode,
    DecisionTimeline,
    Explanation,
    Origin,
    TransitiveCompensationObligation,
)


def _render_direct_compensation(obligation: Optional[CompensationObligation]) -> Optional[Dict[str, Any]]:
    if obligation is None:
        return None
    # `.target` is omitted deliberately: it's this same node, already being
    # rendered -- repeating it here would be presentation-layer duplication
    # of a fact the caller already has.
    return {
        "action_required": obligation.action_required,
        "ambiguous": obligation.ambiguous,
        "reason": obligation.reason,
        "competing_superseder_ids": obligation.competing_superseder_ids,
    }


def _render_transitive_compensation(obligation: TransitiveCompensationObligation) -> Dict[str, Any]:
    # `.irreversible` omitted for the same reason as `.target` above --
    # this list is already scoped to one node's DecisionNode.
    return {
        "superseded_ancestor_id": obligation.superseded_ancestor_id,
        "corrector_id": obligation.corrector_id,
    }


def _render_origin(origin: Optional[Origin]) -> Optional[Dict[str, Any]]:
    if origin is None:
        return None
    return {
        "org_id": origin.org_id,
        "team_id": origin.team_id,
        "role_id": origin.role_id,
        "agent_id": origin.agent_id,
        "run_id": origin.run_id,
    }


def _render_graph_level_obligation(obligation: TransitiveCompensationObligation) -> Dict[str, Any]:
    # Unlike the per-node rendering above, a graph-wide obligations list
    # does need to identify which warrant is irreversible -- there's no
    # enclosing node to imply it.
    return {
        "irreversible_id": obligation.irreversible.id,
        "irreversible_content": obligation.irreversible.content,
        "superseded_ancestor_id": obligation.superseded_ancestor_id,
        "corrector_id": obligation.corrector_id,
    }


def render_node_details(node: DecisionNode, *, order: int) -> Dict[str, Any]:
    """The complete projection of a single DecisionNode -- every field the
    "generic metadata inspector" needs (effect metadata, source metadata,
    redaction status, replay availability, compensation links, supersession
    history, causal relationships), in one shape.

    Reused, unchanged, in two places: once per node inside
    `render_decision_graph()`'s `warrants` list, and, client-side, as the
    Explorer's node inspector panel -- which looks this exact record up
    from the already-fetched `warrants` list by id rather than fetching a
    second, separate "node detail" shape. `render_node_details()` is the
    only place this projection is computed; the inspector is a *view* of a
    node already known to the graph, never a second, parallel model built
    to describe it. There is no `NodeInspection` type in warrant_core, and
    deliberately so -- `DecisionNode` already carries everything this
    function projects; a second type describing the same facts a different
    way would violate "composition over new types" for no reason.
    """
    n = node
    return {
        "id": n.warrant_id,
        "short_id": n.warrant_id[:8],
        "content": n.content,
        "source_kind": n.source_metadata.source_kind,
        "epistemic": n.source_metadata.epistemic,
        "reversibility": n.reversibility,
        "effectful": n.source_metadata.effectful,
        "origin": _render_origin(n.source_metadata.origin),
        "parent_ids": n.causal_relationships.parent_ids,
        "child_ids": n.causal_relationships.child_ids,
        "supersedes_id": n.supersession_relationships.supersedes_id,
        "superseded_by_ids": n.supersession_relationships.superseded_by_ids,
        "layer": n.causal_relationships.depth,
        "order": order,
        "is_live": n.supersession_relationships.is_live,
        "effect_metadata": {
            "present": n.effect_metadata.present,
            "request": n.effect_metadata.request,
            "response": n.effect_metadata.response,
            "source_name": n.effect_metadata.source_name,
        },
        "redaction_metadata": {"applied": n.redaction_metadata.applied},
        "replay_available": {
            "available": n.replay_available.available,
            "degraded": n.replay_available.degraded,
            "reason": n.replay_available.reason,
        },
        "compensation": {
            "direct": _render_direct_compensation(n.compensation_relationships.direct),
            "transitive": [
                _render_transitive_compensation(o)
                for o in n.compensation_relationships.transitive
            ],
        },
    }


def render_decision_graph(
    graph: DecisionGraph, *, mode: str = "live", call_counts: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Format an already-computed DecisionGraph as JSON. `mode` and
    `call_counts` are the only inputs that don't come from warrant_core --
    both describe how the caller obtained `graph` (a live run vs. a replay,
    and how many real handlers fired), not a fact about the graph itself.
    """
    warrant_json = [render_node_details(n, order=order) for order, n in enumerate(graph.nodes)]

    return {
        "mode": mode,
        "call_counts": call_counts or {},
        "warrant_count": len(graph.nodes),
        "warrants": warrant_json,
        "frontier": {
            "live_ids": graph.frontier.live_ids,
            "conflicts": [
                {"target_id": c.target_id, "competing_superseder_ids": c.competing_superseder_ids}
                for c in graph.frontier.conflicts
            ],
        },
        "obligations": [_render_graph_level_obligation(o) for o in graph.obligations],
    }


def render_health_summary(metrics: DecisionMetrics, health: DecisionHealth) -> Dict[str, Any]:
    """Format an already-computed DecisionMetrics/DecisionHealth pair as
    JSON, for the Operations Center home view. No arithmetic happens here
    beyond field-for-field shaping -- ratios, counts, and status labels are
    passed through exactly as warrant_core computed them. The frontend may
    format a ratio as a percentage string or round a float for display, but
    never decides *whether* a number is good or bad -- that's `health`,
    computed once here, not re-derived client-side.
    """
    return {
        "metrics": {
            "total_decisions": metrics.total_decisions,
            "frontier_size": metrics.frontier_size,
            "superseded_count": metrics.superseded_count,
            "conflict_count": metrics.conflict_count,
            "pending_compensation_count": metrics.pending_compensation_count,
            "effectful_count": metrics.effectful_count,
            "replayable_count": metrics.replayable_count,
            "degraded_replay_count": metrics.degraded_replay_count,
            "replayable_ratio": metrics.replayable_ratio,
            "redacted_count": metrics.redacted_count,
            "average_causal_depth": metrics.average_causal_depth,
            "source_kind_counts": metrics.source_kind_counts,
            "reversibility_counts": metrics.reversibility_counts,
        },
        "health": {
            "overall": health.overall.value,
            "signals": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "summary": s.summary,
                    "detail": s.detail,
                }
                for s in health.signals
            ],
        },
    }


def render_timeline(timeline: DecisionTimeline) -> Dict[str, Any]:
    """Format an already-computed DecisionTimeline as JSON. `event_kind` is
    passed through as the string warrant_core already classified it as
    (axiom/correction/irreversible_action/evidence) -- this function does
    not re-derive it from source_kind/reversibility/supersedes_id, even
    though it could; doing so would be exactly the "generate events in the
    renderer" mistake the domain layer exists to prevent.
    """
    return {
        "entries": [
            {
                "order": e.order,
                "warrant_id": e.warrant_id,
                "event_kind": e.event_kind.value,
                "source_kind": e.source_kind,
                "reversibility": e.reversibility,
                "content": e.content,
                "is_live": e.is_live,
                "supersedes_id": e.supersedes_id,
            }
            for e in timeline.entries
        ],
    }


def render_explanation(explanation: Explanation) -> Dict[str, Any]:
    """Format an already-computed warrant_core.Explanation as JSON."""
    return {
        "target_id": explanation.target_id,
        "nodes": [
            {
                "warrant_id": n.warrant_id,
                "short_id": n.warrant_id[:8],
                "content": n.content,
                "source_kind": n.source_kind,
                "epistemic": n.epistemic,
                "reversibility": n.reversibility,
                "is_live": n.is_live,
                "depth": n.depth,
            }
            for n in explanation.nodes
        ],
        "edges": [{"parent_id": p, "child_id": c} for p, c in explanation.edges],
        "narrative": explanation.narrative,
    }
