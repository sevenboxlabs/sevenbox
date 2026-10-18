import json

from warrant_core import (
    EffectRecord,
    EpistemicClass,
    ReversibilityClass,
    SourceKind,
    WarrantStore,
    compute_health,
    compute_metrics,
    compute_timeline,
    create_warrant,
    explain,
    inspect,
    supersede,
)

from sevenbox_demo.visualize import (
    render_decision_graph,
    render_explanation,
    render_health_summary,
    render_node_details,
    render_timeline,
)


def _refund_style_store():
    store = WarrantStore()
    root = store.add(create_warrant(
        content={"order_id": "4471"}, source_kind=SourceKind.AXIOM,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
    ))
    intent = store.add(create_warrant(
        content={"intent": "refund_request"}, source_kind=SourceKind.ORACLE,
        epistemic=EpistemicClass.INFERRED, reversibility=ReversibilityClass.PURE,
        parent_ids=(root.id,),
        effect_record=EffectRecord(request="infer", response="refund_request", source_name="infer_intent"),
    ))
    elig = store.add(create_warrant(
        content={"eligible": True}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        parent_ids=(intent.id,),
    ))
    action = store.add(create_warrant(
        content={"action": "refund_action(4471)"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.IRREVERSIBLE,
        parent_ids=(elig.id,),
    ))
    return store, root, intent, elig, action


def test_render_decision_graph_is_json_safe_and_carries_every_semantic_field():
    """Functional coverage of the new renderer: build a DecisionGraph the
    same way sevenbox_demo.server does (warrant_core.inspect(), not
    hand-assembled JSON), render it, and confirm it's actually JSON-safe end
    to end -- json.dumps() would raise on a stray dataclass/enum, which is
    exactly the class of bug a purely structural test could miss."""
    store, root, intent, elig, action = _refund_style_store()
    correction = supersede(
        store, target_id=elig.id, content={"claim": "already refunded"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="q", response="already refunded"),
    )

    graph = inspect(store)
    payload = render_decision_graph(graph, mode="live", call_counts={"infer_intent": 1})

    # Round-trips through the exact encoder sevenbox_demo.server uses.
    reencoded = json.loads(json.dumps(payload))
    assert reencoded == payload

    assert payload["mode"] == "live"
    assert payload["call_counts"] == {"infer_intent": 1}
    assert payload["warrant_count"] == 5  # root, intent, elig, action, correction

    by_id = {w["id"]: w for w in payload["warrants"]}
    intent_json = by_id[intent.id]
    assert intent_json["effect_metadata"] == {
        "present": True, "request": "infer", "response": "refund_request",
        "source_name": "infer_intent",
    }
    assert intent_json["redaction_metadata"] == {"applied": False}
    assert intent_json["replay_available"] == {
        "available": True, "degraded": False,
        "reason": "Replay available.",
    }

    action_json = by_id[action.id]
    assert action_json["compensation"]["direct"] is None
    assert len(action_json["compensation"]["transitive"]) == 1
    assert action_json["compensation"]["transitive"][0]["corrector_id"] == correction.id

    elig_json = by_id[elig.id]
    assert elig_json["is_live"] is False
    assert elig_json["superseded_by_ids"] == [correction.id]
    assert elig_json["compensation"]["direct"]["action_required"] is False

    assert payload["frontier"]["live_ids"] == graph.frontier.live_ids
    assert len(payload["obligations"]) == 1
    assert payload["obligations"][0]["irreversible_id"] == action.id


def test_render_node_details_is_the_same_projection_used_inside_the_warrant_list():
    """The node inspector panel calls render_node_details() to get the
    exact same shape already embedded in render_decision_graph()'s
    `warrants` list -- proving there is no second, parallel representation
    of a node computed anywhere, just one function reused."""
    store, root, intent, elig, action = _refund_style_store()
    graph = inspect(store)
    payload = render_decision_graph(graph, mode="live", call_counts={})

    intent_order = graph.nodes.index(next(n for n in graph.nodes if n.warrant_id == intent.id))
    intent_node = next(n for n in graph.nodes if n.warrant_id == intent.id)
    standalone = render_node_details(intent_node, order=intent_order)

    from_list = next(w for w in payload["warrants"] if w["id"] == intent.id)
    assert standalone == from_list

    reencoded = json.loads(json.dumps(standalone))
    assert reencoded == standalone


def test_render_explanation_is_json_safe():
    store, root, intent, elig, action = _refund_style_store()
    explanation = explain(store, action.id)
    payload = render_explanation(explanation)

    reencoded = json.loads(json.dumps(payload))
    assert reencoded == payload

    assert payload["target_id"] == action.id
    assert len(payload["nodes"]) == 4  # action, elig, intent, root
    assert len(payload["narrative"]) == 4


def test_render_health_summary_is_json_safe_and_never_reinterprets_status():
    store, root, intent, elig, action = _refund_style_store()
    supersede(
        store, target_id=elig.id, content={"claim": "already refunded"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="q", response="already refunded"),
    )

    graph = inspect(store)
    metrics = compute_metrics(graph)
    health = compute_health(metrics)
    payload = render_health_summary(metrics, health)

    reencoded = json.loads(json.dumps(payload))
    assert reencoded == payload

    assert payload["metrics"]["total_decisions"] == metrics.total_decisions
    assert payload["metrics"]["pending_compensation_count"] == 1
    # health.overall is CRITICAL because of the pending compensation --
    # the renderer must pass that categorical judgment through unchanged,
    # never recompute it from the raw counts.
    assert payload["health"]["overall"] == "critical"
    signal_names = {s["name"] for s in payload["health"]["signals"]}
    assert signal_names == {"compensation_backlog", "replay_degraded", "frontier_conflicts"}


def test_render_timeline_is_json_safe_and_passes_event_kind_through_unchanged():
    store, root, intent, elig, action = _refund_style_store()
    correction = supersede(
        store, target_id=elig.id, content={"claim": "already refunded"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="q", response="already refunded"),
    )

    timeline = compute_timeline(inspect(store))
    payload = render_timeline(timeline)

    reencoded = json.loads(json.dumps(payload))
    assert reencoded == payload

    by_id = {e["warrant_id"]: e for e in payload["entries"]}
    assert by_id[root.id]["event_kind"] == "axiom"
    assert by_id[intent.id]["event_kind"] == "evidence"
    assert by_id[action.id]["event_kind"] == "irreversible_action"
    # `elig` was superseded by `correction` -- the renderer must reflect
    # that classification exactly as warrant_core computed it, never
    # re-derive "is this a correction" from supersedes_id itself.
    assert by_id[correction.id]["event_kind"] == "correction"
    assert by_id[correction.id]["supersedes_id"] == elig.id
    assert [e["order"] for e in payload["entries"]] == list(range(len(payload["entries"])))


def test_visualize_module_never_binds_a_store_facing_query_function():
    """Architectural guard, not just a convention: sevenbox_demo.visualize must
    never import WarrantStore or any warrant_core function that queries a
    store or computes a summary (inspect, explain, compute_frontier,
    compensation_for, find_transitive_obligations, compute_metrics,
    compute_health, compute_timeline). If one of these ever gets imported
    back in, it's a sign business logic is creeping back into the renderer
    -- exactly the mistake this module's docstring exists to prevent. This
    test fails loudly on that regression instead of relying on code review
    to catch it.
    """
    import sevenbox_demo.visualize as visualize_module

    forbidden_names = {
        "WarrantStore",
        "SqliteWarrantStore",
        "inspect",
        "explain",
        "compute_frontier",
        "compensation_for",
        "find_transitive_obligations",
        "compute_metrics",
        "compute_health",
        "compute_timeline",
    }
    bound_names = set(vars(visualize_module))
    leaked = forbidden_names & bound_names
    assert not leaked, (
        f"sevenbox_demo.visualize bound store-facing query function(s) {leaked} -- "
        f"this module must only format warrant_core domain objects it's handed, "
        f"never query a store itself. Move this call to the caller (e.g. "
        f"sevenbox_demo.server) and pass the resulting domain object in instead."
    )
