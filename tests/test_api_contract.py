"""
Contract test for sevenbox_demo's HTTP JSON responses.

WHY THIS FILE EXISTS: `sevenbox_demo/server.py`'s `_send_graph_payload()` merges
`render_decision_graph()` and `render_health_summary()` into one JSON
response shared by `/api/run`, `/api/replay`, and `/api/load`. Every prior
test in this project asserts specific *values* on specific fields
("warrant_count == 6", "health.overall == 'critical'") but nothing asserted
the *complete shape* -- every key that's supposed to be there, and nothing
extra. A refactor that silently renamed or dropped a field (the frontend's
only contract with the backend, since sevenbox_demo has no other API consumer
yet) could pass every existing test and still break `app.js` at runtime.

This is deliberately hand-rolled structural assertions, not a JSON Schema
library: this project has zero external dependencies in warrant_core and
avoids them everywhere by default, and
the payload is shallow enough that exact `set(d.keys()) == {...}` checks
per nested shape are both sufficient and easier to read than a schema DSL
would be here.

Treat this as the closest thing sevenbox_demo's demo API has to a stable
contract: it is not versioned or published anywhere else, but this test is
where "what shape does /api/run return" is authoritative and enforced --
see also sevenbox_demo/server.py's own docstring, which documents the same
contract in prose for a human reader.

STRICT BY CHOICE, NOT BY DEFAULT: `_assert_exact_keys()` rejects unknown
keys, not just missing ones -- an additive field lands here as a required
test update, not a silent pass. That's the right tradeoff while `sevenbox_demo`
has exactly one consumer (`app.js`, shipped in the same repo, same commit)
and isn't published anywhere external -- every past commit in this pass
has, in fact, needed to touch this file when the payload grew, and that's
been useful friction, not accidental breakage. If `sevenbox_demo` ever grows an
external consumer that doesn't control its own upgrade timing (a published
REST API, a third-party integration), the right move is loosening this to
"required keys present, extra keys tolerated" for forward compatibility --
not before, since nothing today needs that flexibility and guessing at it
now would be speculative.
"""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.parse import quote

import pytest

from sevenbox_demo.server import DemoHandler
from sevenbox_demo.visualize import render_decision_graph, render_health_summary, render_timeline
from warrant_core import (
    DischargeOutcome,
    EffectRecord,
    EpistemicClass,
    ReversibilityClass,
    SourceKind,
    SqliteWarrantStore,
    WarrantStore,
    compensation_for,
    compute_health,
    compute_metrics,
    compute_timeline,
    create_warrant,
    discharge_obligation,
    inspect,
    supersede,
)

WARRANT_KEYS = {
    "id", "short_id", "content", "source_kind", "epistemic", "reversibility",
    "effectful", "origin", "parent_ids", "child_ids", "supersedes_id",
    "superseded_by_ids", "layer", "order", "is_live", "effect_metadata",
    "redaction_metadata", "replay_available", "compensation",
}
EFFECT_METADATA_KEYS = {"present", "request", "response", "source_name"}
REDACTION_METADATA_KEYS = {"applied"}
REPLAY_AVAILABLE_KEYS = {"available", "degraded", "reason"}
COMPENSATION_KEYS = {"direct", "transitive", "discharges"}
DIRECT_COMPENSATION_KEYS = {"action_required", "ambiguous", "reason", "competing_superseder_ids"}
TRANSITIVE_COMPENSATION_KEYS = {"superseded_ancestor_id", "corrector_id"}
DISCHARGE_KEYS = {"discharge_id", "outcome", "reason", "evidence_id"}
FRONTIER_KEYS = {"live_ids", "conflicts"}
CONFLICT_KEYS = {"target_id", "competing_superseder_ids"}
GRAPH_LEVEL_OBLIGATION_KEYS = {
    "irreversible_id", "irreversible_content", "superseded_ancestor_id", "corrector_id",
}
METRICS_KEYS = {
    "total_decisions", "frontier_size", "superseded_count", "conflict_count",
    "pending_compensation_count", "effectful_count", "replayable_count",
    "degraded_replay_count", "replayable_ratio", "redacted_count",
    "average_causal_depth", "source_kind_counts", "reversibility_counts",
}
HEALTH_KEYS = {"overall", "signals"}
HEALTH_SIGNAL_KEYS = {"name", "status", "summary", "detail"}
TIMELINE_KEYS = {"entries"}
TIMELINE_ENTRY_KEYS = {
    "order", "warrant_id", "event_kind", "source_kind", "reversibility",
    "content", "is_live", "supersedes_id",
}
TOP_LEVEL_GRAPH_PAYLOAD_KEYS = {
    "mode", "call_counts", "warrant_count", "warrants", "frontier",
    "obligations", "metrics", "health", "timeline",
}
VALID_HEALTH_STATUSES = {"healthy", "degraded", "critical"}
VALID_SOURCE_KINDS = {
    "oracle", "tool", "human", "sensor", "clock", "random", "derived", "axiom", "federated",
}
VALID_REVERSIBILITY = {"pure", "compensable", "irreversible"}
VALID_EVENT_KINDS = {"axiom", "correction", "irreversible_action", "evidence"}


def _assert_exact_keys(d, expected, label):
    assert isinstance(d, dict), f"{label} must be a dict, got {type(d).__name__}"
    assert set(d.keys()) == expected, (
        f"{label} keys mismatch.\n  missing: {expected - set(d.keys())}\n  extra: {set(d.keys()) - expected}"
    )


def _assert_warrant_shape(w, label):
    _assert_exact_keys(w, WARRANT_KEYS, label)
    assert isinstance(w["id"], str) and w["id"]
    assert w["short_id"] == w["id"][:8]
    assert w["source_kind"] in VALID_SOURCE_KINDS
    assert w["reversibility"] in VALID_REVERSIBILITY
    assert isinstance(w["effectful"], bool)
    assert w["origin"] is None or isinstance(w["origin"], dict)
    assert isinstance(w["parent_ids"], list) and all(isinstance(p, str) for p in w["parent_ids"])
    assert isinstance(w["child_ids"], list) and all(isinstance(c, str) for c in w["child_ids"])
    assert w["supersedes_id"] is None or isinstance(w["supersedes_id"], str)
    assert isinstance(w["superseded_by_ids"], list)
    assert isinstance(w["layer"], int) and w["layer"] >= 0
    assert isinstance(w["order"], int) and w["order"] >= 0
    assert isinstance(w["is_live"], bool)

    _assert_exact_keys(w["effect_metadata"], EFFECT_METADATA_KEYS, f"{label}.effect_metadata")
    assert isinstance(w["effect_metadata"]["present"], bool)
    if not w["effect_metadata"]["present"]:
        assert w["effect_metadata"]["request"] is None
        assert w["effect_metadata"]["response"] is None
        assert w["effect_metadata"]["source_name"] is None

    _assert_exact_keys(w["redaction_metadata"], REDACTION_METADATA_KEYS, f"{label}.redaction_metadata")
    assert isinstance(w["redaction_metadata"]["applied"], bool)

    _assert_exact_keys(w["replay_available"], REPLAY_AVAILABLE_KEYS, f"{label}.replay_available")
    assert isinstance(w["replay_available"]["available"], bool)
    assert isinstance(w["replay_available"]["degraded"], bool)
    assert isinstance(w["replay_available"]["reason"], str) and w["replay_available"]["reason"]

    _assert_exact_keys(w["compensation"], COMPENSATION_KEYS, f"{label}.compensation")
    direct = w["compensation"]["direct"]
    if direct is not None:
        _assert_exact_keys(direct, DIRECT_COMPENSATION_KEYS, f"{label}.compensation.direct")
    for t in w["compensation"]["transitive"]:
        _assert_exact_keys(t, TRANSITIVE_COMPENSATION_KEYS, f"{label}.compensation.transitive[]")
    for d in w["compensation"]["discharges"]:
        _assert_exact_keys(d, DISCHARGE_KEYS, f"{label}.compensation.discharges[]")
        assert d["outcome"] in {"compensated", "not_needed"}
        assert isinstance(d["reason"], str) and d["reason"]


def _assert_graph_payload_shape(payload):
    _assert_exact_keys(payload, TOP_LEVEL_GRAPH_PAYLOAD_KEYS, "payload")
    assert isinstance(payload["mode"], str)
    assert isinstance(payload["call_counts"], dict)
    assert isinstance(payload["warrant_count"], int)
    assert payload["warrant_count"] == len(payload["warrants"])

    for i, w in enumerate(payload["warrants"]):
        _assert_warrant_shape(w, f"warrants[{i}]")

    _assert_exact_keys(payload["frontier"], FRONTIER_KEYS, "frontier")
    assert isinstance(payload["frontier"]["live_ids"], list)
    for c in payload["frontier"]["conflicts"]:
        _assert_exact_keys(c, CONFLICT_KEYS, "frontier.conflicts[]")

    for o in payload["obligations"]:
        _assert_exact_keys(o, GRAPH_LEVEL_OBLIGATION_KEYS, "obligations[]")

    _assert_exact_keys(payload["metrics"], METRICS_KEYS, "metrics")
    m = payload["metrics"]
    for key in (
        "total_decisions", "frontier_size", "superseded_count", "conflict_count",
        "pending_compensation_count", "effectful_count", "replayable_count",
        "degraded_replay_count", "redacted_count",
    ):
        assert isinstance(m[key], int), f"metrics.{key} must be an int"
    assert m["replayable_ratio"] is None or isinstance(m["replayable_ratio"], float)
    assert isinstance(m["average_causal_depth"], (int, float))
    assert isinstance(m["source_kind_counts"], dict)
    assert isinstance(m["reversibility_counts"], dict)

    _assert_exact_keys(payload["health"], HEALTH_KEYS, "health")
    assert payload["health"]["overall"] in VALID_HEALTH_STATUSES
    assert len(payload["health"]["signals"]) == 3  # compensation_backlog, replay_degraded, frontier_conflicts
    for s in payload["health"]["signals"]:
        _assert_exact_keys(s, HEALTH_SIGNAL_KEYS, "health.signals[]")
        assert s["status"] in VALID_HEALTH_STATUSES

    _assert_exact_keys(payload["timeline"], TIMELINE_KEYS, "timeline")
    entries = payload["timeline"]["entries"]
    assert len(entries) == payload["warrant_count"]  # one timeline entry per Warrant, no filtering
    for i, e in enumerate(entries):
        _assert_exact_keys(e, TIMELINE_ENTRY_KEYS, f"timeline.entries[{i}]")
        assert e["event_kind"] in VALID_EVENT_KINDS
        assert e["order"] == i  # insertion order, not a timestamp -- see decision_timeline.py
        assert isinstance(e["is_live"], bool)
    # The warrant-level view and the timeline agree on which ids exist --
    # two independent projections of the same DecisionGraph, never allowed
    # to silently diverge.
    assert {e["warrant_id"] for e in entries} == {w["id"] for w in payload["warrants"]}


# --- HTTP-level: proves the server's actual wiring produces the contract ---

@pytest.fixture()
def demo_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get_json(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def test_api_run_payload_matches_the_documented_contract(demo_server):
    payload = _get_json(demo_server + "/api/run?scenario=refund")
    _assert_graph_payload_shape(payload)
    # Sanity anchors, so this test fails loudly if the fixture scenario
    # itself changes shape rather than silently exercising fewer branches:
    assert payload["warrant_count"] == 6
    assert any(w["effect_metadata"]["present"] for w in payload["warrants"])
    assert any(not w["effect_metadata"]["present"] for w in payload["warrants"])
    assert any(w["compensation"]["transitive"] for w in payload["warrants"])
    assert any(w["compensation"]["direct"] is not None for w in payload["warrants"])
    # This fixture has no Origin set anywhere -- the non-None origin dict
    # shape ({org_id, team_id, role_id, agent_id, run_id}) is covered
    # separately in tests/test_visualize.py's direct render_decision_graph()
    # tests, not reachable through either demo scenario today.
    assert all(w["origin"] is None for w in payload["warrants"])


def test_api_load_payload_matches_the_documented_contract(demo_server, tmp_path):
    db_path = str(tmp_path / "contract.db")
    store = SqliteWarrantStore(db_path)
    store.add(create_warrant(
        content={"v": "root"}, source_kind=SourceKind.AXIOM,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
    ))
    store.close()

    payload = _get_json(demo_server + f"/api/load?path={quote(db_path)}")
    _assert_graph_payload_shape(payload)
    assert payload["mode"] == "loaded"


# --- Renderer-level: covers branches the two fixed demo scenarios never
# reach (a genuine frontier conflict, an ambiguous direct compensation) ---

def test_conflict_and_ambiguous_direct_compensation_shapes():
    store = WarrantStore()
    root = store.add(create_warrant(
        content={"v": "root"}, source_kind=SourceKind.AXIOM,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.IRREVERSIBLE,
    ))
    # Two competing live correctors on the same target -- a genuine
    # frontier.Conflict, and an ambiguous direct CompensationObligation on
    # `root` (IRREVERSIBLE, 2 live superseders), neither reachable through
    # the refund/kyc demo scenarios.
    supersede(
        store, target_id=root.id, content={"v": "correction A"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="a", response="a"),
    )
    supersede(
        store, target_id=root.id, content={"v": "correction B"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="b", response="b"),
    )

    graph = inspect(store)
    metrics = compute_metrics(graph)
    health = compute_health(metrics)
    payload = render_decision_graph(graph, mode="live", call_counts={})
    payload.update(render_health_summary(metrics, health))
    payload["timeline"] = render_timeline(compute_timeline(graph))

    _assert_graph_payload_shape(payload)
    assert len(payload["frontier"]["conflicts"]) == 1
    root_json = next(w for w in payload["warrants"] if w["id"] == root.id)
    assert root_json["compensation"]["direct"] is not None
    assert root_json["compensation"]["direct"]["ambiguous"] is True


def test_discharged_obligation_shape():
    store = WarrantStore()
    root = store.add(create_warrant(
        content={"v": "root"}, source_kind=SourceKind.AXIOM,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
    ))
    action = store.add(create_warrant(
        content={"claim": "refund_action(4471)"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.IRREVERSIBLE,
        parent_ids=(root.id,),
    ))
    supersede(
        store, target_id=action.id, content={"claim": "action was erroneous"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="q", response="a"),
    )
    obligation = compensation_for(store, action.id)
    discharge_obligation(
        store, obligation, DischargeOutcome.NOT_NEEDED,
        reason="target was already gone",
    )

    graph = inspect(store)
    metrics = compute_metrics(graph)
    health = compute_health(metrics)
    payload = render_decision_graph(graph, mode="live", call_counts={})
    payload.update(render_health_summary(metrics, health))
    payload["timeline"] = render_timeline(compute_timeline(graph))

    _assert_graph_payload_shape(payload)
    action_json = next(w for w in payload["warrants"] if w["id"] == action.id)
    assert action_json["compensation"]["direct"] is None  # resolved, nothing pending
    discharges = action_json["compensation"]["discharges"]
    assert len(discharges) == 1
    assert discharges[0]["outcome"] == "not_needed"
    assert discharges[0]["reason"] == "target was already gone"
    assert discharges[0]["evidence_id"] is None
