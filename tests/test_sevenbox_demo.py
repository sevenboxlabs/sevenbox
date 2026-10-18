import json
import os
import tempfile
import threading
import urllib.request
from urllib.parse import quote

import pytest

from sevenbox_demo.server import DemoHandler
from http.server import ThreadingHTTPServer
from warrant_core import (
    EpistemicClass,
    ReversibilityClass,
    SourceKind,
    SqliteWarrantStore,
    create_warrant,
)


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


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.read()


def test_index_and_static_assets_are_served(demo_server):
    status, body = _get(demo_server + "/")
    assert status == 200
    assert b"Sevenbox" in body

    status, body = _get(demo_server + "/static/app.js")
    assert status == 200
    assert b"openExampleSystem" in body

    status, body = _get(demo_server + "/static/style.css")
    assert status == 200
    assert b"--bg" in body


def test_replay_before_run_returns_400():
    # Fresh server-module state: this test intentionally hits /api/replay
    # first on a brand new server instance to check the guard.
    server = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import sevenbox_demo.server as server_module
        server_module._last_live_store["refund"] = None  # clean state for this test
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/replay")
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected an HTTPError for 400 status"
        except urllib.error.HTTPError as e:
            assert e.code == 400
            payload = json.loads(e.read())
            assert "error" in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_run_then_replay_end_to_end(demo_server):
    status, body = _get(demo_server + "/api/run")
    assert status == 200
    live = json.loads(body)
    assert live["mode"] == "live"
    assert live["warrant_count"] == 6
    assert live["call_counts"] == {"infer_intent": 1, "lookup_order": 1}
    assert len(live["frontier"]["live_ids"]) == 5
    assert len(live["obligations"]) == 1

    # The Operations Center home view rides on the same response --
    # no second round-trip.
    assert live["metrics"]["total_decisions"] == 6
    assert live["metrics"]["pending_compensation_count"] == 1
    assert live["health"]["overall"] == "critical"  # the refund scenario's
    # own uncompensated obligation, same fact live["obligations"] reports
    assert len(live["timeline"]["entries"]) == 6
    event_kinds = [e["event_kind"] for e in live["timeline"]["entries"]]
    assert event_kinds[0] == "axiom"  # the ticket
    assert "correction" in event_kinds  # the refund-history supersede()

    status, body = _get(demo_server + "/api/replay")
    assert status == 200
    replay = json.loads(body)
    assert replay["mode"] == "replay"
    assert replay["call_counts"] == {"infer_intent": 0, "lookup_order": 0}, (
        "the real handlers must not be invoked again during replay"
    )
    assert set(replay["frontier"]["live_ids"]) == set(live["frontier"]["live_ids"])
    assert replay["metrics"] == live["metrics"]
    assert replay["health"] == live["health"]
    assert replay["timeline"] == live["timeline"]


def test_unknown_path_is_404(demo_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(demo_server + "/nonexistent")
    assert exc_info.value.code == 404


def test_scenarios_endpoint_lists_every_registered_scenario(demo_server):
    status, body = _get(demo_server + "/api/scenarios")
    assert status == 200
    payload = json.loads(body)
    assert set(payload["scenarios"]) == {
        "refund", "kyc", "security-incident", "procurement", "deployment",
        "customer-support", "document-review",
    }


def test_kyc_scenario_runs_through_the_same_generic_visualization(demo_server):
    """Proves the decoupling, not just claims it: a second, unrelated domain
    goes through the exact same /api/run + serialize_store path."""
    status, body = _get(demo_server + "/api/run?scenario=kyc")
    assert status == 200
    live = json.loads(body)
    assert live["mode"] == "live"
    assert live["call_counts"] == {"extract_identity": 1, "check_watchlist": 1}
    assert len(live["obligations"]) == 1

    status, body = _get(demo_server + "/api/replay?scenario=kyc")
    assert status == 200
    replay = json.loads(body)
    assert replay["call_counts"] == {"extract_identity": 0, "check_watchlist": 0}


def test_security_incident_scenario_produces_a_real_frontier_conflict(demo_server):
    """The first scenario to exercise a genuine, unresolved frontier
    conflict -- two independent, live correctors of the same target,
    neither superseding the other -- through the real server, not just
    warrant_core's own unit tests."""
    status, body = _get(demo_server + "/api/run?scenario=security-incident")
    assert status == 200
    live = json.loads(body)
    assert live["mode"] == "live"
    assert live["call_counts"] == {"triage_alert": 1, "query_telemetry": 1}
    assert len(live["frontier"]["conflicts"]) == 1
    assert len(live["frontier"]["conflicts"][0]["competing_superseder_ids"]) == 2
    assert live["metrics"]["conflict_count"] == 1
    assert live["health"]["overall"] == "critical"

    status, body = _get(demo_server + "/api/replay?scenario=security-incident")
    assert status == 200
    replay = json.loads(body)
    assert replay["call_counts"] == {"triage_alert": 0, "query_telemetry": 0}
    assert len(replay["frontier"]["conflicts"]) == 1


def test_procurement_scenario_produces_real_redactions_and_a_branch(demo_server):
    """The first scenario to actually apply RedactionPolicy end to end
    through the real server (kyc's own demo scenario never does), and the
    first with a genuine fork/merge shape in the causal graph instead of a
    single chain."""
    status, body = _get(demo_server + "/api/run?scenario=procurement")
    assert status == 200
    live = json.loads(body)
    assert live["mode"] == "live"
    assert live["call_counts"] == {"extract_vendor_info": 1, "check_vendor_risk": 1}
    assert live["metrics"]["redacted_count"] == 2  # extract_vendor_info + check_vendor_risk
    assert len(live["obligations"]) == 1

    branch_warrants = [
        w for w in live["warrants"] if w["content"].get("path") in ("auto_approve", "manual_escalation")
    ]
    assert len(branch_warrants) == 2
    assert branch_warrants[0]["parent_ids"] == branch_warrants[1]["parent_ids"]

    status, body = _get(demo_server + "/api/replay?scenario=procurement")
    assert status == 200
    replay = json.loads(body)
    assert replay["call_counts"] == {"extract_vendor_info": 0, "check_vendor_risk": 0}


def test_deployment_scenario_exercises_human_and_compensable(demo_server):
    """The first scenario to use @human and @compensable through the real
    server -- every prior scenario only ever produced oracle/tool/derive
    warrants and IRREVERSIBLE (never COMPENSABLE) reversibility."""
    status, body = _get(demo_server + "/api/run?scenario=deployment")
    assert status == 200
    live = json.loads(body)
    assert live["mode"] == "live"
    assert live["call_counts"] == {"infer_deployment_risk": 1, "run_ci_checks": 1}
    assert "human" in {w["source_kind"] for w in live["warrants"]}
    assert "compensable" in {w["reversibility"] for w in live["warrants"]}
    assert len(live["obligations"]) == 1

    status, body = _get(demo_server + "/api/replay?scenario=deployment")
    assert status == 200
    replay = json.loads(body)
    assert replay["call_counts"] == {"infer_deployment_risk": 0, "run_ci_checks": 0}


def test_customer_support_scenario_runs_live_and_replay(demo_server):
    status, body = _get(demo_server + "/api/run?scenario=customer-support")
    assert status == 200
    live = json.loads(body)
    assert live["mode"] == "live"
    assert live["call_counts"] == {"classify_ticket": 1, "lookup_account": 1}
    assert len(live["obligations"]) == 1

    status, body = _get(demo_server + "/api/replay?scenario=customer-support")
    assert status == 200
    replay = json.loads(body)
    assert replay["call_counts"] == {"classify_ticket": 0, "lookup_account": 0}


def test_document_review_scenario_runs_live_and_replay(demo_server):
    status, body = _get(demo_server + "/api/run?scenario=document-review")
    assert status == 200
    live = json.loads(body)
    assert live["mode"] == "live"
    assert live["call_counts"] == {"extract_clauses": 1, "check_compliance": 1}
    assert len(live["obligations"]) == 1

    status, body = _get(demo_server + "/api/replay?scenario=document-review")
    assert status == 200
    replay = json.loads(body)
    assert replay["call_counts"] == {"extract_clauses": 0, "check_compliance": 0}


def test_unknown_scenario_name_returns_400(demo_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(demo_server + "/api/run?scenario=does-not-exist")
    assert exc_info.value.code == 400


def test_explain_endpoint_returns_a_narrative_for_a_real_warrant(demo_server):
    status, body = _get(demo_server + "/api/run")
    live = json.loads(body)
    target_id = live["warrants"][-1]["id"]

    status, body = _get(demo_server + f"/api/explain?warrant_id={target_id}")
    assert status == 200
    explanation = json.loads(body)
    assert explanation["target_id"] == target_id
    assert len(explanation["nodes"]) == 5
    assert len(explanation["narrative"]) == 5
    assert explanation["nodes"][0]["warrant_id"] == target_id


def test_explain_endpoint_rejects_unknown_warrant_id(demo_server):
    _get(demo_server + "/api/run")  # ensure a live store exists
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(demo_server + "/api/explain?warrant_id=not-a-real-id")
    assert exc_info.value.code == 400


def test_explain_endpoint_before_any_run_returns_400():
    server = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import sevenbox_demo.server as server_module
        server_module._last_live_store["refund"] = None
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(f"http://127.0.0.1:{port}/api/explain?warrant_id=x")
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def _build_unrelated_sqlite_store(db_path):
    """A store with no relationship to either registered scenario -- content
    from a third, made-up domain, built directly against warrant_core, never
    touching sevenbox_sdk or a SevenboxSession. Proves /api/load works against data
    this process never ran anything to produce."""
    store = SqliteWarrantStore(db_path)
    root = store.add(create_warrant(
        content={"shipment_id": "sh-001"}, source_kind=SourceKind.AXIOM,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
    ))
    leaf = store.add(create_warrant(
        content={"eta": "2026-08-10"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        parent_ids=(root.id,),
    ))
    store.close()
    return root, leaf


def test_load_endpoint_inspects_an_arbitrary_sqlite_file_with_no_scenario_involved(demo_server):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "shipping.db")
        root, leaf = _build_unrelated_sqlite_store(db_path)

        status, body = _get(demo_server + f"/api/load?path={quote(db_path)}")
        assert status == 200
        payload = json.loads(body)
        assert payload["mode"] == "loaded"
        assert payload["warrant_count"] == 2
        assert {w["id"] for w in payload["warrants"]} == {root.id, leaf.id}

        # No compensation obligations, no conflicts, no effectful warrants
        # at all in this fixture -- a clean bill of health, computed
        # server-side, not inferred by the test from the raw counts.
        assert payload["metrics"]["total_decisions"] == 2
        assert payload["metrics"]["replayable_ratio"] is None
        assert payload["health"]["overall"] == "healthy"


def test_load_endpoint_rejects_a_missing_path_without_creating_one(demo_server):
    with tempfile.TemporaryDirectory() as tmp:
        missing_path = os.path.join(tmp, "does-not-exist.db")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(demo_server + f"/api/load?path={quote(missing_path)}")
        assert exc_info.value.code == 400
        # The whole point of validating the path first: sqlite3.connect()
        # would otherwise silently create an empty database here.
        assert not os.path.exists(missing_path)


def test_load_endpoint_requires_a_path_param(demo_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(demo_server + "/api/load")
    assert exc_info.value.code == 400


def test_load_explain_endpoint_works_against_the_loaded_file(demo_server):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "shipping.db")
        root, leaf = _build_unrelated_sqlite_store(db_path)

        status, body = _get(
            demo_server + f"/api/load/explain?path={quote(db_path)}&warrant_id={leaf.id}"
        )
        assert status == 200
        explanation = json.loads(body)
        assert explanation["target_id"] == leaf.id
        assert len(explanation["nodes"]) == 2


def test_load_explain_endpoint_rejects_unknown_warrant_id(demo_server):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "shipping.db")
        _build_unrelated_sqlite_store(db_path)

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(demo_server + f"/api/load/explain?path={quote(db_path)}&warrant_id=not-a-real-id")
        assert exc_info.value.code == 400
