"""
sevenbox_demo.server -- a small local HTTP server for the Sevenbox demo, built entirely
on Python's standard library (http.server), matching warrant_core's own
zero-dependency ethos. No Flask, no FastAPI, no external web framework.

Routes:
  GET  /                          -> static/index.html
  GET  /static/<file>             -> static assets (css/js)
  GET  /api/run?scenario=X        -> runs scenario X live, returns a JSON snapshot
  GET  /api/replay?scenario=X     -> replays scenario X's last live run
  GET  /api/explain?scenario=X&warrant_id=Y -> "why did this happen" for one Warrant
  GET  /api/load?path=P           -> inspects an arbitrary SqliteWarrantStore file,
                                      no scenario involved at all
  GET  /api/load/explain?path=P&warrant_id=Y -> "why did this happen" for a
                                      Warrant in that same file

`scenario` defaults to "refund". Two are registered (refund, kyc) --
deliberately more than one, so that sevenbox_demo.visualize's genericity is
demonstrated by the server actually using it for two different domains, not
just claimed. `/api/load` is the complementary proof: a store this process
never ran anything to produce -- any Sevenbox-shaped SQLite file -- goes through
the exact same DecisionGraph -> renderer path.

Every route above that renders a graph (`/api/run`, `/api/replay`,
`/api/load`) also carries `metrics`/`health` in the same JSON response, via
`_send_graph_payload()` -- the Operations Center home view and the
Decision Explorer come from one fetch, not two. No separate `/api/metrics`
endpoint exists yet: nothing today needs metrics without the graph that
produced them, and a metrics-only endpoint would be speculative ahead of a
real consumer (e.g. a lightweight polling widget) that doesn't exist yet.

This module is the only place in sevenbox_demo that constructs a
warrant_core.DecisionGraphSource -- it is the orchestration layer: translate
an HTTP request into a *source* (which store, backed by what), then hand the
DecisionGraph/Explanation that source produces to sevenbox_demo.visualize for
pure JSON formatting. Nothing here (or in sevenbox_demo.visualize) opens a
SqliteWarrantStore and reads from it directly for rendering purposes --
`SqliteDecisionGraphSource` owns that. The one exception is scenario replay's
`reference_store` (`_last_live_store` below): that's a mechanism internal to
SevenboxSession's replay contract, not a decision-inspection read, so it isn't
routed through a DecisionGraphSource. A future CLI or REST API follows this
same shape: construct a source, call load()/explain(), hand the result to a
renderer.

JSON RESPONSE CONTRACT for /api/run, /api/replay, /api/load (identical
shape, built by `_send_graph_payload()`): the only contract `app.js` has
with this server, and (since sevenbox_demo has no other consumer yet) the
closest thing to a stable API this demo has. Not published or versioned
anywhere else -- enforced instead by
tests/test_api_contract.py::_assert_graph_payload_shape, which asserts the
*exact* key set of every nested shape below, not just presence. Treat that
test, not this docstring, as authoritative if they ever disagree; update
both together.

  {
    "mode": str, "call_counts": {str: int}, "warrant_count": int,
    "warrants": [{
      "id", "short_id": str, "content": any,
      "source_kind", "epistemic", "reversibility": str,
      "effectful", "is_live": bool,
      "origin": {"org_id", "team_id", "role_id", "agent_id", "run_id": str|null} | null,
      "parent_ids", "child_ids", "superseded_by_ids": [str],
      "supersedes_id": str | null, "layer", "order": int,
      "effect_metadata": {"present": bool, "request", "response": any, "source_name": str|null},
      "redaction_metadata": {"applied": bool},
      "replay_available": {"available", "degraded": bool, "reason": str},
      "compensation": {
        "direct": {"action_required", "ambiguous": bool, "reason": str, "competing_superseder_ids": [str]} | null,
        "transitive": [{"superseded_ancestor_id", "corrector_id": str}],
      },
    }],
    "frontier": {"live_ids": [str], "conflicts": [{"target_id": str, "competing_superseder_ids": [str]}]},
    "obligations": [{"irreversible_id": str, "irreversible_content": any, "superseded_ancestor_id", "corrector_id": str}],
    "metrics": {  # warrant_core.decision_metrics.DecisionMetrics, unchanged
      "total_decisions", "frontier_size", "superseded_count", "conflict_count",
      "pending_compensation_count", "effectful_count", "replayable_count",
      "degraded_replay_count", "redacted_count": int,
      "replayable_ratio": float | null, "average_causal_depth": float,
      "source_kind_counts", "reversibility_counts": {str: int},
    },
    "health": {  # warrant_core.decision_health.DecisionHealth, unchanged
      "overall": "healthy"|"degraded"|"critical",
      "signals": [{"name", "status": str, "summary", "detail": str}],  # always 3
    },
    "timeline": {  # warrant_core.decision_timeline.DecisionTimeline, unchanged
      "entries": [{
        "order": int,  # ordinal position, NOT a timestamp -- see decision_timeline.py
        "warrant_id": str, "content": any,
        "event_kind": "axiom"|"correction"|"irreversible_action"|"evidence",
        "source_kind", "reversibility": str, "is_live": bool,
        "supersedes_id": str | null,
      }],
    },
  }
"""
from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlsplit

from warrant_core import (
    InMemoryDecisionGraphSource,
    SqliteDecisionGraphSource,
    WarrantStore,
    compute_health,
    compute_metrics,
    compute_timeline,
)

from . import (
    customer_support_scenario,
    deployment_scenario,
    document_review_scenario,
    kyc_scenario,
    procurement_scenario,
    scenario,
    security_incident_scenario,
)
from .visualize import (
    render_decision_graph,
    render_explanation,
    render_health_summary,
    render_timeline,
)

STATIC_DIR = Path(__file__).parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".woff2": "font/woff2",
}

_SCENARIOS = {
    "refund": scenario,
    "kyc": kyc_scenario,
    "security-incident": security_incident_scenario,
    "procurement": procurement_scenario,
    "deployment": deployment_scenario,
    "customer-support": customer_support_scenario,
    "document-review": document_review_scenario,
}

# Demo-scoped state: the most recent live run's store, per scenario, so
# /api/replay has something to replay from. A real product would scope this
# per-session; a local single-user demo doesn't need to.
_last_live_store: Dict[str, Optional[WarrantStore]] = {name: None for name in _SCENARIOS}


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "SevenboxDemo/0.1"

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404, f"Not found: {path.name}")
            return
        data = path.read_bytes()
        content_type = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_graph_payload(self, graph, *, mode: str, call_counts: Dict[str, int]) -> None:
        """Shared by every route that renders a DecisionGraph: the JSON
        response always carries the node list AND the Operations Center
        summary (metrics + health + timeline) for that same graph, computed
        here -- the one place in sevenbox_demo that calls compute_metrics()/
        compute_health()/compute_timeline() -- and merged before sending.
        One response, no second round-trip for the home view; see this
        module's docstring for why a graph-only endpoint already produces
        enough to build the home screen from, and why a metrics-only
        endpoint isn't justified yet (nothing today needs metrics without
        the graph).
        """
        payload = render_decision_graph(graph, mode=mode, call_counts=call_counts)
        metrics = compute_metrics(graph)
        health = compute_health(metrics)
        payload.update(render_health_summary(metrics, health))
        payload["timeline"] = render_timeline(compute_timeline(graph))
        self._send_json(payload)

    def _scenario_module(self, query: Dict[str, list]):
        name = query.get("scenario", ["refund"])[0]
        module = _SCENARIOS.get(name)
        if module is None:
            return None, name
        return module, name

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming convention)
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
            return

        if path.startswith("/static/"):
            # Real containment check, not a blanket "no slashes" rule --
            # static/fonts/*.woff2 is a legitimate one-level subdirectory
            # (added for the self-hosted Inter/IBM Plex Mono files), so
            # traversal has to be ruled out by resolving the path and
            # checking it's still under STATIC_DIR, not by rejecting
            # every "/" in the requested filename.
            filename = path[len("/static/"):]
            candidate = (STATIC_DIR / filename).resolve()
            try:
                candidate.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self.send_error(400, "Invalid static path")
                return
            self._send_file(candidate)
            return

        if path == "/api/scenarios":
            self._send_json({"scenarios": list(_SCENARIOS)})
            return

        if path == "/api/run":
            module, name = self._scenario_module(query)
            if module is None:
                self._send_json({"error": f"Unknown scenario {name!r}"}, status=400)
                return
            session, calls = module.run_live()
            _last_live_store[name] = session.store
            graph = InMemoryDecisionGraphSource(store=session.store).load()
            self._send_graph_payload(graph, mode="live", call_counts=calls)
            return

        if path == "/api/replay":
            module, name = self._scenario_module(query)
            if module is None:
                self._send_json({"error": f"Unknown scenario {name!r}"}, status=400)
                return
            reference = _last_live_store.get(name)
            if reference is None:
                self._send_json(
                    {"error": "No live run to replay yet. Click Run first."}, status=400
                )
                return
            session, calls = module.run_replay(reference)
            graph = InMemoryDecisionGraphSource(store=session.store).load()
            self._send_graph_payload(graph, mode="replay", call_counts=calls)
            return

        if path == "/api/explain":
            module, name = self._scenario_module(query)
            if module is None:
                self._send_json({"error": f"Unknown scenario {name!r}"}, status=400)
                return
            warrant_id = query.get("warrant_id", [None])[0]
            store = _last_live_store.get(name)
            if store is None:
                self._send_json({"error": "No live run yet. Click Run first."}, status=400)
                return
            if warrant_id is None or warrant_id not in store:
                self._send_json({"error": f"Unknown warrant_id {warrant_id!r}"}, status=400)
                return
            explanation = InMemoryDecisionGraphSource(store=store).explain(warrant_id)
            self._send_json(render_explanation(explanation))
            return

        if path == "/api/load":
            path_param = query.get("path", [None])[0]
            error = self._validate_sqlite_path(path_param)
            if error is not None:
                self._send_json({"error": error}, status=400)
                return
            graph = SqliteDecisionGraphSource(path=path_param).load()
            self._send_graph_payload(graph, mode="loaded", call_counts={})
            return

        if path == "/api/load/explain":
            path_param = query.get("path", [None])[0]
            error = self._validate_sqlite_path(path_param)
            if error is not None:
                self._send_json({"error": error}, status=400)
                return
            warrant_id = query.get("warrant_id", [None])[0]
            if warrant_id is None:
                self._send_json({"error": "Missing warrant_id"}, status=400)
                return
            try:
                explanation = SqliteDecisionGraphSource(path=path_param).explain(warrant_id)
            except KeyError:
                self._send_json({"error": f"Unknown warrant_id {warrant_id!r}"}, status=400)
                return
            self._send_json(render_explanation(explanation))
            return

        self.send_error(404)

    @staticmethod
    def _validate_sqlite_path(path_param: Optional[str]) -> Optional[str]:
        """Returns an error message if `path_param` can't be opened as an
        existing file, None if it's fine to hand to SqliteDecisionGraphSource.
        Checked here, not left to sqlite3 itself: sqlite3.connect() silently
        *creates* an empty database at a nonexistent path rather than
        raising, which would turn a typo'd path into "here's an empty graph"
        instead of a clear error.
        """
        if not path_param:
            return "Missing path"
        if not Path(path_param).is_file():
            return f"No such file: {path_param!r}"
        return None

    def log_message(self, fmt: str, *args) -> None:
        pass  # keep the demo's terminal output quiet


def run_demo_server(port: int = 8420, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), DemoHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"Sevenbox Operations Center running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass  # headless environment -- the URL above is still printed
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
