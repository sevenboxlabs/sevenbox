const KIND_COLORS = {
  axiom: "var(--c-axiom)",
  oracle: "var(--c-oracle)",
  tool: "var(--c-tool)",
  derived: "var(--c-derived)",
  human: "var(--c-human)",
};

// Static, non-decision-making display maps: warrant_core already decided
// `status` (healthy/degraded/critical) -- these only pick an icon/label for
// a category the server computed. Same role KIND_COLORS already plays for
// source_kind. Never used to derive a status, only to display one.
const STATUS_ICONS = { healthy: "\u{1F7E2}", degraded: "⚠️", critical: "\u{1F534}" };
const STATUS_LABELS = { healthy: "Healthy", degraded: "Degraded", critical: "Critical" };

// Same static-lookup role as STATUS_ICONS above, for the event_kind
// warrant_core.decision_timeline already classified each timeline entry
// as. Never used to decide what kind an entry is -- only to pick a badge
// for the kind the server already sent.
const EVENT_KIND_LABELS = {
  axiom: "Axiom",
  correction: "Correction",
  irreversible_action: "Irreversible",
  evidence: "Evidence",
};

const openSystemBtn = document.getElementById("open-system-btn");
const systemPickerPanel = document.getElementById("system-picker-panel");
const systemPickerExamplesEl = document.getElementById("system-picker-examples");
const replayBtn = document.getElementById("replay-btn");
const loadPathInput = document.getElementById("load-path-input");
const loadBtn = document.getElementById("load-btn");
const statusEl = document.getElementById("status");
const explainOutput = document.getElementById("explain-output");
const nodeMetadataEl = document.getElementById("node-metadata");
const panelNavEl = document.getElementById("panel-nav");
const contextBarEl = document.getElementById("context-bar");
const ctxSystemEl = document.getElementById("ctx-system");
const ctxExecutionEl = document.getElementById("ctx-execution");
const ctxHealthEl = document.getElementById("ctx-health");
const ctxAlertsEl = document.getElementById("ctx-alerts");
const ctxSelectionEl = document.getElementById("ctx-selection");
const themeToggleBtn = document.getElementById("theme-toggle");

const EXECUTION_MODE_LABELS = { live: "Live run", replay: "Replay", loaded: "Loaded from file" };

// The most recently rendered payload -- the node inspector looks a
// clicked node's full record up from here (render_node_details()'s output,
// already present once per warrant) rather than fetching a second, separate
// "node detail" shape from the server. See sevenbox_demo/visualize.py's
// render_node_details() docstring: the inspector is a view of a node
// already known to the graph, never a parallel model describing it again.
let lastData = null;

// The one shared Selection Context: which decision every detail panel
// currently shows. Set only by selectWarrant() -- never read or written
// directly by a panel's own render function.
let selectedWarrantId = null;

// The Decision Browser's one active filter. Every predicate here reads
// a field render_node_details()
// already sends on every warrant -- no new backend query, no search, no
// combining filters. "All" is the only default that makes sense before an
// operator has picked a question to ask.
let activeBrowserFilter = "all";

// `description` exists so a filter with zero matches never reads as
// ambiguous ("is this broken, or did it just never happen here?") -- it's
// shown both as the button's hover tooltip and, when a filter matches
// nothing, appended to the empty state so the operator learns what "0"
// actually means instead of guessing. See the operator review that found
// "Redacted (0)" reading as untrustworthy with no other System available
// to compare against.
// Grouped like a workflow, not a facet list: "All" stands alone (reset),
// "Needs attention" is the set of questions an operator opens the Browser
// to answer when something might be wrong, "Explore" is everything else
// -- informational, not urgent. Same predicates as before; grouping is
// presentation only (Phase 2, Commit 3).
const BROWSER_FILTERS = [
  { id: "all", label: "All", group: null, description: "Every decision in this Execution, regardless of any of these facts.", predicate: () => true },
  {
    id: "pending-compensation", label: "Pending compensation", group: "attention",
    description: "an irreversible decision whose justification was later corrected, with no compensation recorded for it yet.",
    predicate: (w) => (w.compensation.direct && w.compensation.direct.action_required) || w.compensation.transitive.length > 0,
  },
  {
    id: "replay-degraded", label: "Replay degraded", group: "attention",
    description: "a decision whose replay key can collide with another effectful call sharing the same parents, because no source_name was recorded on its EffectRecord.",
    predicate: (w) => w.replay_available.degraded,
  },
  {
    id: "replay-unavailable", label: "Replay unavailable", group: "attention",
    description: "a decision that cannot be reproduced from replay at all.",
    predicate: (w) => !w.replay_available.available,
  },
  {
    id: "redacted", label: "Redacted", group: "explore",
    description: "a decision whose captured request/response was sanitized before being stored, rather than kept as originally received.",
    predicate: (w) => w.redaction_metadata.applied,
  },
  {
    id: "produced-effects", label: "Produced effects", group: "explore",
    description: "a decision that invoked something outside this process -- it carries a captured EffectRecord.",
    predicate: (w) => w.effect_metadata.present,
  },
  {
    id: "superseded", label: "Superseded", group: "explore",
    description: "a decision that something else has since corrected -- no longer part of the current frontier.",
    predicate: (w) => !w.is_live,
  },
  {
    id: "frontier", label: "Frontier (live)", group: "explore",
    description: "a decision still part of what the system currently believes -- nothing has superseded it.",
    predicate: (w) => w.is_live,
  },
];
const BROWSER_GROUP_LABELS = { attention: "Needs attention", explore: "Explore" };

// The System currently being observed -- { kind: "example", name } or
// { kind: "local", path }. This is today's DecisionGraphSource-backed
// data, generalized to the System/Execution vocabulary the rest of this
// file uses. Two things read this: which
// /api/explain route a node click should use (purely a routing fact, never
// a domain judgment -- the server exposes two symmetric explain routes,
// see sevenbox_demo/server.py), and the "currently open" label the System
// Picker shows. Reset on every new example/local-system open.
let currentSystem = { kind: "example", name: "refund" };

// --- Panel registry ---
//
// A panel is one independent view of the current Execution's data. Panels
// don't know about each other, about Systems, or about how the current
// DecisionGraph was obtained -- they receive a container + the current
// payload and render into it, nothing more.
//
// `mount(container, data)` for every panel below is a moved function, not
// a rewritten one: each body is the exact render*() call this section used
// before the panel shell existed. `container` is accepted for API
// consistency with panels added later (which should render into it
// directly) -- these seven don't use it yet, since their render*()
// functions already know their own element ids and changing that isn't
// required to move them into the registry. That's a real, intentional gap
// closed the next time one of these panels' internals actually change, not
// a placeholder pretending to be finished.
const PANELS = [];
let activePanelId = null;

function registerPanel(panel) {
  PANELS.push(panel);
}

function renderPanelNav() {
  const groups = {};
  PANELS.forEach((p) => {
    (groups[p.group] = groups[p.group] || []).push(p);
  });
  panelNavEl.innerHTML = Object.entries(groups)
    .map(
      ([group, panels]) => `
        <div class="panel-nav-group">
          <div class="panel-nav-group-label">${escapeHtml(group)}</div>
          ${panels
            .map(
              (p) =>
                `<button class="panel-nav-btn" data-panel-id="${p.id}">${panelIcon(p.id)}<span>${escapeHtml(p.title)}</span></button>`
            )
            .join("")}
        </div>
      `
    )
    .join("");
  panelNavEl.querySelectorAll(".panel-nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => showPanel(btn.dataset.panelId));
  });
}

function showPanel(panelId) {
  const panel = PANELS.find((p) => p.id === panelId);
  if (!panel) return;
  activePanelId = panelId;
  PANELS.forEach((p) => {
    document.getElementById(p.sectionId).classList.toggle("hidden", p.id !== panelId);
  });
  panelNavEl.querySelectorAll(".panel-nav-btn").forEach((btn) => {
    btn.classList.toggle("panel-nav-active", btn.dataset.panelId === panelId);
  });
}

// One monoline SVG per panel (Sequence 6 polish pass), stroke="currentColor"
// so each inherits the nav button's own text color -- active/hover/inactive
// all fall out of the existing .panel-nav-btn rules for free, nothing icon-
// specific to keep in sync. Chosen for what the panel actually answers, not
// decoration: a pulse for system health, a filtered list for the Browser, a
// small node graph for the Execution Graph, a crosshair for inspecting one
// decision, a link for a causal chain, a clock for chronological order, two
// stacked layers for "what's currently live," and an undo arrow for
// compensating an irreversible action.
const PANEL_ICON_SVG = {
  "operations-overview": '<path d="M3 12h4l2-7 4 14 2-7h6"/>',
  "decision-browser": '<path d="M8 6h13M8 12h13M8 18h13"/><path d="M3 6h.01M3 12h.01M3 18h.01"/>',
  "execution-graph": '<circle cx="5" cy="6" r="2.2"/><circle cx="19" cy="6" r="2.2"/><circle cx="12" cy="18" r="2.2"/><path d="M7 7.2 10 16M17 7.2 14 16M7.2 6h9.6"/>',
  "node-inspector": '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
  "evidence-chain": '<path d="M15 7h3a5 5 0 0 1 0 10h-3m-6 0H6a5 5 0 0 1 0-10h3"/><line x1="8" y1="12" x2="16" y2="12"/>',
  "timeline": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/>',
  "frontier": '<polygon points="12 3 3 8 12 13 21 8 12 3"/><polyline points="3 14 12 19 21 14"/>',
  "compensation": '<path d="M3 10h10a5 5 0 0 1 5 5v1"/><path d="M7 6 3 10l4 4"/>',
};
function panelIcon(panelId) {
  return `<svg class="panel-nav-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PANEL_ICON_SVG[panelId] || ""}</svg>`;
}

registerPanel({
  id: "operations-overview", title: "Operations Overview", group: "Overview",
  sectionId: "panel-operations-overview",
  mount: (container, data) => { renderHealth(data); renderMetrics(data); },
});
registerPanel({
  id: "decision-browser", title: "Decision Browser", group: "Explorer",
  sectionId: "panel-decision-browser",
  mount: (container, data) => renderDecisionBrowser(data),
});
registerPanel({
  id: "execution-graph", title: "Execution Graph", group: "Explorer",
  sectionId: "panel-execution-graph",
  mount: (container, data) => renderGraph(data),
});
// "Supporting", not "Explorer" (ux_review.md P1-4): Decision Browser is
// the primary investigation surface, Execution Graph the primary visual
// artifact (Phase 2) -- the five below are reactive object views (Node
// Inspector, Evidence Chain) or narrow, specific-question tools
// (Timeline, Frontier, Compensation), not places an operator starts.
// Same nav-group mechanism every group already uses, just a second
// instance of it.
registerPanel({
  id: "node-inspector", title: "Node Inspector", group: "Supporting",
  sectionId: "panel-node-inspector",
  mount: () => {}, // reacts to node selection (selectWarrant), not to render(data)
});
registerPanel({
  id: "evidence-chain", title: "Evidence Chain", group: "Supporting",
  sectionId: "panel-evidence-chain",
  mount: () => {}, // reacts to node selection (selectWarrant), not to render(data)
});
registerPanel({
  id: "timeline", title: "Timeline", group: "Supporting",
  sectionId: "panel-timeline",
  mount: (container, data) => renderTimeline(data),
});
registerPanel({
  id: "frontier", title: "Frontier", group: "Supporting",
  sectionId: "panel-frontier",
  mount: (container, data) => renderFrontier(data),
});
registerPanel({
  id: "compensation", title: "Compensation", group: "Supporting",
  sectionId: "panel-compensation",
  mount: (container, data) => renderObligations(data),
});

function setStatus(text) {
  statusEl.textContent = text;
}

// fetch() rejects with a plain TypeError when the request never reached a
// server at all (connection refused, DNS failure, offline) -- as opposed
// to a JSON-encoded {error: "..."} from a request that *did* reach this
// process's own server. That distinction is the difference between "the
// local sevenbox ops server isn't up yet" and "something in warrant_core
// rejected this," so it's worth naming instead of surfacing the raw
// "TypeError: Failed to fetch" string, which reads as a crash rather than
// a calm, expected state on a cold start before the server has come up.
function describeFetchError(err) {
  if (err instanceof TypeError) {
    return "Couldn't reach the local server. Is `sevenbox ops` still running?";
  }
  return String(err);
}

// NOT HTML-safe on its own -- JSON.stringify escapes JSON syntax
// characters, not "<"/">"/"&". A Warrant's content is arbitrary data from
// wherever it originated (an oracle's model output, a tool's response, a
// human's free-text axiom) -- every call site that inserts this into
// innerHTML must wrap it in escapeHtml() first (see Decision Browser,
// Timeline, Frontier, Compensation). Callers assigning it via .textContent
// (the graph node tooltip) don't need to, and shouldn't -- escaping here
// would make textContent display literal "&lt;" instead of "<".
function shortContent(content) {
  const s = JSON.stringify(content);
  return s.length > 60 ? s.slice(0, 57) + "..." : s;
}

function formatCallCounts(callCounts) {
  return Object.entries(callCounts || {})
    .map(([name, n]) => `${name}=${n}x`)
    .join(", ");
}

// Title-cases hyphen/underscore-separated System names (e.g.
// "security-incident" -> "Security Incident"). Every System name shown
// through this project was a single word until real multi-word scenario
// slugs (security-incident, and any future one) exposed that the old
// single-char version never handled that case -- a generic display gap,
// not specific to any one scenario.
function capitalize(s) {
  return s
    .split(/[-_]/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function resetNodeSelection() {
  selectedWarrantId = null;
  activeBrowserFilter = "all"; // a filter tuned to the old System's data would be confusing carried into a new one
  explainOutput.innerHTML = '<p class="empty-note">Click a node above.</p>';
  nodeMetadataEl.innerHTML = '<p class="empty-note">Click a node above.</p>';
  renderCausalChain({ nodes: [], edges: [] });
}

// The Persistent Context Bar (docs/ux_review.md P0-1): the one thing that
// stays visible and current no matter which panel is open -- System,
// Execution, Health, Alerts, Selection. Reuses fields every other render
// function already reads (data.mode, data.health, selectedWarrantId) --
// no new data, no new query, just a fifth place those same facts render.
function renderContextBar() {
  contextBarEl.classList.remove("context-bar-healthy", "context-bar-degraded", "context-bar-critical");

  if (!lastData) {
    ctxSystemEl.textContent = "None open";
    ctxExecutionEl.textContent = "—";
    ctxHealthEl.textContent = "—";
    ctxHealthEl.className = "context-value";
    ctxAlertsEl.textContent = "—";
    ctxAlertsEl.className = "context-value";
    ctxSelectionEl.textContent = "None";
    ctxSelectionEl.className = "context-value";
    return;
  }

  ctxSystemEl.textContent =
    currentSystem.kind === "local"
      ? `${currentSystem.path} (local)`
      : `${capitalize(currentSystem.name)} (example)`;

  ctxExecutionEl.textContent = EXECUTION_MODE_LABELS[lastData.mode] || lastData.mode;

  const health = lastData.health;
  if (health) {
    ctxHealthEl.textContent = `${STATUS_ICONS[health.overall] || ""} ${STATUS_LABELS[health.overall] || health.overall}`;
    ctxHealthEl.className = `context-value context-clickable health-${health.overall}`;
    contextBarEl.classList.add(`context-bar-${health.overall}`);

    const alertCount = health.signals.filter((s) => s.status !== "healthy").length;
    ctxAlertsEl.textContent = String(alertCount);
    ctxAlertsEl.className = `context-value context-clickable${alertCount > 0 ? ` health-${health.overall}` : ""}`;
  } else {
    ctxHealthEl.textContent = "—";
    ctxHealthEl.className = "context-value";
    ctxAlertsEl.textContent = "—";
    ctxAlertsEl.className = "context-value";
  }

  if (selectedWarrantId) {
    const w = lastData.warrants.find((x) => x.id === selectedWarrantId);
    ctxSelectionEl.textContent = w ? `${w.short_id}... (${w.source_kind})` : `${selectedWarrantId.slice(0, 8)}...`;
    ctxSelectionEl.className = "context-value context-clickable";
  } else {
    ctxSelectionEl.textContent = "None";
    ctxSelectionEl.className = "context-value";
  }
}

// --- System Picker ---
//
// One origin-agnostic entry point for opening a System, whichever backend
// produces it. "Examples" is populated from GET /api/scenarios --
// unchanged endpoint, unchanged data -- this section only decides how it's
// presented: as Systems to open, never as "scenarios to run." A "Recent"
// section belongs here too, per the product direction, but isn't built
// yet: it needs a persistence mechanism (what counts as "recent," where it
// lives across reloads) that hasn't been designed, and adding an empty
// section that never populates would misrepresent a real capability as
// already built. Left out honestly rather than faked -- see this repo's
// engineering rules on not claiming something works before it does.

function toggleSystemPicker() {
  systemPickerPanel.classList.toggle("hidden");
}

function hideSystemPicker() {
  systemPickerPanel.classList.add("hidden");
}

async function loadExampleSystems() {
  try {
    const res = await fetch("/api/scenarios");
    const data = await res.json();
    systemPickerExamplesEl.innerHTML = data.scenarios
      .map(
        (name) =>
          `<button class="system-picker-item" data-example="${escapeHtml(name)}">${escapeHtml(capitalize(name))}</button>`
      )
      .join("");
    systemPickerExamplesEl.querySelectorAll(".system-picker-item").forEach((btn) => {
      btn.addEventListener("click", () => openExampleSystem(btn.dataset.example));
    });
  } catch (err) {
    // The one network call that fires automatically on page load, before
    // the operator has done anything -- if it fails, that's a cold-start
    // state worth naming calmly in both places someone might look: the
    // picker itself (empty until opened) and the status line (visible
    // immediately, without opening anything).
    const message = describeFetchError(err);
    systemPickerExamplesEl.innerHTML = `<p class="empty-note">${message}</p>`;
    setStatus(message);
  }
}

async function openExampleSystem(name) {
  hideSystemPicker();
  setStatus(`Opening ${name}...`);
  currentSystem = { kind: "example", name };
  resetNodeSelection();
  // Opening a System should feel like opening a new document, not leaving
  // the operator stranded on whatever panel they happened to be on --
  // replayCurrentSystem() deliberately doesn't do this, since a replay is
  // refreshing the same document, not opening a different one.
  showPanel("operations-overview");
  try {
    const res = await fetch(`/api/run?scenario=${encodeURIComponent(name)}`);
    const data = await res.json();
    if (data.error) {
      setStatus(data.error);
      return;
    }
    render(data);
    replayBtn.disabled = false;
    setStatus(`${capitalize(name)} -- live run complete -- ${formatCallCounts(data.call_counts)}.`);
  } catch (err) {
    setStatus(describeFetchError(err));
  }
}

async function replayCurrentSystem() {
  if (currentSystem.kind !== "example") return; // replay-btn is disabled for local systems; defensive only
  setStatus("Replaying from the captured store...");
  replayBtn.disabled = true;
  resetNodeSelection();
  try {
    const res = await fetch(`/api/replay?scenario=${encodeURIComponent(currentSystem.name)}`);
    const data = await res.json();
    if (data.error) {
      setStatus(data.error);
      return;
    }
    render(data);
    setStatus(`Replay complete -- ${formatCallCounts(data.call_counts)} (should be 0s: no real handler ran).`);
  } catch (err) {
    setStatus(describeFetchError(err));
  } finally {
    replayBtn.disabled = false;
  }
}

async function openLocalSystem() {
  const path = loadPathInput.value.trim();
  if (!path) {
    setStatus("Enter a path first.");
    return;
  }
  hideSystemPicker();
  setStatus(`Opening ${path}...`);
  loadBtn.disabled = true;
  currentSystem = { kind: "local", path };
  resetNodeSelection();
  showPanel("operations-overview"); // same reasoning as openExampleSystem() above
  try {
    const res = await fetch(`/api/load?path=${encodeURIComponent(path)}`);
    const data = await res.json();
    if (data.error) {
      setStatus(data.error);
      return;
    }
    render(data);
    // Replay reproduces a captured live run (SevenboxSession's contract) --
    // an arbitrary local system was never a live run this process made.
    replayBtn.disabled = true;
    setStatus(`Loaded ${data.warrant_count} warrant(s) from ${path}.`);
  } catch (err) {
    setStatus(describeFetchError(err));
  } finally {
    loadBtn.disabled = false;
  }
}

// Which /api/explain route to call is a routing fact carried in
// currentSystem (see its declaration above) -- never a domain judgment
// made here.
async function explainWarrant(warrantId) {
  explainOutput.innerHTML = '<p class="empty-note">Loading...</p>';
  try {
    const url = currentSystem.kind === "local"
      ? `/api/load/explain?path=${encodeURIComponent(currentSystem.path)}&warrant_id=${encodeURIComponent(warrantId)}`
      : `/api/explain?scenario=${encodeURIComponent(currentSystem.name)}&warrant_id=${encodeURIComponent(warrantId)}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.error) {
      // The server's error text can reflect a raw URL parameter (e.g.
      // "Unknown warrant_id '<...>'" echoes whatever was requested) --
      // escapeHtml() here, not just on Warrant content, closes the same
      // class of gap for server-reflected text too. See docs/security-
      // review.md.
      explainOutput.innerHTML = `<p class="empty-note">${escapeHtml(data.error)}</p>`;
      renderCausalChain({ nodes: [], edges: [] });
      return;
    }
    explainOutput.innerHTML = data.narrative
      .map((line) => `<div class="explain-line">${escapeHtml(line)}</div>`)
      .join("");
    renderCausalChain(data);
  } catch (err) {
    explainOutput.innerHTML = `<p class="empty-note">${describeFetchError(err)}</p>`;
  }
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function idList(ids) {
  return ids && ids.length ? ids.map((id) => id.slice(0, 8) + "...").join(", ") : "—";
}

function jsonBlock(label, value) {
  return `<details class="json-viewer"><summary>${escapeHtml(label)}</summary><pre>${escapeHtml(
    JSON.stringify(value, null, 2)
  )}</pre></details>`;
}

// Selecting a node means both things a click has always meant in this UI:
// inspect its full metadata (client-side, from the already-fetched
// payload) and fetch its causal narrative (a server round-trip -- explain()
// is a different warrant_core query, not a field on the node itself). Plus,
// updating the one shared selectedWarrantId (the Selection Context) and
// reflecting it on the
// Execution Graph even when the click didn't originate there.
function selectWarrant(warrantId) {
  selectedWarrantId = warrantId;
  inspectNode(warrantId);
  explainWarrant(warrantId);
  if (lastData) {
    renderGraph(lastData);
    renderDecisionBrowser(lastData);
  }
  renderContextBar();
}

// For list panels (Timeline, Compensation): selecting a row means "take me
// to this decision," not just "update it in the background" -- see section
// 13's detail-panel-vs-list-panel distinction. Graph/causal-chain clicks
// call selectWarrant() directly and deliberately don't navigate. The
// Decision Browser is the same case as the graph, not Timeline/Compensation
// -- it's a workspace the operator is actively browsing, not a summary card
// pointing elsewhere, so its rows call selectWarrant() directly too (see
// renderDecisionBrowser below).
function navigateToWarrant(warrantId) {
  selectWarrant(warrantId);
  showPanel("node-inspector");
}

function inspectNode(warrantId) {
  if (!lastData) return;
  const w = lastData.warrants.find((x) => x.id === warrantId);
  renderNodeMetadata(w || null);
}

function renderNodeMetadata(w) {
  if (!w) {
    nodeMetadataEl.innerHTML = '<p class="empty-note">Click a node above.</p>';
    return;
  }

  const origin = w.origin
    ? Object.entries(w.origin)
        .filter(([, v]) => v)
        .map(([k, v]) => `${k}=${escapeHtml(v)}`)
        .join(", ") || "(set, all fields empty)"
    : "—";

  const direct = w.compensation.direct;
  const directText = direct
    ? `${direct.action_required ? "action required" : "no action required"}` +
      `${direct.ambiguous ? " (ambiguous — competing correctors)" : ""} — ${escapeHtml(direct.reason)}`
    : "—";

  const transitive = w.compensation.transitive;
  const transitiveText = transitive.length
    ? transitive
        .map((t) => `owed since ${t.superseded_ancestor_id.slice(0, 8)}... was corrected by ${t.corrector_id.slice(0, 8)}...`)
        .join("<br>")
    : "—";

  // Direct/Transitive above already correctly show "—" once an obligation is
  // discharged (compensation_for()/find_transitive_obligations() stop
  // reporting it) -- which looks identical to "never had one." This row is
  // what makes that distinction visible: *that* and *why* it was resolved,
  // not just silence where an alert used to be.
  const discharges = w.compensation.discharges;
  const dischargesText = discharges.length
    ? discharges
        .map((d) => `${d.outcome === "compensated" ? "compensated" : "not needed"} — ${escapeHtml(d.reason)}`)
        .join("<br>")
    : "—";

  const replay = w.replay_available;
  const replayText = `${replay.available ? (replay.degraded ? "available, degraded" : "available") : "unavailable"} — ${escapeHtml(replay.reason)}`;

  const statusClass = w.is_live ? "status-pill-live" : "status-pill-superseded";
  const statusText = w.is_live ? "Live" : "Superseded";
  const statusNarrative = w.is_live
    ? "Part of what the system currently believes."
    : "No longer part of the current frontier -- something else supersedes it.";

  // Object-page order (Phase 2, Commit 2): Identity, Current status,
  // Replay, Compensation, Effects, Relationships, Evidence, Metadata --
  // the same 13 facts as before, every one still present, in the order
  // an operator actually asks about a decision: what is this, is it
  // still valid, can I reproduce it, do I owe anything for it, what did
  // it actually do, what's it connected to, what did it claim, and only
  // then the bookkeeping. Not a table -- each section is a short answer
  // to one question, not a row of an inventory.
  nodeMetadataEl.innerHTML = `
    <div class="inspector-id">${w.id}</div>
    <div class="inspector-badges">
      <span class="kind-badge kind-${w.source_kind}">${w.source_kind}</span>
      <span class="badge">${w.epistemic}</span>
      <span class="badge">${w.effectful ? "effectful" : "pure"}</span>
      <span class="badge">${w.reversibility}</span>
    </div>

    <h3 class="subsection-heading">Current status</h3>
    <div class="inspector-status">
      <span class="status-pill ${statusClass}">${statusText}</span>
      <span class="inspector-line">${statusNarrative}</span>
    </div>

    <h3 class="subsection-heading">Replay</h3>
    <p class="inspector-line">${replayText}</p>
    <p class="inspector-line empty-note">This is about this one decision's replay key. The header's "Replay from store" button re-derives the whole Execution, not just this decision.</p>

    <h3 class="subsection-heading">Compensation</h3>
    <dl class="metadata-list">
      <dt>Direct</dt><dd>${directText}</dd>
      <dt>Transitive</dt><dd>${transitiveText}</dd>
      <dt>Discharged</dt><dd>${dischargesText}</dd>
    </dl>

    <h3 class="subsection-heading">Effects</h3>
    ${
      w.effect_metadata.present
        ? `<p class="inspector-line">Redaction: ${w.redaction_metadata.applied ? "applied to effect metadata" : "not applied"}</p>` +
          jsonBlock(`request (source: ${w.effect_metadata.source_name || "unknown"})`, w.effect_metadata.request) +
          jsonBlock("response", w.effect_metadata.response)
        : `<p class="inspector-line empty-note">No effect captured -- this decision is pure.</p>`
    }

    <h3 class="subsection-heading">Relationships</h3>
    <dl class="metadata-list">
      <dt>Supersedes</dt><dd>${idList(w.supersedes_id ? [w.supersedes_id] : [])}</dd>
      <dt>Superseded by</dt><dd>${idList(w.superseded_by_ids)}</dd>
      <dt>Parents</dt><dd>${idList(w.parent_ids)}</dd>
      <dt>Children</dt><dd>${idList(w.child_ids)}</dd>
    </dl>

    <h3 class="subsection-heading">Evidence</h3>
    ${jsonBlock("content", w.content)}

    <h3 class="subsection-heading">Metadata</h3>
    <dl class="metadata-list">
      <dt>Origin</dt><dd>${origin}</dd>
    </dl>
  `;
}

function render(data) {
  lastData = data;
  PANELS.forEach((p) => p.mount(document.getElementById(p.sectionId), data));
  renderContextBar();
}

// formatRatio/formatDepth below are display-unit formatting only (fraction
// -> percentage string, float -> one decimal place) -- neither branches on
// whether a number is "good," which is the one thing this file must never
// decide. Every status/health judgment comes from data.health, computed
// once in warrant_core.decision_health and rendered here unchanged.
function formatRatio(ratio) {
  return ratio === null || ratio === undefined ? "N/A" : `${Math.round(ratio * 100)}%`;
}

function formatDepth(depth) {
  return Number(depth).toFixed(1);
}

// Where an alert takes you to investigate it -- an alert is category-level
// (HealthSignal carries no warrant id, see docs/operator_workflows.md
// workflow 1), so it can only navigate to a panel, never select a single
// decision the way Compensation/Timeline rows do. `browserFilter`, when
// present, also applies the matching Decision Browser preset on arrival --
// an alert should land you exactly where you can investigate it, not one
// click short of it. replay_degraded now has a real destination (the
// Decision Browser's own "replay-degraded" preset, built after this map
// was first written -- the two didn't know about each other until now).
const ALERT_NAVIGATION = {
  compensation_backlog: { panel: "compensation" },
  frontier_conflicts: { panel: "frontier" },
  replay_degraded: { panel: "decision-browser", browserFilter: "replay-degraded" },
};

// Sequence 6 polish: a category tag and an icon per signal *name* (the
// stable slug decision_health.py documents, e.g. "compensation_backlog" --
// never derived from `summary`/`detail`, which are prose meant to change
// freely). Purely a display label for a fact warrant_core already computed
// -- same role KIND_COLORS/STATUS_ICONS already play, not a new judgment.
const SIGNAL_CATEGORY_LABEL = {
  compensation_backlog: "Compensation",
  replay_degraded: "Replay Integrity",
  frontier_conflicts: "Frontier Conflict",
};
const SIGNAL_ICON_PATH = {
  compensation_backlog: '<path d="M3 10h10a5 5 0 0 1 5 5v1"/><path d="M7 6 3 10l4 4"/>',
  replay_degraded: '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 3v6h-6"/>',
  frontier_conflicts: '<path d="M10.3 3.9 2.5 18a2 2 0 0 0 1.7 3h15.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
};
// One line of static, honest copy per overall status -- describes what the
// status *means*, not a fact pulled from data (no count, no timestamp).
const HEALTH_SUBTITLE = {
  critical: "Immediate action required.",
  degraded: "Review recommended.",
  healthy: "No action needed.",
};
function inlineSvg(pathMarkup, size) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${pathMarkup}</svg>`;
}
const SHIELD_ICON_PATH = '<path d="M12 2 4 5v6c0 5 3.4 8.7 8 10 4.6-1.3 8-5 8-10V5l-8-3z"/><path d="M12 8v4M12 16h.01"/>';
const CHEVRON_RIGHT_PATH = '<path d="M9 6l6 6-6 6"/>';

// Alerts are not a second data source: they're DecisionHealth.signals
// filtered to the non-healthy ones. No new backend concept, no alert
// persistence/thresholds/history.
function renderHealth(data) {
  const overallEl = document.getElementById("health-overall");
  const alertsHeadingEl = document.getElementById("alerts-heading");
  const alertsEl = document.getElementById("active-alerts");
  const summaryEl = document.getElementById("health-signal-summary");
  const health = data.health;

  ["healthy", "degraded", "critical"].forEach((s) => overallEl.classList.remove(`health-banner-${s}`));

  if (!health) {
    overallEl.innerHTML = '<p class="empty-note">Open a System to see its health.</p>';
    alertsHeadingEl.textContent = "Active Alerts";
    alertsEl.innerHTML = "";
    summaryEl.innerHTML = "";
    return;
  }

  const alerts = health.signals.filter((s) => s.status !== "healthy");

  overallEl.classList.add(`health-banner-${health.overall}`);
  overallEl.innerHTML = `
    <span class="health-overall-icon health-overall-icon-${health.overall}">${inlineSvg(SHIELD_ICON_PATH, 22)}</span>
    <span class="health-overall-body">
      <span class="health-overall-title health-${health.overall}">${STATUS_LABELS[health.overall] || health.overall}</span>
      <span class="health-overall-subtitle">${HEALTH_SUBTITLE[health.overall] || ""}</span>
    </span>
    ${
      alerts.length
        ? `<button type="button" class="health-overall-alerts-link" id="health-overall-alerts-link">
             <span class="status-pill status-pill-${health.overall === "critical" ? "critical" : "degraded"}">${alerts.length}</span>
             Active alert${alerts.length === 1 ? "" : "s"}
             ${inlineSvg(CHEVRON_RIGHT_PATH, 16)}
           </button>`
        : ""
    }
  `;
  const alertsLink = document.getElementById("health-overall-alerts-link");
  if (alertsLink) {
    alertsLink.addEventListener("click", () => {
      document.getElementById("alerts-heading").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  alertsHeadingEl.textContent = `Active Alerts (${alerts.length})`;
  alertsEl.innerHTML = alerts.length
    ? alerts
        .map((s) => {
          const nav = ALERT_NAVIGATION[s.name];
          const attrs = nav
            ? ` data-panel-target="${nav.panel}"${nav.browserFilter ? ` data-browser-filter="${nav.browserFilter}"` : ""}`
            : "";
          const pillVariant = s.status === "critical" ? "critical" : "degraded";
          return `
            <li class="alert-item alert-${s.status}${nav ? " alert-clickable" : ""}"${attrs}>
              <span class="alert-icon alert-icon-${s.status}">${inlineSvg(SIGNAL_ICON_PATH[s.name] || SHIELD_ICON_PATH, 17)}</span>
              <span class="alert-body">
                <span class="alert-body-top">
                  <span class="alert-title">${escapeHtml(s.summary)}</span>
                  <span class="status-pill status-pill-${pillVariant}">${STATUS_LABELS[s.status] || s.status}</span>
                </span>
                <span class="signal-detail">${escapeHtml(s.detail)}</span>
                ${SIGNAL_CATEGORY_LABEL[s.name] ? `<span class="alert-tag alert-tag-${pillVariant}">${escapeHtml(SIGNAL_CATEGORY_LABEL[s.name])}</span>` : ""}
              </span>
            </li>
          `;
        })
        .join("")
    // "Where do I go next?" (Phase 2, Commit 4) still needs an answer
    // when the honest answer is "nowhere urgent" -- a healthy system
    // doesn't leave the operator without a next step, it points at the
    // Execution Graph, the same primary-artifact panel Commit 1 already
    // established as where an investigation naturally starts.
    : `
        <li class="alert-item alert-item-clear alert-clickable" data-panel-target="execution-graph">
          <span class="alert-icon alert-icon-healthy">${inlineSvg(SHIELD_ICON_PATH, 17)}</span>
          <span class="alert-body">
            <span class="alert-title health-healthy">No active alerts -- all checks passing.</span>
            <span class="signal-detail">Nothing needs attention right now -- explore the Execution Graph →</span>
          </span>
        </li>
      `;
  alertsEl.querySelectorAll(".alert-clickable").forEach((li) => {
    li.addEventListener("click", () => {
      if (li.dataset.browserFilter) setBrowserFilter(li.dataset.browserFilter);
      showPanel(li.dataset.panelTarget);
    });
  });

  // Three real counts (always summing to health.signals.length -- 3 today),
  // never a fourth "resolved" figure warrant_core has no history to back --
  // see the comment on #health-signal-summary in index.html.
  const bySeverity = { critical: 0, degraded: 0, healthy: 0 };
  health.signals.forEach((s) => { bySeverity[s.status] = (bySeverity[s.status] || 0) + 1; });
  summaryEl.innerHTML = ["critical", "degraded", "healthy"]
    .map(
      (status) => `
        <div class="signal-summary-tile signal-summary-tile-${status}">
          <span class="signal-summary-icon">${inlineSvg(SHIELD_ICON_PATH, 16)}</span>
          <span class="signal-summary-value">${bySeverity[status]}</span>
          <span class="signal-summary-label">${STATUS_LABELS[status] || status}</span>
        </div>
      `
    )
    .join("");
}

function renderMetrics(data) {
  const grid = document.getElementById("metrics-grid");
  const m = data.metrics;
  if (!m) {
    grid.innerHTML = "";
    return;
  }

  const tiles = [
    [m.total_decisions, "Decisions"],
    [m.frontier_size, "Frontier"],
    [m.pending_compensation_count, "Pending Compensation"],
    [m.conflict_count, "Conflicts"],
    [formatRatio(m.replayable_ratio), "Replay Coverage"],
    [m.degraded_replay_count, "Degraded Replay Keys"],
    [m.redacted_count, "Redactions"],
    [formatDepth(m.average_causal_depth), "Avg. Causal Depth"],
    [m.superseded_count, "Superseded (all-time)"],
  ];

  grid.innerHTML = tiles
    .map(
      ([value, label]) => `
        <div class="metric-tile">
          <span class="metric-value">${value}</span>
          <span class="metric-label">${label}</span>
        </div>
      `
    )
    .join("");
}

// Shared by the main Explorer graph and the causal chain viewer in the
// Node Inspector -- same layout algorithm, same KIND_COLORS lookup, same
// live/superseded CSS classes, same short_id labels. Deliberately one
// drawing function, not two similar ones: a node's visual identity (color,
// shape, label) is guaranteed consistent between the two views *because*
// it's the same code, not because two implementations were kept in sync by
// hand.
//
// `nodes`: [{id, shortId, sourceKind, isLive, layer, content}]
//   `layer` is caller-assigned x-position-by-generation; the two callers
//   mean different things by it (insertion-order depth for the main graph,
//   inverted causal distance-from-target for the chain viewer) and both
//   pre-compute it before calling this function -- this function only lays
//   nodes out by whatever layer number it's given.
// `edges`: [{from, to, kind}], kind is "parent" or "supersede" (CSS class only).
function drawGraphSvg(svg, nodes, edges, { onNodeClick, highlightId } = {}) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!nodes.length) return;

  const byLayer = {};
  nodes.forEach((n) => {
    (byLayer[n.layer] = byLayer[n.layer] || []).push(n);
  });

  const layerWidth = 170;
  const rowHeight = 76;
  const marginX = 60;
  const marginY = 40;
  const radius = 24;

  const positions = {};
  const maxLayer = Math.max(...nodes.map((n) => n.layer), 0);
  const maxRows = Math.max(...Object.values(byLayer).map((l) => l.length), 1);

  Object.entries(byLayer).forEach(([layer, list]) => {
    list.forEach((n, i) => {
      positions[n.id] = {
        x: marginX + Number(layer) * layerWidth,
        y: marginY + i * rowHeight + (maxRows - list.length) * (rowHeight / 2),
      };
    });
  });

  const width = marginX * 2 + (maxLayer + 1) * layerWidth;
  const height = Math.max(200, marginY * 2 + maxRows * rowHeight);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const ns = "http://www.w3.org/2000/svg";
  function el(tag, attrs) {
    const e = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    return e;
  }

  // edges first, so nodes draw on top
  edges.forEach((edge) => {
    const from = positions[edge.from];
    const to = positions[edge.to];
    if (!from || !to) return;
    svg.appendChild(
      el("line", {
        x1: from.x, y1: from.y, x2: to.x, y2: to.y,
        class: edge.kind === "supersede" ? "edge-supersede" : "edge-parent",
      })
    );
  });

  nodes.forEach((n) => {
    const p = positions[n.id];
    const g = el("g", {});
    const classes = ["node-circle", n.isLive ? "live" : "superseded"];
    const isHighlighted = n.id === highlightId;
    if (isHighlighted) classes.push("node-highlight");
    // A highlighted "tool" node's ring color (--accent) sits close in hue
    // to --kind-tool's own stroke color -- a thicker same-hue ring reads
    // as barely-there next to a tool node specifically, even though it's
    // obvious on every other kind (confirmed: a real operator missed it).
    // Bumping the radius too makes selection visible by *size*, which
    // doesn't depend on hue contrast at all, on top of the existing
    // stroke-width/color change in .node-highlight (style.css).
    const circle = el("circle", {
      cx: p.x, cy: p.y, r: isHighlighted ? radius + 5 : radius,
      class: classes.join(" "),
      stroke: KIND_COLORS[n.sourceKind] || "#888",
    });
    if (onNodeClick) circle.addEventListener("click", () => onNodeClick(n.id));
    g.appendChild(circle);

    const label = el("text", { x: p.x, y: p.y + 3, "text-anchor": "middle", class: "node-label" });
    label.textContent = n.shortId;
    label.style.pointerEvents = "none";
    g.appendChild(label);

    const kindLabel = el("text", {
      x: p.x, y: p.y + radius + 13, "text-anchor": "middle", class: "node-kind-label",
    });
    kindLabel.textContent = n.sourceKind;
    g.appendChild(kindLabel);

    const titleEl = el("title", {});
    titleEl.textContent = `${n.sourceKind}${n.isLive ? "" : " (superseded)"}\n${shortContent(n.content)}\n(click to inspect)`;
    g.appendChild(titleEl);

    svg.appendChild(g);
  });
}

// The Decision Browser: the generic way to discover decisions -- filter
// the current Execution's own
// data.warrants by a fact warrant_core already computed, click a row to
// select it. No fetch, no search, no new backend query -- BROWSER_FILTERS'
// predicates read exactly the fields render_node_details() already sends.
// "Effects", "Replay", "Decision Search", "Redactions" from earlier
// future-panel lists are not separate panels -- they're exactly the presets
// below, one filterable browser underneath all of them.
function setBrowserFilter(filterId) {
  activeBrowserFilter = filterId;
  if (lastData) renderDecisionBrowser(lastData);
}

// Decision Browser as an investigation workspace, not a filter list
// (Phase 2, Commit 3): "All" stands alone as the reset state, the other
// seven are grouped into "Needs attention" / "Explore" so picking one
// feels like choosing what to investigate, not toggling a database
// facet. Every filtered view (not "All") also gets a persistent context
// line above the results restating the question being asked -- an
// operator scrolling a filtered list should never lose track of which
// question they're currently answering.
function renderBrowserFilterButton(f, data) {
  const count = data.warrants.filter(f.predicate).length;
  const active = f.id === activeBrowserFilter ? " browser-filter-active" : "";
  return `<button class="browser-filter-btn${active}" data-filter-id="${f.id}" title="${escapeHtml(f.description)}">${escapeHtml(f.label)} (${count})</button>`;
}

function renderDecisionBrowser(data) {
  const filterBar = document.getElementById("browser-filter-bar");
  const resultsEl = document.getElementById("browser-results");
  if (!data) {
    filterBar.innerHTML = "";
    resultsEl.innerHTML = '<p class="empty-note">Open a System to browse its decisions.</p>';
    return;
  }

  const allFilter = BROWSER_FILTERS.find((f) => f.group === null);
  const groupIds = ["attention", "explore"];
  filterBar.innerHTML = `
    <div class="browser-filter-row">${renderBrowserFilterButton(allFilter, data)}</div>
    ${groupIds
      .map(
        (g) => `
          <div class="browser-filter-group">
            <span class="browser-filter-group-label">${escapeHtml(BROWSER_GROUP_LABELS[g])}</span>
            <div class="browser-filter-row">
              ${BROWSER_FILTERS.filter((f) => f.group === g).map((f) => renderBrowserFilterButton(f, data)).join("")}
            </div>
          </div>
        `
      )
      .join("")}
  `;
  filterBar.querySelectorAll(".browser-filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => setBrowserFilter(btn.dataset.filterId));
  });

  const active = BROWSER_FILTERS.find((f) => f.id === activeBrowserFilter) || BROWSER_FILTERS[0];
  const matches = data.warrants.filter(active.predicate);

  // Investigation flow (docs/ux_review.md priority 4): when a filter can
  // only possibly mean one decision, select it -- the operator would have
  // clicked it next anyway. Not hidden behavior: it's the same one rule
  // every time ("exactly one match -> select it"), true whether the
  // filter came from a manual click or an alert's navigation. Guarded on
  // the id actually changing so this can't recurse against itself --
  // selectWarrant() re-renders this same panel to reflect the selection.
  if (matches.length === 1 && matches[0].id !== selectedWarrantId) {
    selectWarrant(matches[0].id);
    return;
  }

  const contextLine = active.id === "all"
    ? ""
    : `<p class="browser-context"><strong>${escapeHtml(active.label)}</strong> — ${escapeHtml(active.description)}</p>`;

  const rowsHtml = matches.length
    ? matches
        .map((w) => {
          const selected = w.id === selectedWarrantId ? " browser-row-selected" : "";
          return `
            <div class="browser-row${selected}" data-warrant-id="${w.id}">
              <span class="browser-row-kind kind-badge kind-${w.source_kind}">${w.source_kind}</span>
              <span class="browser-row-content">${escapeHtml(shortContent(w.content))}</span>
              <span class="browser-row-flags">${w.is_live ? "" : "superseded"}</span>
            </div>
          `;
        })
        .join("")
    : '<p class="empty-note">No decisions currently match this.</p>';

  resultsEl.innerHTML = contextLine + rowsHtml;
  // Selecting a row only updates the Selection Context -- it does not
  // navigate. The Browser is a workspace the operator is actively looking
  // at, the same situation as a graph click, not a summary card like
  // Compensation/Timeline -- see navigateToWarrant()'s comment above.
  resultsEl.querySelectorAll(".browser-row").forEach((row) => {
    row.addEventListener("click", () => selectWarrant(row.dataset.warrantId));
  });
}

function renderGraph(data) {
  // Only ever runs once a System is actually open (render()/mount() never
  // fire before then) -- so unconditionally hiding the empty-note here,
  // rather than showing it again on some other condition, is correct: a
  // System, once opened, stays open for the rest of the session.
  document.getElementById("graph-empty-note").classList.add("hidden");
  const svg = document.getElementById("graph-svg");
  const ordered = [...data.warrants].sort((a, b) => a.order - b.order);
  const nodes = ordered.map((w) => ({
    id: w.id, shortId: w.short_id, sourceKind: w.source_kind,
    isLive: w.is_live, layer: w.layer, content: w.content,
  }));
  const edges = [];
  data.warrants.forEach((w) => {
    w.parent_ids.forEach((pid) => edges.push({ from: pid, to: w.id, kind: "parent" }));
    if (w.supersedes_id) edges.push({ from: w.id, to: w.supersedes_id, kind: "supersede" });
  });
  drawGraphSvg(svg, nodes, edges, { onNodeClick: selectWarrant, highlightId: selectedWarrantId });
}

// The causal chain viewer: draws data.nodes/data.edges from an
// /api/explain response exactly as received. No relationship is
// reconstructed from warrants here -- warrant_core.explain() already did
// that walk; this function only lays out and draws the DAG it returned.
function renderCausalChain(data) {
  const svg = document.getElementById("causal-chain-svg");
  if (!svg) return;
  if (!data.nodes || !data.nodes.length) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    return;
  }
  // explain()'s `depth` is causal distance FROM the target (0 = target
  // itself, increasing toward its roots) -- the opposite direction of the
  // main graph's `layer` (0 = roots, increasing toward derived results).
  // Inverting here keeps the same left-to-right "causes flow into effects"
  // reading in both diagrams -- a presentation choice, made once, here,
  // not a new fact about the graph.
  const maxDepth = Math.max(...data.nodes.map((n) => n.depth), 0);
  const nodes = data.nodes.map((n) => ({
    id: n.warrant_id, shortId: n.short_id, sourceKind: n.source_kind,
    isLive: n.is_live, layer: maxDepth - n.depth, content: n.content,
  }));
  const edges = data.edges.map((e) => ({ from: e.parent_id, to: e.child_id, kind: "parent" }));
  drawGraphSvg(svg, nodes, edges, { onNodeClick: selectWarrant, highlightId: data.target_id });
}

// The only place dimmed=superseded and dashed-red=supersede are explained
// now that the graph panel's header carries no instructional prose
// (Phase 2 Commit 1 moved it here entirely, not just duplicated it --
// docs/ux_review.md P2-1 is where the color-only version of this gap was
// first found).
function renderLegend() {
  const legend = document.getElementById("legend");
  const kindItems = Object.entries(KIND_COLORS)
    .map(
      ([kind, color]) =>
        `<span><span class="legend-swatch" style="background:${color}"></span>${kind}</span>`
    )
    .join("");
  const structuralItems = `
    <span><span class="legend-swatch legend-swatch-dim" style="background:${KIND_COLORS.axiom}"></span>superseded (dimmed)</span>
    <span><span class="legend-line"></span>causal parent</span>
    <span><span class="legend-line legend-line-supersede"></span>supersede</span>
  `;
  legend.innerHTML = kindItems + structuralItems;
}

function renderTimeline(data) {
  // Reads data.timeline.entries -- warrant_core.decision_timeline's own
  // projection of the graph, already in causal/insertion order (not a
  // timestamp -- see decision_timeline.py) and already classified per
  // entry (event_kind). This function does not sort data.warrants or
  // infer "is this a correction" itself; that would be exactly the
  // "generate events in the UI" mistake the domain layer exists to avoid.
  //
  // The one thing it does cross-reference: data.obligations (already
  // present in the same payload, computed by warrant_core, not derived
  // here) to mark an irreversible_action row still awaiting compensation
  // -- directly answering this panel's own question ("did a correction
  // arrive before or after an irreversible action already acted"). Reuses
  // .status-pill-pending exactly as Compensation defines it, not a new
  // component.
  const pendingIds = new Set((data.obligations || []).map((o) => o.irreversible_id));
  const list = document.getElementById("timeline-list");
  list.innerHTML = "";
  data.timeline.entries.forEach((e) => {
    const li = document.createElement("li");
    li.className = "timeline-item" + (e.is_live ? "" : " superseded");
    li.style.cursor = "pointer";
    const pendingPill =
      e.event_kind === "irreversible_action" && pendingIds.has(e.warrant_id)
        ? `<span class="status-pill status-pill-pending timeline-pending-pill">Pending compensation</span>`
        : "";
    li.innerHTML = `
      <span class="timeline-index">${e.order}</span>
      <span class="timeline-kind kind-badge kind-${e.source_kind}">${e.source_kind}</span>
      <span class="timeline-event-kind timeline-event-${e.event_kind}">${EVENT_KIND_LABELS[e.event_kind] || e.event_kind}</span>
      <span class="timeline-content">${escapeHtml(shortContent(e.content))}</span>
      ${pendingPill}
    `;
    li.addEventListener("click", () => navigateToWarrant(e.warrant_id));
    list.appendChild(li);
  });
}

function renderFrontier(data) {
  const liveEl = document.getElementById("frontier-live");
  const conflictsEl = document.getElementById("frontier-conflicts");

  // Conflicts get a subsection heading only when there's something to
  // attend to -- the common case (no conflicts) stays exactly as clean
  // as before, nothing new to scan past. Same "needs attention" framing
  // Decision Browser's filter groups and Operations Overview's alerts
  // already established, not a new pattern invented for this panel.
  conflictsEl.innerHTML = data.frontier.conflicts.length
    ? `<h3 class="subsection-heading">Unresolved conflicts</h3>` +
      data.frontier.conflicts
        .map(
          (c) =>
            `<div class="conflict-item">Conflict on <span class="mono-inline">${c.target_id.slice(0, 8)}...</span> -- ` +
            `${c.competing_superseder_ids.length} competing corrections, unresolved.</div>`
        )
        .join("")
    : "";

  const liveWarrants = data.warrants.filter((w) => data.frontier.live_ids.includes(w.id));
  const liveHeading = data.frontier.conflicts.length ? `<h3 class="subsection-heading">Live</h3>` : "";
  liveEl.innerHTML = liveWarrants.length
    ? liveHeading +
      liveWarrants
        .map(
          (w) =>
            `<div class="frontier-item">` +
            `<span class="browser-row-kind kind-badge kind-${w.source_kind}">${w.source_kind}</span>` +
            `<span class="mono-inline">${escapeHtml(shortContent(w.content))}</span>` +
            `</div>`
        )
        .join("")
    : '<p class="empty-note">Nothing live yet -- open a System.</p>';
}

function renderObligations(data) {
  // Status pill + one-sentence narrative -- the same pattern Node
  // Inspector's Current Status section established (.inspector-status
  // wrapping a .status-pill and a narrative line), reused here rather
  // than left as a bare paragraph. .status-pill-pending is a new color
  // variant on the *existing* .status-pill family, the same way
  // .status-pill-live/.status-pill-superseded are just variants of it --
  // not a new component.
  const el = document.getElementById("obligations-list");
  el.innerHTML = data.obligations.length
    ? data.obligations
        .map(
          (o) =>
            `<div class="obligation-item obligation-clickable" data-warrant-id="${o.irreversible_id}">` +
            `<div class="inspector-status">` +
            `<span class="status-pill status-pill-pending">Pending compensation</span>` +
            `<span class="inspector-line"><span class="mono-inline">${escapeHtml(shortContent(o.irreversible_content))}</span>'s justification ` +
            `(ancestor <span class="mono-inline">${o.superseded_ancestor_id.slice(0, 8)}...</span>) was corrected ` +
            `by <span class="mono-inline">${o.corrector_id.slice(0, 8)}...</span>, and nothing supersedes the action itself directly.</span>` +
            `</div>` +
            `</div>`
        )
        .join("")
    : '<p class="empty-note">No open compensation obligations.</p>';
  el.querySelectorAll(".obligation-clickable").forEach((div) => {
    div.addEventListener("click", () => navigateToWarrant(div.dataset.warrantId));
  });
}

openSystemBtn.addEventListener("click", toggleSystemPicker);
replayBtn.addEventListener("click", replayCurrentSystem);
loadBtn.addEventListener("click", openLocalSystem);
loadPathInput.addEventListener("keydown", (evt) => {
  if (evt.key === "Enter") openLocalSystem();
});

// A dropdown that only closes by re-clicking the same button it opened
// with is a dead end the moment someone clicks anywhere else expecting it
// to go away -- the standard pattern every dropdown in software this app
// is being held to (Linear, GitHub, Datadog) already uses. Click outside
// or Escape closes it; clicking inside it (an example, the path input,
// the Open button) never does.
document.addEventListener("click", (evt) => {
  if (systemPickerPanel.classList.contains("hidden")) return;
  if (systemPickerPanel.contains(evt.target) || evt.target === openSystemBtn) return;
  hideSystemPicker();
});
document.addEventListener("keydown", (evt) => {
  if (evt.key === "Escape") hideSystemPicker();
});

// Context Bar quick-jumps -- always the same destination, never surprising:
// Health/Alerts always mean "go see why" (Operations Overview); Selection
// always means "go see what" (Node Inspector). No-ops when there's nothing
// to jump to yet, guarded the same way renderContextBar() itself is.
ctxHealthEl.addEventListener("click", () => { if (lastData) showPanel("operations-overview"); });
ctxAlertsEl.addEventListener("click", () => { if (lastData) showPanel("operations-overview"); });
ctxSelectionEl.addEventListener("click", () => { if (selectedWarrantId) showPanel("node-inspector"); });

// Theme toggle -- a client-only display preference (localStorage), never a
// fact warrant_core knows about or a server round-trip. The inline script
// at the top of index.html already applied a stored "light" choice before
// this file loaded (avoids a flash of dark-then-light); this only wires the
// click handler and keeps the icon/aria-label in sync going forward.
function applyTheme(theme) {
  if (theme === "light") {
    document.documentElement.setAttribute("data-theme", "light");
    themeToggleBtn.setAttribute("aria-label", "Switch to dark theme");
  } else {
    document.documentElement.removeAttribute("data-theme");
    themeToggleBtn.setAttribute("aria-label", "Switch to light theme");
  }
}
function initThemeToggle() {
  let stored = null;
  try { stored = localStorage.getItem("sevenbox-theme"); } catch (e) { /* private/blocked storage -- fall back to dark */ }
  applyTheme(stored === "light" ? "light" : "dark");
  themeToggleBtn.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    applyTheme(next);
    try { localStorage.setItem("sevenbox-theme", next); } catch (e) { /* preference just won't persist across reloads */ }
  });
}

renderLegend();
renderPanelNav();
showPanel("operations-overview");
renderContextBar();
loadExampleSystems();
initThemeToggle();
setStatus('Click "Open System..." to observe an example, or open a local system.');
