"""
Tests for sevenbox_adapters.mastra_adapter.import_mastra_log.

Unlike every other adapter's test file in this project, these tests do
NOT spin up the real framework -- Mastra is TypeScript/Node-only, and
this adapter is a log importer, not an in-process wrapper (see
sevenbox_adapters/mastra_adapter.py's module docstring for why). Instead,
these tests read real fixture logs captured by ACTUALLY running
mastra_bridge/examples/kyc_scenario_parallel.mjs and
kyc_scenario_sequential.mjs against a real `@mastra/core@1.57.0` `Agent`
(network-free, verified in the session that built this adapter) --
`tests/fixtures/mastra_kyc_parallel_log.jsonl` and
`mastra_kyc_sequential_log.jsonl` are that real captured output, checked
in verbatim, not hand-written. This keeps the Python test suite runnable
with `pytest tests/ -q` and no Node.js requirement, while still testing
against genuine framework output rather than a fabricated approximation
of its shape.
"""
from pathlib import Path

from warrant_core import SourceKind, EpistemicClass, ReversibilityClass, create_warrant
from sevenbox_adapters.mastra_adapter import import_mastra_log

FIXTURES = Path(__file__).parent / "fixtures"
PARALLEL_LOG = FIXTURES / "mastra_kyc_parallel_log.jsonl"
SEQUENTIAL_LOG = FIXTURES / "mastra_kyc_sequential_log.jsonl"


def test_wrapped_run_populates_a_warrant_store():
    result = import_mastra_log(PARALLEL_LOG)

    assert len(result.store) == 3  # axiom + 2 tool calls
    axiom = next(w for w in result.store.all() if w.source_kind == SourceKind.AXIOM)
    assert axiom.content == "Run KYC on A-991"
    assert axiom.id == result.run_axiom_id

    watchlist_w = next(w for w in result.store.all() if isinstance(w.content, dict) and "watchlist_match" in w.content)
    document_w = next(w for w in result.store.all() if isinstance(w.content, dict) and "document_verified" in w.content)

    assert watchlist_w.source_kind == SourceKind.TOOL
    assert watchlist_w.effect_record is not None
    assert watchlist_w.effect_record.source_name == "check_watchlist"
    assert watchlist_w.effect_record.request == {"args": {"applicant_id": "A-991"}}
    assert document_w.effect_record.source_name == "verify_document"


def test_two_parallel_tool_calls_get_correct_independent_parents():
    result = import_mastra_log(PARALLEL_LOG)

    axiom = next(w for w in result.store.all() if w.source_kind == SourceKind.AXIOM)
    watchlist_w = next(w for w in result.store.all() if isinstance(w.content, dict) and "watchlist_match" in w.content)
    document_w = next(w for w in result.store.all() if isinstance(w.content, dict) and "document_verified" in w.content)

    assert watchlist_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids != (watchlist_w.id,), "no false dependency between siblings of one step"
    assert result.frontier == frozenset({watchlist_w.id, document_w.id})


def test_sequential_calls_chain_correctly_across_steps():
    result = import_mastra_log(SEQUENTIAL_LOG)

    assert len(result.store) == 3  # axiom + 2 tool calls, one per step
    axiom = next(w for w in result.store.all() if w.source_kind == SourceKind.AXIOM)
    watchlist_w = next(w for w in result.store.all() if isinstance(w.content, dict) and "watchlist_match" in w.content)
    document_w = next(w for w in result.store.all() if isinstance(w.content, dict) and "document_verified" in w.content)

    assert watchlist_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids == (watchlist_w.id,), "the second step must chain onto the first, not the axiom"
    assert result.frontier == frozenset({document_w.id})


def test_run_finish_produces_no_warrant():
    result = import_mastra_log(PARALLEL_LOG)

    assert not any(
        w.content == "KYC complete." for w in result.store.all()
    ), "run_finish is informational only -- it must never become a Warrant"


def test_importing_the_same_log_twice_is_idempotent():
    """Replay, for this adapter, means re-processing the same log --
    never re-executing anything (module docstring's "Replay is different
    here, not absent"). Content-addressing guarantees identical ids
    across two independent imports of the identical file into fresh
    stores."""
    first = import_mastra_log(PARALLEL_LOG)
    second = import_mastra_log(PARALLEL_LOG)

    assert {w.id for w in first.store.all()} == {w.id for w in second.store.all()}
    assert first.frontier == second.frontier
    assert first.run_axiom_id == second.run_axiom_id


def test_two_independent_agent_logs_can_be_merged_into_one_decision():
    """The `Agent -> Agent A / Agent B -> Decision` shape: import two
    independent logs into the SAME store, then parent a Decision Warrant
    on both logs' own final frontiers."""
    shared_store = None
    result_a = import_mastra_log(PARALLEL_LOG, store=shared_store)
    shared_store = result_a.store
    result_b = import_mastra_log(SEQUENTIAL_LOG, store=shared_store)

    assert len(shared_store) == 6  # 3 warrants from each independent log

    decision = create_warrant(
        content={"decision": "proceed"},
        source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        parent_ids=tuple(sorted(result_a.frontier | result_b.frontier)),
    )
    shared_store.add(decision)

    assert set(decision.parent_ids) == result_a.frontier | result_b.frontier
    assert len(decision.parent_ids) == 3  # 2 from the parallel log's frontier + 1 from the sequential log's


def test_step_with_no_preceding_run_start_gets_a_lazy_axiom():
    raw_log = (
        '{"type":"step","step_index":1,"calls":'
        '[{"call_id":"c1","name":"lookup_order","args":{"order_id":"4471"},'
        '"ok":true,"result":{"status":"shipped"}}]}'
    )
    result = import_mastra_log(raw_log)

    assert len(result.store) == 2  # lazily-created axiom + 1 tool call
    axiom = next(w for w in result.store.all() if w.source_kind == SourceKind.AXIOM)
    tool_w = next(w for w in result.store.all() if w.source_kind == SourceKind.TOOL)
    assert tool_w.parent_ids == (axiom.id,)


def test_a_failed_call_still_produces_a_warrant_with_no_response():
    raw_log = (
        '{"type":"run_start","input":"go"}\n'
        '{"type":"step","step_index":1,"calls":'
        '[{"call_id":"c1","name":"flaky_tool","args":{"x":1},"ok":false,"result":null}]}'
    )
    result = import_mastra_log(raw_log)

    tool_w = next(w for w in result.store.all() if w.source_kind == SourceKind.TOOL)
    assert tool_w.content is None
    assert tool_w.effect_record.response is None
    assert tool_w.effect_record.source_name == "flaky_tool"
