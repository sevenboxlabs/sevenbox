import pytest

from warrant_core import (
    CompensationObligation,
    DischargeError,
    DischargeOutcome,
    EffectRecord,
    EpistemicClass,
    ReversibilityClass,
    SourceKind,
    WarrantStore,
    compensation_for,
    create_warrant,
    discharge_obligation,
    find_transitive_obligations,
    supersede,
)


def _root(store):
    return store.add(create_warrant(
        content={"claim": "root"}, source_kind=SourceKind.AXIOM,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
    ))


def _direct_scenario(store):
    """An irreversible action, directly superseded once -- the case
    compensation_for() never checked for an existing discharge before this fix."""
    root = _root(store)
    action = store.add(create_warrant(
        content={"claim": "refund_action(4471)"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.IRREVERSIBLE,
        parent_ids=(root.id,),
    ))
    correction = supersede(
        store, target_id=action.id, content={"claim": "action was erroneous"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="q", response="a"),
    )
    return action, correction


# --- Bug fix: compensation_for() must respect discharges, same as
# find_transitive_obligations() already did ---

def test_compensation_for_stops_reporting_once_discharged():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    assert obligation.action_required is True

    comp_action = store.add(create_warrant(
        content={"claim": "reversed the refund"}, source_kind=SourceKind.TOOL,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.COMPENSABLE,
        parent_ids=(action.id,),
        effect_record=EffectRecord(request="q", response="a"),
    ))
    discharge_obligation(
        store, obligation, DischargeOutcome.COMPENSATED,
        reason="reversed manually", evidence_id=comp_action.id,
    )

    assert compensation_for(store, action.id) is None


# --- Core discharge behavior ---

def test_discharge_as_compensated_cites_target_corrector_and_evidence():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    comp_action = store.add(create_warrant(
        content={"claim": "reversed the refund"}, source_kind=SourceKind.TOOL,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.COMPENSABLE,
        parent_ids=(action.id,),
        effect_record=EffectRecord(request="q", response="a"),
    ))
    discharge = discharge_obligation(
        store, obligation, DischargeOutcome.COMPENSATED,
        reason="reversed manually", evidence_id=comp_action.id,
    )
    assert set(discharge.parent_ids) == {action.id, correction.id, comp_action.id}
    assert discharge.content["discharge"]["outcome"] == "compensated"


def test_discharge_as_not_needed_does_not_require_evidence():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    discharge = discharge_obligation(
        store, obligation, DischargeOutcome.NOT_NEEDED,
        reason="target was already deleted outside Sevenbox",
    )
    assert discharge.content["discharge"]["evidence_id"] is None
    assert compensation_for(store, action.id) is None


def test_discharge_compensated_without_evidence_is_rejected():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    with pytest.raises(DischargeError):
        discharge_obligation(store, obligation, DischargeOutcome.COMPENSATED, reason="done")


def test_discharge_rejects_empty_reason():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    with pytest.raises(DischargeError):
        discharge_obligation(store, obligation, DischargeOutcome.NOT_NEEDED, reason="")


def test_discharge_refuses_conflicting_corrections_direct_case():
    store = WarrantStore()
    root = _root(store)
    action = store.add(create_warrant(
        content={"claim": "refund_action(4471)"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.IRREVERSIBLE,
        parent_ids=(root.id,),
    ))
    supersede(
        store, target_id=action.id, content={"claim": "already refunded, no-op"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="q1", response="a1"),
    )
    supersede(
        store, target_id=action.id, content={"claim": "fraudulent, must claw back"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="q2", response="a2"),
    )
    obligation = compensation_for(store, action.id)
    assert obligation.ambiguous is True
    with pytest.raises(DischargeError):
        discharge_obligation(store, obligation, DischargeOutcome.NOT_NEEDED, reason="pick one")


def test_discharge_refuses_conflicting_correctors_transitive_case():
    store = WarrantStore()
    root = _root(store)
    eligible = store.add(create_warrant(
        content={"claim": "order eligible"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        parent_ids=(root.id,),
    ))
    action = store.add(create_warrant(
        content={"claim": "refund_action(4471)"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.IRREVERSIBLE,
        parent_ids=(eligible.id,),
    ))
    supersede(
        store, target_id=eligible.id, content={"claim": "already refunded"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="q1", response="a1"),
    )
    supersede(
        store, target_id=eligible.id, content={"claim": "fraudulent"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="q2", response="a2"),
    )
    obligations = find_transitive_obligations(store)
    assert len(obligations) == 2  # one per competing corrector -- see compensation.py's note
    with pytest.raises(DischargeError):
        discharge_obligation(
            store, obligations[0], DischargeOutcome.NOT_NEEDED, reason="pick one"
        )


def test_discharge_transitive_obligation_empties_the_walk():
    store = WarrantStore()
    root = _root(store)
    eligible = store.add(create_warrant(
        content={"claim": "order eligible"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        parent_ids=(root.id,),
    ))
    action = store.add(create_warrant(
        content={"claim": "refund_action(4471)"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.IRREVERSIBLE,
        parent_ids=(eligible.id,),
    ))
    supersede(
        store, target_id=eligible.id, content={"claim": "already refunded"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="q", response="a"),
    )
    obligations = find_transitive_obligations(store)
    assert len(obligations) == 1
    discharge_obligation(
        store, obligations[0], DischargeOutcome.NOT_NEEDED,
        reason="write target had already been removed",
    )
    assert find_transitive_obligations(store) == []


# --- Idempotency ---

def test_discharging_same_obligation_twice_same_outcome_is_idempotent():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    first = discharge_obligation(
        store, obligation, DischargeOutcome.NOT_NEEDED, reason="moot"
    )
    size_after_first = len(store)
    second = discharge_obligation(
        store, obligation, DischargeOutcome.NOT_NEEDED, reason="moot, said again"
    )
    assert second.id == first.id
    assert len(store) == size_after_first  # no duplicate Warrant was added


def test_discharging_same_obligation_different_outcome_is_rejected():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    discharge_obligation(store, obligation, DischargeOutcome.NOT_NEEDED, reason="moot")
    comp_action = store.add(create_warrant(
        content={"claim": "reversed the refund"}, source_kind=SourceKind.TOOL,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.COMPENSABLE,
        parent_ids=(action.id,),
        effect_record=EffectRecord(request="q", response="a"),
    ))
    with pytest.raises(DischargeError):
        discharge_obligation(
            store, obligation, DischargeOutcome.COMPENSATED,
            reason="actually we did compensate", evidence_id=comp_action.id,
        )


# --- source_kind / epistemic defaults ---

def test_discharge_defaults_to_human_source_and_assumed_epistemic():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    discharge = discharge_obligation(
        store, obligation, DischargeOutcome.NOT_NEEDED, reason="moot"
    )
    assert discharge.source_kind == SourceKind.HUMAN
    assert discharge.epistemic == EpistemicClass.ASSUMED
    assert discharge.effectful is True
    assert discharge.effect_record is not None


def test_discharge_with_overridden_source_kind_and_explicit_epistemic():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    discharge = discharge_obligation(
        store, obligation, DischargeOutcome.NOT_NEEDED, reason="policy decided",
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
    )
    assert discharge.source_kind == SourceKind.TOOL
    assert discharge.epistemic == EpistemicClass.VERIFIED


def test_discharge_with_overridden_source_kind_falls_back_to_default_table():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    discharge = discharge_obligation(
        store, obligation, DischargeOutcome.NOT_NEEDED, reason="policy decided",
        source_kind=SourceKind.ORACLE,
    )
    assert discharge.epistemic == EpistemicClass.INFERRED  # _DEFAULT_EPISTEMIC_FOR_SOURCE


def test_discharge_with_unmappable_source_kind_requires_explicit_epistemic():
    store = WarrantStore()
    action, correction = _direct_scenario(store)
    obligation = compensation_for(store, action.id)
    with pytest.raises(DischargeError):
        discharge_obligation(
            store, obligation, DischargeOutcome.NOT_NEEDED, reason="policy decided",
            source_kind=SourceKind.SENSOR,
        )


# --- Defensive re-check against current store state, not a stale snapshot ---

def test_discharge_refuses_when_nothing_is_actually_pending():
    store = WarrantStore()
    root = _root(store)
    never_superseded = store.add(create_warrant(
        content={"claim": "never corrected"}, source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.IRREVERSIBLE,
        parent_ids=(root.id,),
    ))
    # A hand-built obligation standing in for a stale snapshot -- discharge_obligation()
    # must recompute fresh state rather than trust it.
    fake_obligation = CompensationObligation(
        target=never_superseded, action_required=True, ambiguous=False,
        reason="stale", competing_superseder_ids=["nonexistent-id"],
    )
    with pytest.raises(DischargeError):
        discharge_obligation(
            store, fake_obligation, DischargeOutcome.NOT_NEEDED, reason="moot"
        )
