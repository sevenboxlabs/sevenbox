import pytest

from sevenbox_sdk import SevenboxSession, Traced, derive, irreversible


@derive
def double(n):
    return n * 2


@derive
def maybe_none(n):
    return None if n > 10 else n


def test_no_session_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="No active SevenboxSession"):
        double(21)  # not wrapped in `with SevenboxSession():`


def test_axiom_and_derive_produce_traced_values():
    with SevenboxSession() as session:
        n = session.axiom(21)
        assert isinstance(n, Traced)
        result = double(n)
        assert isinstance(result, Traced)
        assert result.value == 42
        assert len(session.store) == 2


def test_derive_requires_at_least_one_traced_argument():
    with SevenboxSession():
        with pytest.raises(ValueError, match="No Traced"):
            double(21)  # raw int, not session.axiom(21) -- no causal grounding


def test_none_return_propagates_as_plain_none_for_ordinary_control_flow():
    with SevenboxSession() as session:
        n = session.axiom(20)
        result = maybe_none(n)
        assert result is None
        # no Warrant should have been created for a None result
        assert len(session.store) == 1  # just the axiom


def test_getitem_projection_preserves_warrant_id():
    with SevenboxSession() as session:
        record = session.axiom({"order_id": "4471", "status": "shipped"})
        projected = record["order_id"]
        assert isinstance(projected, Traced)
        assert projected.value == "4471"
        assert projected.warrant_id == record.warrant_id


def test_irreversible_modifier_sets_reversibility():
    @irreversible
    @derive
    def act(n):
        return {"action": n}

    with SevenboxSession() as session:
        n = session.axiom(5)
        result = act(n)
        w = session.store.get(result.warrant_id)
        from warrant_core import ReversibilityClass
        assert w.reversibility == ReversibilityClass.IRREVERSIBLE


def test_two_sessions_do_not_leak_into_each_other():
    with SevenboxSession() as s1:
        a = s1.axiom(1)
        double(a)
        assert len(s1.store) == 2

    with SevenboxSession() as s2:
        assert len(s2.store) == 0


def test_an_explicitly_passed_empty_store_is_not_silently_discarded():
    """Regression: `store or WarrantStore()` used Python truthiness, and an
    empty store (len() == 0) is falsy, so it was being silently replaced with
    a fresh WarrantStore() even when the caller explicitly passed one. Found
    while wiring up SqliteWarrantStore, which is empty exactly when you'd
    most want to use it (a fresh database)."""
    from warrant_core import WarrantStore
    my_store = WarrantStore()
    with SevenboxSession(store=my_store) as session:
        assert session.store is my_store
        a = session.axiom(1)
        double(a)
    assert len(my_store) == 2
