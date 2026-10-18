import pytest

from warrant_core import SourceKind, EpistemicClass, ReversibilityClass
from sevenbox_adapters import ADDITIONAL_PARENTS_KEY, GraphSession


def test_sequential_calls_keep_the_frontier_at_size_one():
    """Backward compatibility, stated as a property: a strictly sequential
    chain of record_node() calls never grows the internal frontier past one
    element -- matching the old scalar-pointer behavior exactly."""
    gs = GraphSession()
    gs._ensure_started({"ticket": 91})
    assert len(gs._frontier) == 1

    def step_one(state):
        return {"n": 1}

    def step_two(state):
        return {"n": 2}

    gs.record_node(
        node_name="step_one", source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        input_state={}, real_fn=step_one,
    )
    assert len(gs._frontier) == 1

    gs.record_node(
        node_name="step_two", source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        input_state={}, real_fn=step_two,
    )
    assert len(gs._frontier) == 1

    # And parent chaining behaves exactly as the old scalar pointer did:
    # the second node's parent is the first node's output, not the axiom.
    warrants = gs.store.all()
    axiom = warrants[0]
    first = warrants[1]
    second = warrants[2]
    assert first.parent_ids == (axiom.id,)
    assert second.parent_ids == (first.id,)


def test_explicit_additional_parents_produces_a_real_multi_parent_warrant():
    """The achievable, honest version of fan-in: a pure node explicitly
    declares which prior Warrant(s) it's joining, rather than the adapter
    trying (and failing) to infer LangGraph's topology automatically."""
    gs = GraphSession()
    gs._ensure_started({"seed": True})
    root_id = next(iter(gs._frontier))

    def branch(state):
        return {"branch_result": "x"}

    gs.record_node(
        node_name="branch", source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        input_state={}, real_fn=branch,
    )
    branch_id = next(iter(gs._frontier))

    def explicit_join(state):
        return {"joined": True, ADDITIONAL_PARENTS_KEY: [root_id]}

    gs.record_node(
        node_name="explicit_join", source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        input_state={}, real_fn=explicit_join,
    )
    result_id = next(iter(gs._frontier))
    result_warrant = gs.store.get(result_id)

    assert set(result_warrant.parent_ids) == {root_id, branch_id}
    # the reserved key never leaks into the stored content
    assert ADDITIONAL_PARENTS_KEY not in result_warrant.content
    assert result_warrant.content == {"joined": True}


def test_additional_parents_on_an_effectful_node_raises_loudly():
    gs = GraphSession()
    gs._ensure_started({"seed": True})

    def bad_effectful_node(state):
        return {"x": 1, ADDITIONAL_PARENTS_KEY: ["some-id"]}

    with pytest.raises(ValueError, match="only supported for pure"):
        gs.record_node(
            node_name="bad", source_kind=SourceKind.TOOL,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            input_state={}, real_fn=bad_effectful_node,
        )
    # the failed call must not have mutated the frontier or the store
    assert len(gs.store) == 1


def test_additional_parents_referencing_unknown_id_raises_clearly():
    gs = GraphSession()
    gs._ensure_started({"seed": True})

    def bad_node(state):
        return {"x": 1, ADDITIONAL_PARENTS_KEY: ["not-a-real-warrant-id"]}

    with pytest.raises(ValueError, match="not present in the store"):
        gs.record_node(
            node_name="bad", source_kind=SourceKind.DERIVED,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            input_state={}, real_fn=bad_node,
        )


def test_additional_parents_key_absent_is_fully_backward_compatible():
    """No declaration at all -- today's exact behavior, byte for byte."""
    gs = GraphSession()
    gs._ensure_started({"seed": True})
    root_id = next(iter(gs._frontier))

    def plain_node(state):
        return {"x": 1}

    gs.record_node(
        node_name="plain", source_kind=SourceKind.DERIVED,
        epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
        input_state={}, real_fn=plain_node,
    )
    result_id = next(iter(gs._frontier))
    result_warrant = gs.store.get(result_id)
    assert result_warrant.parent_ids == (root_id,)
    assert result_warrant.content == {"x": 1}
