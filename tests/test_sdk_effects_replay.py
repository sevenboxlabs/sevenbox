from sevenbox_sdk import SevenboxSession, derive, irreversible, oracle, tool


@oracle
def infer_intent(ticket):
    return {"intent": "refund_request", "order_id": ticket["order_id"]}


@tool
def lookup_order(order_id):
    return {"status": "shipped"}


@derive
def eligibility(order):
    return {"eligible": order["status"] == "shipped"}


@irreversible
@derive
def refund_action(elig):
    if not elig["eligible"]:
        return None
    return {"action": "refund_action"}


def run_scenario(session: SevenboxSession):
    ticket = session.axiom({"order_id": "4471"})
    intent = infer_intent(ticket)
    order = lookup_order(intent["order_id"])
    elig = eligibility(order)
    action = refund_action(elig)
    return action


def test_sdk_effect_functions_are_called_in_live_mode():
    calls = []

    @oracle
    def counting_intent(ticket):
        calls.append(ticket)
        return {"intent": "refund_request", "order_id": ticket["order_id"]}

    with SevenboxSession() as session:
        ticket = session.axiom({"order_id": "1"})
        counting_intent(ticket)

    assert len(calls) == 1


def test_sdk_live_then_replay_never_calls_real_functions_again():
    call_log = []

    @oracle
    def tracked_intent(ticket):
        call_log.append(("oracle", ticket))
        return {"intent": "refund_request", "order_id": ticket["order_id"]}

    @tool
    def tracked_lookup(order_id):
        call_log.append(("tool", order_id))
        return {"status": "shipped"}

    def scenario(session):
        ticket = session.axiom({"order_id": "4471"})
        intent = tracked_intent(ticket)
        order = tracked_lookup(intent["order_id"])
        elig = eligibility(order)
        return refund_action(elig)

    with SevenboxSession() as live_session:
        live_action = scenario(live_session)

    assert len(call_log) == 2  # one oracle call, one tool call
    live_ids = {w.id for w in live_session.frontier().live}

    with SevenboxSession(replay_from=live_session.store) as replay_session:
        replay_action = scenario(replay_session)

    # no NEW calls happened during replay
    assert len(call_log) == 2
    assert replay_action.value == live_action.value

    replay_ids = {w.id for w in replay_session.frontier().live}
    assert replay_ids == live_ids


def test_full_scenario_produces_a_flagged_compensation_after_correction():
    with SevenboxSession() as session:
        action = run_scenario(session)
        assert action.value == {"action": "refund_action"}

        # find the eligibility warrant to supersede -- walk up from action
        eligibility_id = None
        for parent_id in session.store.get(action.warrant_id).parent_ids:
            eligibility_id = parent_id
        assert eligibility_id is not None

        from warrant_core import EffectRecord, EpistemicClass, ReversibilityClass, SourceKind, supersede
        supersede(
            session.store, target_id=eligibility_id,
            content={"claim": "already refunded"}, source_kind=SourceKind.TOOL,
            epistemic=EpistemicClass.VERIFIED, reversibility=ReversibilityClass.PURE,
            effect_record=EffectRecord(request="q", response="already refunded"),
        )

        obligations = session.obligations()
        assert len(obligations) == 1
        assert obligations[0].irreversible.id == action.warrant_id
