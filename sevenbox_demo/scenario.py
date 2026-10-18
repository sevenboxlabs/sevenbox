"""
sevenbox_demo.scenario -- the refund scenario used throughout every phase of this
project, running here exactly the way a customer's application would: through
`sevenbox_sdk`'s public decorators and `SevenboxSession`, plus `warrant_core`'s public
`supersede()` and `find_transitive_obligations()`. Nothing in this file imports
from a private module or reaches into warrant_core/warrant_runtime/sevenbox_sdk
internals -- it is ordinary application code sitting on top of the SDK, same as
examples/refund_scenario_sdk.py.

This file does not modify warrant_core, warrant_runtime, sevenbox_sdk, or
sevenbox_adapters in any way -- it only imports their public surface.
"""
from __future__ import annotations

from typing import Dict

from sevenbox_sdk import SevenboxSession, derive, irreversible, oracle, tool
from warrant_core import (
    EffectRecord,
    EpistemicClass,
    ReversibilityClass,
    SourceKind,
    WarrantStore,
    supersede,
)

# Call counters, reset at the start of each run. This lives entirely in demo
# code -- it's how the UI proves "the real handler was not called again during
# replay" without needing any support from the SDK itself, which already
# guarantees that behavior (see sevenbox_sdk's own tests); this is just making it
# visible.
_call_counts: Dict[str, int] = {"infer_intent": 0, "lookup_order": 0}


@oracle
def infer_intent(ticket):
    _call_counts["infer_intent"] += 1
    return {"intent": "refund_request", "order_id": ticket["order_id"]}


@tool
def lookup_order(order_id):
    _call_counts["lookup_order"] += 1
    return {"status": "shipped"}


@derive
def eligibility(order):
    return {"eligible": order["status"] == "shipped"}


@irreversible
@derive
def refund_action(elig):
    if not elig["eligible"]:
        return None
    return {"action": "refund_action(4471)"}


def _run(session: SevenboxSession) -> None:
    ticket = session.axiom({"order_id": "4471"})
    intent = infer_intent(ticket)
    order = lookup_order(intent["order_id"])
    elig = eligibility(order)
    refund_action(elig)

    # New evidence arrives after the fact -- the same out-of-band correction
    # pattern used in every prior example in this project. This is not
    # something a rule reacts to automatically; it's an explicit correction,
    # via the public `supersede()` function, same as a customer would call it.
    supersede(
        session.store, target_id=elig.warrant_id,
        content={"claim": "order #4471 was already refunded earlier"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(
            request="refund-history check", response="already refunded"
        ),
    )


def run_live() -> tuple[SevenboxSession, Dict[str, int]]:
    _call_counts["infer_intent"] = 0
    _call_counts["lookup_order"] = 0
    session = SevenboxSession()
    with session:
        _run(session)
    return session, dict(_call_counts)


def run_replay(reference_store: WarrantStore) -> tuple[SevenboxSession, Dict[str, int]]:
    _call_counts["infer_intent"] = 0
    _call_counts["lookup_order"] = 0
    session = SevenboxSession(replay_from=reference_store)
    with session:
        _run(session)
    return session, dict(_call_counts)
