from typing import TypedDict

import pytest

langgraph = pytest.importorskip("langgraph")
from langgraph.graph import END, START, StateGraph  # noqa: E402

from warrant_core import RedactionAction, RedactionPolicy, SourceKind  # noqa: E402
from sevenbox_adapters.langgraph_adapter import sevenbox_node, sevenbox_run  # noqa: E402


class KYCState(TypedDict, total=False):
    applicant: dict
    verification: dict


def verify_identity(state):
    return {"verification": {"ssn": state["applicant"]["ssn"], "status": "verified"}}


def test_langgraph_adapter_redacts_effect_record_when_declared():
    g = StateGraph(KYCState)
    g.add_node("verify", sevenbox_node(
        verify_identity, source_kind=SourceKind.TOOL,
        redaction=RedactionPolicy(fields={"ssn": RedactionAction.HASH}),
    ))
    g.add_edge(START, "verify")
    g.add_edge("verify", END)
    compiled = g.compile()

    with sevenbox_run() as krun:
        result = compiled.invoke({"applicant": {"ssn": "123-45-6789", "name": "Ana"}})

    # LangGraph's own result is completely unaffected by redaction
    assert result["verification"]["ssn"] == "123-45-6789"

    tool_warrants = [w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL]
    assert len(tool_warrants) == 1
    assert tool_warrants[0].effect_record.request["applicant"]["ssn"].startswith("sha256:")
    assert tool_warrants[0].effect_record.request["applicant"]["name"] == "Ana"
    # content itself (what downstream nodes and derivations see) is never redacted
    assert tool_warrants[0].content["verification"]["ssn"] == "123-45-6789"
    assert tool_warrants[0].effect_record.redacted is True


def test_langgraph_adapter_unaffected_without_a_policy():
    g = StateGraph(KYCState)
    g.add_node("verify", sevenbox_node(verify_identity, source_kind=SourceKind.TOOL))
    g.add_edge(START, "verify")
    g.add_edge("verify", END)
    compiled = g.compile()

    with sevenbox_run() as krun:
        compiled.invoke({"applicant": {"ssn": "123-45-6789", "name": "Ana"}})

    tool_warrants = [w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL]
    assert tool_warrants[0].effect_record.request["applicant"]["ssn"] == "123-45-6789"
    assert tool_warrants[0].effect_record.redacted is False
