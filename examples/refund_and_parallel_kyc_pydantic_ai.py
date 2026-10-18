"""
The refund scenario one more time -- this time as an actual PydanticAI
app (built against the real `pydantic_ai` package), wrapped with
sevenbox_adapters.pydantic_ai_adapter instead of rewritten in Sevenbox's own
vocabulary. Same "install in an afternoon" comparison the other two
framework examples make: two changes -- `@sevenbox_tool` under `@agent.tool`,
`sevenbox_agent_run()` around the run call site. Tool bodies untouched.

Also demonstrates this adapter's headline difference from the other two:
a SECOND scenario below (KYC, two independent checks) gets correct,
independent parents for genuinely parallel tool calls with NO third
change -- no `hooks=krun.hooks` (openai_agents_adapter), no
`ADDITIONAL_PARENTS_KEY` (langgraph_adapter). See the module docstring
of sevenbox_adapters/pydantic_ai_adapter.py for why: PydanticAI's own
RunContext.run_step already carries the turn information the other two
frameworks don't expose to a tool call directly.

Uses pydantic_ai.models.function.FunctionModel for a real, network-free
offline run -- no API key needed.

Run: PYTHONPATH=. python3 examples/refund_and_parallel_kyc_pydantic_ai.py
"""
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from warrant_core import EpistemicClass, ReversibilityClass, SourceKind, find_transitive_obligations, supersede
from warrant_core import EffectRecord
from sevenbox_adapters.pydantic_ai_adapter import sevenbox_agent_run, sevenbox_tool


# --- Scenario 1: sequential refund, same shape as the other two examples ---

def refund_model_fn(messages, info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart(tool_name="lookup_order", args={"order_id": "4471"})])
    if len(messages) == 3:
        return ModelResponse(parts=[ToolCallPart(tool_name="issue_refund", args={"order_id": "4471", "amount": 340})])
    return ModelResponse(parts=[TextPart(content="Order 4471 has shipped -- refund issued.")])


refund_agent = Agent(FunctionModel(refund_model_fn))


@refund_agent.tool
@sevenbox_tool(source_kind=SourceKind.TOOL, name="lookup_order")
def lookup_order(ctx: RunContext, order_id: str) -> dict:
    return {"status": "shipped", "order_id": order_id}


@refund_agent.tool
@sevenbox_tool(source_kind=SourceKind.TOOL, reversibility=ReversibilityClass.IRREVERSIBLE, name="issue_refund")
def issue_refund(ctx: RunContext, order_id: str, amount: int) -> dict:
    return {"action": "refund_issued", "order_id": order_id, "amount": amount}


def run_refund_scenario():
    print("=" * 60)
    print("SCENARIO 1: sequential refund")
    print("=" * 60)

    with sevenbox_agent_run() as krun:
        krun.session.start("Refund order 4471")
        result = refund_agent.run_sync("Refund order 4471")

    print("PydanticAI's own result (unchanged):", result.output)
    print("\nWhat Sevenbox captured, that plain PydanticAI would not give you:")
    for w in krun.session.store.all():
        print(f"  - [{w.source_kind.value:6s} / {w.reversibility.value:11s}] {w.content}")

    obligations_before = find_transitive_obligations(krun.session.store)
    order_warrant = next(w for w in krun.session.store.all() if "status" in w.content)
    supersede(
        krun.session.store, target_id=order_warrant.id,
        content={"claim": "order #4471 was already refunded earlier"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="refund-history check", response="already refunded"),
    )
    obligations = find_transitive_obligations(krun.session.store)
    print(f"\nCompensation obligations found: {len(obligations)} (before correction: {len(obligations_before)})")
    for o in obligations:
        print(f"  - {o.irreversible.content} needs compensation")

    print("\nReplaying the same run from the captured store:")
    with sevenbox_agent_run(replay_from=krun.session.store) as replay_run:
        replay_run.session.start("Refund order 4471")
        replay_result = refund_agent.run_sync("Refund order 4471")
    assert replay_result.output == result.output
    print("Replay reproduced the exact result, with no real tool bodies executed.")


# --- Scenario 2: genuine parallel fan-out, zero extra wiring ---

def kyc_model_fn(messages, info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:  # ONE turn, TWO tool calls
        return ModelResponse(parts=[
            ToolCallPart(tool_name="check_watchlist", args={"applicant_id": "A-991"}),
            ToolCallPart(tool_name="verify_document", args={"applicant_id": "A-991"}),
        ])
    return ModelResponse(parts=[TextPart(content="KYC check complete for A-991.")])


kyc_agent = Agent(FunctionModel(kyc_model_fn))


@kyc_agent.tool
@sevenbox_tool(source_kind=SourceKind.TOOL, name="check_watchlist")
def check_watchlist(ctx: RunContext, applicant_id: str) -> dict:
    return {"applicant_id": applicant_id, "watchlist_match": False}


@kyc_agent.tool
@sevenbox_tool(source_kind=SourceKind.TOOL, name="verify_document")
def verify_document(ctx: RunContext, applicant_id: str) -> dict:
    return {"applicant_id": applicant_id, "document_verified": True}


def run_parallel_kyc_scenario():
    print("\n" + "=" * 60)
    print("SCENARIO 2: genuine parallel fan-out (no hooks=, no ADDITIONAL_PARENTS_KEY)")
    print("=" * 60)

    with sevenbox_agent_run() as krun:
        krun.session.start("Run KYC on A-991")
        result = kyc_agent.run_sync("Run KYC on A-991")

    print("PydanticAI's own result (unchanged):", result.output)
    print("\nWarrant graph:")
    for w in krun.session.store.all():
        parents = [p[:8] for p in w.parent_ids] or ["(none)"]
        print(f"  id={w.id[:8]}  [{w.source_kind.value:6s}]  parents={parents}  content={w.content}")

    warrants = {w.id: w for w in krun.session.store.all()}
    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    watchlist_w = next(w for w in warrants.values() if "watchlist_match" in w.content)
    document_w = next(w for w in warrants.values() if "document_verified" in w.content)

    assert watchlist_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids == (axiom.id,)
    assert document_w.parent_ids != (watchlist_w.id,)
    print("\nBoth checks correctly cite only the axiom as their parent -- no false")
    print("dependency between two genuinely independent pieces of evidence, and no")
    print("extra code was needed to get this right.")


if __name__ == "__main__":
    run_refund_scenario()
    run_parallel_kyc_scenario()
