"""
Same KYC scenario as the original bug-finding test
(test_parallel_toolcalls_kyc.py), now run against the REAL, integrated
fix: krun.hooks passed to Runner.run(...), instead of a separate
TurnAwareAgentSession proof-of-concept. This is what a Sevenbox user gets
today, out of the box, with the one-line change the module docstring
now documents.

Run: PYTHONPATH=. python3 examples/refund_scenario_openai_agents_parallel_kyc.py
"""
import asyncio
import json

from agents import Agent, Runner, RunConfig, function_tool
from agents.models.interface import Model, ModelProvider, ModelResponse
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from warrant_core import SourceKind
from sevenbox_adapters.openai_agents_adapter import sevenbox_agent_run, sevenbox_tool


@function_tool
@sevenbox_tool(source_kind=SourceKind.TOOL)
def check_watchlist(applicant_id: str) -> dict:
    return {"applicant_id": applicant_id, "watchlist_match": False}


@function_tool
@sevenbox_tool(source_kind=SourceKind.TOOL)
def verify_document(applicant_id: str) -> dict:
    return {"applicant_id": applicant_id, "document_verified": True}


class ParallelToolCallFakeModel(Model):
    """Returns BOTH tool calls in ONE response -- a genuine single-turn
    fan-out."""

    def __init__(self):
        self._turn = 0

    async def get_response(self, system_instructions, input, model_settings, tools,
                            output_schema, handoffs, tracing, *, previous_response_id,
                            conversation_id, prompt) -> ModelResponse:
        self._turn += 1
        if self._turn == 1:
            call_1 = ResponseFunctionToolCall(
                type="function_call", call_id="call_1", name="check_watchlist",
                arguments=json.dumps({"applicant_id": "A-991"}),
            )
            call_2 = ResponseFunctionToolCall(
                type="function_call", call_id="call_2", name="verify_document",
                arguments=json.dumps({"applicant_id": "A-991"}),
            )
            return ModelResponse(output=[call_1, call_2], usage=Usage(), response_id="fake-1")
        msg = ResponseOutputMessage(
            id="fake-2", type="message", role="assistant", status="completed",
            content=[ResponseOutputText(
                type="output_text", text="KYC check complete for A-991.", annotations=[],
            )],
        )
        return ModelResponse(output=[msg], usage=Usage(), response_id="fake-2")

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("not exercised by this test")


class ParallelToolCallFakeModelProvider(ModelProvider):
    def __init__(self, model):
        self._model = model

    def get_model(self, model_name):
        return self._model


async def main():
    agent = Agent(
        name="KYCAgent",
        instructions="Run KYC checks: check the watchlist and verify the document.",
        tools=[check_watchlist, verify_document],
    )

    with sevenbox_agent_run() as krun:
        krun.session.start({"applicant_id": "A-991"})
        result = await Runner.run(
            agent, "Run KYC on applicant A-991",
            hooks=krun.hooks,  # <-- the entire fix, from the caller's side
            run_config=RunConfig(model_provider=ParallelToolCallFakeModelProvider(ParallelToolCallFakeModel())),
        )

    print("Agents SDK's own result (unchanged):", result.final_output)
    print("\n--- Warrant graph (real, integrated fix, via hooks=krun.hooks) ---")
    warrants = {w.id: w for w in krun.session.store.all()}
    for w in krun.session.store.all():
        parent_summary = [p[:8] for p in w.parent_ids] or ["(none)"]
        print(f"  id={w.id[:8]}  [{w.source_kind.value:6s}]  parents={parent_summary}  content={w.content}")

    axiom = next(w for w in warrants.values() if w.source_kind == SourceKind.AXIOM)
    watchlist_w = next(w for w in warrants.values() if "watchlist_match" in w.content)
    document_w = next(w for w in warrants.values() if "document_verified" in w.content)

    assert watchlist_w.parent_ids == (axiom.id,), "watchlist check should cite only the axiom"
    assert document_w.parent_ids == (axiom.id,), "document check should cite only the axiom"
    assert document_w.parent_ids != (watchlist_w.id,), "no false dependency between siblings"
    print("\nAll assertions passed: both checks are independent children of the axiom.")


if __name__ == "__main__":
    asyncio.run(main())
