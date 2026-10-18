"""
The refund scenario one more time -- this time as an actual OpenAI Agents
SDK app (built against the real `openai-agents` package, `import agents`),
wrapped with sevenbox_adapters.openai_agents_adapter instead of rewritten in
Sevenbox's own vocabulary.

Two changes to an existing Agents SDK app: `@sevenbox_tool` under
`@function_tool` on each tool function, wrap the `Runner.run(...)` call
site in `sevenbox_agent_run()`. Neither tool function's body is touched.

Runs with zero API key and zero network calls -- the `FakeModel` below is
a real `agents.models.interface.Model` implementation, not a mock of
Sevenbox's own code. It exists so this example is runnable offline, exactly
like the LangGraph example needs no real LLM either; swap it for
`from agents.models.openai_provider import OpenAIProvider` (or any other
real provider) and nothing about the Sevenbox integration below changes --
`sevenbox_tool`/`sevenbox_agent_run` don't know or care which Model produced the
tool call.

Run: PYTHONPATH=. python3 examples/refund_scenario_openai_agents.py
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

from warrant_core import (
    EpistemicClass,
    ReversibilityClass,
    SourceKind,
    find_transitive_obligations,
    supersede,
)
from warrant_core import EffectRecord
from sevenbox_adapters.openai_agents_adapter import sevenbox_agent_run, sevenbox_tool


# --- these two functions are completely ordinary Agents SDK tools -----------
# --- nothing about their bodies changes to work with Sevenbox --------------------

@function_tool
@sevenbox_tool(source_kind=SourceKind.TOOL)
def lookup_order(order_id: str) -> dict:
    return {"status": "shipped", "order_id": order_id}


@function_tool
@sevenbox_tool(source_kind=SourceKind.TOOL, reversibility=ReversibilityClass.IRREVERSIBLE)
def issue_refund(order_id: str) -> dict:
    return {"action": "refund_issued", "order_id": order_id}


class FakeModel(Model):
    """Scripted, offline stand-in for a real LLM: look up the order, then
    issue the refund, then answer. See module docstring -- swap for a
    real provider, nothing else changes."""

    def __init__(self):
        self._turn = 0

    async def get_response(self, system_instructions, input, model_settings, tools,
                            output_schema, handoffs, tracing, *, previous_response_id,
                            conversation_id, prompt) -> ModelResponse:
        self._turn += 1
        if self._turn == 1:
            call = ResponseFunctionToolCall(
                type="function_call", call_id="call_1", name="lookup_order",
                arguments=json.dumps({"order_id": "4471"}),
            )
            return ModelResponse(output=[call], usage=Usage(), response_id="fake-1")
        if self._turn == 2:
            call = ResponseFunctionToolCall(
                type="function_call", call_id="call_2", name="issue_refund",
                arguments=json.dumps({"order_id": "4471"}),
            )
            return ModelResponse(output=[call], usage=Usage(), response_id="fake-2")
        msg = ResponseOutputMessage(
            id="fake-3", type="message", role="assistant", status="completed",
            content=[ResponseOutputText(
                type="output_text", text="Order 4471 has shipped -- refund issued.", annotations=[],
            )],
        )
        return ModelResponse(output=[msg], usage=Usage(), response_id="fake-3")

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("not exercised by this example")


class FakeModelProvider(ModelProvider):
    def __init__(self, model):
        self._model = model

    def get_model(self, model_name):
        return self._model


def build_agent():
    return Agent(
        name="RefundAgent",
        instructions="Help customers with refund requests: look up the order, then issue the refund.",
        tools=[lookup_order, issue_refund],
    )


async def main():
    agent = build_agent()

    # --- change 1 already made above (@sevenbox_tool); change 2: wrap the run ---
    with sevenbox_agent_run() as krun:
        krun.session.start({"ticket": {"order_id": "4471"}})
        result = await Runner.run(
            agent, "I'd like a refund for order 4471",
            run_config=RunConfig(model_provider=FakeModelProvider(FakeModel())),
        )

    print("Agents SDK's own result (unchanged):", result.final_output)

    print("\nWhat Sevenbox captured, that plain Agents SDK would not give you:")
    for w in krun.session.frontier().live:
        print(f"  - [{w.source_kind.value:6s} / {w.reversibility.value:12s}] {w.content}")

    # New evidence arrives after the fact -- same pattern as every prior
    # example: the order lookup (an ANCESTOR of the already-issued,
    # irreversible refund) gets corrected, not the refund action itself.
    order_warrant = next(
        w for w in krun.session.store.all()
        if w.source_kind == SourceKind.TOOL and w.reversibility == ReversibilityClass.PURE
    )
    supersede(
        krun.session.store, target_id=order_warrant.id,
        content={"claim": "order #4471 was already refunded earlier"},
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="refund-history check", response="already refunded"),
    )

    obligations = find_transitive_obligations(krun.session.store)
    print(f"\nCompensation obligations found: {len(obligations)}")
    for o in obligations:
        print(f"  - {o.irreversible.content} needs compensation")

    # Replay: rerun the same agent, same Runner.run() call, zero real tool
    # calls made -- neither tool's body (and, with a real Model, the LLM
    # call) executes again.
    print("\nReplaying the same run from the captured store:")
    with sevenbox_agent_run(replay_from=krun.session.store) as replay_run:
        replay_run.session.start({"ticket": {"order_id": "4471"}})
        replay_result = await Runner.run(
            agent, "I'd like a refund for order 4471",
            run_config=RunConfig(model_provider=FakeModelProvider(FakeModel())),
        )

    assert replay_result.final_output == result.final_output
    print("Replay reproduced the Agents SDK's own result exactly, with no "
          "real tool body executed.")


if __name__ == "__main__":
    asyncio.run(main())
