# OpenAI Agents SDK

```bash
pip install "warrant-core[openai-agents]"
```

One change to an existing Agents SDK app: `@sevenbox_tool` under
`@function_tool` on each tool function, `sevenbox_agent_run()` around the
`Runner.run(...)` call site. Passing `hooks=krun.hooks` is what gives
genuinely parallel tool calls correct, independent parents.

```python
from agents import Agent, Runner, function_tool
from warrant_core import SourceKind
from sevenbox_adapters.openai_agents_adapter import sevenbox_tool, sevenbox_agent_run

@function_tool
@sevenbox_tool(source_kind=SourceKind.TOOL)
def lookup_order(order_id: str) -> dict:
    return {"status": "shipped"}

agent = Agent(name="Support", tools=[lookup_order], instructions="...", model=your_model)

async def main():
    with sevenbox_agent_run() as krun:
        await Runner.run(agent, "Where is order 4471?", hooks=krun.hooks)
        for w in krun.session.frontier().live:
            print(w.content)
```

`@sevenbox_tool` goes **under** `@function_tool`, not the other way
around — `@function_tool` needs the real function signature to build
its JSON schema, and stacking them the other way breaks that.

Verified against the installed `openai-agents` package with a
network-free fake `Model` producing a real tool call — the pattern
above is the exact wiring that ran; `your_model` stands in for a real
`Model` (a live API-backed one, or a network-free test double). A
complete, runnable, no-API-key version is in
[`tests/test_openai_agents_adapter.py`](../../tests/test_openai_agents_adapter.py);
a full worked example with replay is
[`examples/refund_scenario_openai_agents.py`](../../examples/refund_scenario_openai_agents.py).
