# PydanticAI

```bash
pip install "warrant-core[pydantic-ai]"
```

One change to an existing PydanticAI app: `@sevenbox_tool` under
`@agent.tool` on each tool function, `sevenbox_agent_run()` around the
`agent.run(...)`/`run_sync(...)` call site. Correct parallel fan-out is
the **default** here, no extra `hooks=` needed — PydanticAI's
`RunContext.run_step` already tells the adapter which calls belong to
the same turn.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from warrant_core import SourceKind
from sevenbox_adapters.pydantic_ai_adapter import sevenbox_tool, sevenbox_agent_run

agent = Agent(TestModel())


@agent.tool
@sevenbox_tool(source_kind=SourceKind.TOOL)
def lookup_order(ctx: RunContext, order_id: str) -> dict:
    return {"status": "shipped"}


with sevenbox_agent_run() as krun:
    agent.run_sync("Where is order 4471?")
    for w in krun.session.frontier().live:
        print(w.content)
```

Verified by running this exact snippet against the installed
`pydantic-ai` package (`TestModel` is PydanticAI's own official,
network-free test double — no fake-model code needed to try this
yourself).

Full example with parallel KYC + refund tool calls:
[`examples/refund_and_parallel_kyc_pydantic_ai.py`](../../examples/refund_and_parallel_kyc_pydantic_ai.py).
