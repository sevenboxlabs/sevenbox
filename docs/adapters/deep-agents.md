# Deep Agents

```bash
pip install "warrant-core[deep-agents]"
```

One middleware installs into `create_deep_agent(middleware=[...])` and
instruments **every** tool the agent has automatically — the built-in
filesystem tools, shell execution, and sub-agent delegation (`task`)
included. No per-tool decoration needed, unlike every other adapter in
this list.

```python
from deepagents import create_deep_agent
from langchain_core.tools import tool
from sevenbox_adapters.deep_agents_adapter import sevenbox_deep_agent_run

@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"status for {order_id}: shipped"

with sevenbox_deep_agent_run() as krun:
    agent = create_deep_agent(model=your_model, tools=[lookup_order], middleware=[krun.middleware])
    agent.invoke({"messages": [{"role": "user", "content": "check order 4471"}]})
    for w in krun.session.frontier().live:
        print(w.content)
```

Verified by running this exact pattern against the installed
`deepagents`/`langchain` packages (`your_model` stands in for a real or
test `BaseChatModel` — a complete network-free version using
`FakeMessagesListChatModel` is in
[`tests/test_deep_agents_adapter.py`](../../tests/test_deep_agents_adapter.py)).

Sub-agent delegation gets its own causal root at the delegation
decision — a sub-agent's internal work doesn't pollute the parent
agent's frontier, and vice versa.
