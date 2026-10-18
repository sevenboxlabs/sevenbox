# Microsoft Agent Framework

```bash
pip install "warrant-core[agent-framework]"
```

> Installs cleanly on its own, but conflicts (`resolution-too-deep`)
> with this project's other 12 framework extras if installed into the
> same environment. Use a dedicated venv for this one.

`krun.instrument(client)` attaches directly to the client's own
`function_middleware` list. **Do not** use
`Agent(middleware=[...])` — a real bug in the framework itself silently
discards a `FunctionMiddleware` passed that way (verified by reading
`ChatMiddlewareLayer`'s source).

```python
from agent_framework import Agent
from sevenbox_adapters.agent_framework_adapter import sevenbox_agent_framework_run

with sevenbox_agent_framework_run() as krun:
    client = YourRealChatClient(...)
    krun.instrument(client)          # NOT Agent(middleware=[...]) -- see above
    agent = Agent(client=client, name="tester", tools=[lookup_order, check_watchlist])
    result = await agent.run("check order 4471")
    print(krun.session.frontier().live)
```

`krun.instrument(...)` must be called once per client used inside the
`with` block.

Verified in its own isolated venv (11/11 tests) against the real
installed `agent-framework`/`agent-framework-orchestrations` packages
while writing this documentation. Full network-free reference:
[`tests/test_agent_framework_adapter.py`](../../tests/test_agent_framework_adapter.py).
