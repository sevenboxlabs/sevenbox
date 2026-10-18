# LlamaIndex Agents

```bash
pip install "warrant-core[llamaindex]"
```

Wrap the run in `sevenbox_llamaindex_run()`, consume
`handler.stream_events()` yourself, and call `krun.observe(event)` on
each one. `AgentOutput.tool_calls` gives the whole batch already typed,
before dispatch — no correlation guesswork.

```python
from llama_index.core.agent.workflow import FunctionAgent
from sevenbox_adapters.llamaindex_adapter import sevenbox_llamaindex_run

with sevenbox_llamaindex_run() as krun:
    krun.session.start("check order 4471")
    agent = FunctionAgent(tools=[lookup_order], llm=your_llm)
    handler = agent.run("check order 4471")
    async for event in handler.stream_events():
        krun.observe(event)
    result = await handler
    for w in krun.session.frontier().live:
        print(w.content)
```

Verified against the real installed `llama-index-core` package — this
is the exact pattern
[`tests/test_llamaindex_adapter.py`](../../tests/test_llamaindex_adapter.py)
exercises (with a network-free `FunctionCallingLLM` subclass standing
in for `your_llm`).

**Real limitation, adversarially verified:** don't share one session
across two independent, concurrently-running agent turns — event
consumption here is single-threaded by construction, and overlapping
turns corrupt each other's tracking. One session per run, or consume
several runs strictly sequentially.
