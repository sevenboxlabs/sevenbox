# AWS Strands Agents

```bash
pip install "warrant-core[strands]"
```

Wrap the run in `sevenbox_strands_run()`, pass `hooks=[krun.observer]` when
constructing the `Agent`. `AfterToolCallEvent` bundles the call, its
result, and any exception into one event — the cleanest correlation of
any adapter in this set.

```python
from strands import Agent
from sevenbox_adapters.strands_adapter import sevenbox_strands_run

with sevenbox_strands_run() as krun:
    krun.session.start("check order 4471")
    agent = Agent(model=your_model, tools=[lookup_order], hooks=[krun.observer])
    result = await agent.invoke_async("check order 4471")
    for w in krun.session.frontier().live:
        print(w.content)
```

Verified against the real installed `strands-agents` package — this is
the exact pattern
[`tests/test_strands_adapter.py`](../../tests/test_strands_adapter.py)
exercises (with a scripted, network-free `Model` implementing the real
Bedrock Converse streaming protocol standing in for `your_model`).

**Genuinely positive finding, the first found across this whole
programme:** when one sibling in a parallel batch fails, the
successful sibling's result is *not* lost — each `AfterToolCallEvent`
reports its own status independently. Real OS-thread parallelism is on
by default, with no opt-in required.
