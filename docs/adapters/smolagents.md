# Smolagents

```bash
pip install "warrant-core[smolagents]"
```

Wrap the run in `sevenbox_smolagents_run()`, iterate
`agent.run(task, stream=True)` yourself, and call `krun.observe(event)`
per event. `ToolOutput.output` is already the raw Python return value —
this is the only adapter in the set needing no `sevenbox_tool()` wrapper
around the tool function at all.

```python
from smolagents import ToolCallingAgent
from sevenbox_adapters.smolagents_adapter import sevenbox_smolagents_run

with sevenbox_smolagents_run() as krun:
    krun.session.start("check order 4471")
    agent = ToolCallingAgent(tools=[lookup_order], model=your_model)
    for event in agent.run("check order 4471", stream=True):
        krun.observe(event)
    for w in krun.session.frontier().live:
        print(w.content)
```

Verified against the real installed `smolagents` package — this is the
exact pattern
[`tests/test_smolagents_adapter.py`](../../tests/test_smolagents_adapter.py)
exercises (with a scripted, network-free `Model` standing in for
`your_model`).

**Two real limitations, adversarially verified:** (1) when one sibling
in a parallel batch fails, the framework itself discards the
successful sibling's result too — not a Sevenbox limitation, a Smolagents
one; (2) sessions shared across truly concurrent runs (real OS threads,
not just async) corrupt tracking. `CodeAgent` is out of scope entirely
— it never emits a per-tool-call event to observe.
