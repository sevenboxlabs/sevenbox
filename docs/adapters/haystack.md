# Haystack Agents

```bash
pip install "warrant-core[haystack]"
```

Wrap the run in `sevenbox_haystack_run()`, pass
`hooks=krun.observer.hooks` when constructing the `Agent`. Structurally
different from every other adapter here: Haystack's hooks receive the
Agent's *entire* mutable `State`, not a typed per-call event — this
adapter figures out what changed by diffing message-list length before
and after.

```python
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from sevenbox_adapters.haystack_adapter import sevenbox_haystack_run

with sevenbox_haystack_run() as krun:
    krun.session.start("check order 4471")
    agent = Agent(chat_generator=your_generator, tools=[lookup_order], hooks=krun.observer.hooks)
    result = agent.run(messages=[ChatMessage.from_user("check order 4471")])
    for w in krun.session.frontier().live:
        print(w.content)
```

Verified against the real installed `haystack-ai` package — this is
the exact pattern
[`tests/test_haystack_adapter.py`](../../tests/test_haystack_adapter.py)
exercises (with a network-free `ChatGenerator` standing in for
`your_generator`).

**Genuinely positive finding, independently confirmed** (the same
property Strands has): when one sibling in a parallel batch fails, the
successful sibling's result is not lost — confirmed directly from the
framework's own docstring text. Real OS-thread parallelism
(`ThreadPoolExecutor`) is on by default, no opt-in required.
