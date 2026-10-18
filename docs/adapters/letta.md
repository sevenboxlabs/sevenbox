# Letta

```bash
pip install "warrant-core[letta]"
```

Structurally different from every other adapter here: Letta's real
boundary is an HTTP API (a running Letta server), not an in-process
loop, so this wraps `letta-client` (the official REST SDK) rather than
instrumenting a framework's own call chain.

```python
from letta_client import Letta
from sevenbox_adapters.letta_adapter import sevenbox_letta_run

client = Letta(base_url="http://localhost:8283")

with sevenbox_letta_run() as krun:
    krun.session.start("check order 4471")
    response = client.agents.messages.create(agent_id=your_agent_id, input="check order 4471")
    krun.observe(response)
    for w in krun.session.frontier().live:
        print(w.content)
```

Verified against the real installed `letta-client` package with a
mocked HTTP transport returning real, schema-accurate response bodies —
this is the exact pattern
[`tests/test_letta_adapter.py`](../../tests/test_letta_adapter.py)
exercises. **Weaker verification than the other adapters, stated
directly**: this confirms the adapter parses real response shapes
correctly, but Letta's own server-side dispatch code never runs in this
verification — there's no way to run it without a live Postgres-backed
server.

If you use raw `threading.Thread`s with a shared `SevenboxSession` (rather
than letting Letta's own client handle concurrency), remember Python's
own `contextvars` don't propagate to a new thread automatically —
`contextvars.copy_context().run(...)` is the fix, not something this
adapter can do for you.
