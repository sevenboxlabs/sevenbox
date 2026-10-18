# OpenHands

```bash
pip install "warrant-core[openhands]"
```

Wrap your `Conversation` run in `sevenbox_openhands_run()` and add
`krun.callback` to the conversation's `callbacks` list. Batches are
grouped by `ActionEvent.llm_response_id` — an exact signal, not a
timing heuristic.

```python
from openhands.sdk import Agent, Conversation
from sevenbox_adapters.openhands_adapter import sevenbox_openhands_run

with sevenbox_openhands_run() as krun:
    krun.session.start("check order 4471")
    conversation = Conversation(agent=your_agent, callbacks=[krun.callback])
    conversation.send_message("check order 4471")
    conversation.run()
    for w in krun.session.frontier().live:
        print(w.content)
```

Verified against the real installed `openhands-sdk` package — the
pattern above is the exact wiring exercised in
[`tests/test_openhands_adapter.py`](../../tests/test_openhands_adapter.py),
which drives a real `Agent`/`Conversation` through real tool
definitions with only the underlying `litellm` call faked (no network,
no API key). OpenHands tools need their own `Action`/`Observation`/
`ToolExecutor` classes regardless of Sevenbox — the test file has a
complete, minimal pair (`lookup_order`, `check_watchlist`) to copy from.

**Real, documented limitation:** no replay-without-execution support —
an architectural limitation of this integration, not merely
unimplemented.
