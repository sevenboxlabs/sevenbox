# Google ADK

```bash
pip install "warrant-core[google-adk]"
```

Wrap the run in `sevenbox_google_adk_run()`, iterate `runner.run(...)`
yourself, and call `krun.observe(event)` per event. Parallel tool
calls arrive already batched inside one `Event`.

```python
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types
from sevenbox_adapters.google_adk_adapter import sevenbox_google_adk_run

agent = LlmAgent(name="support_agent", model=your_llm, tools=[lookup_order])
session_service = InMemorySessionService()
session = session_service.create_session_sync(app_name="app", user_id="u1")
runner = Runner(app_name="app", agent=agent, session_service=session_service)

with sevenbox_google_adk_run() as krun:
    krun.session.start("check order 4471")
    for event in runner.run(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="check order 4471")]),
    ):
        krun.observe(event)
    for w in krun.session.frontier().live:
        print(w.content)
```

Verified against the real installed `google-adk` package — this is the
exact pattern
[`tests/test_google_adk_adapter.py`](../../tests/test_google_adk_adapter.py)
exercises (with a fake `BaseLlm` standing in for `your_llm` — real
`Runner`/`LlmAgent` orchestration code runs in-process, not just a
mocked transport).

**Counterintuitive, verified finding:** ADK's "parallel" tool calls
don't actually run on real threads in the normal (non-voice) flow —
they're dispatched via `asyncio.gather` plus a direct blocking call,
serialized in practice. `RunConfig.tool_thread_pool_config` looks like
it changes this; it doesn't, outside the LIVE/voice path.
