# Framework adapters

Fourteen adapters, one per agent framework Sevenbox has been integrated
against. Each page here is a short, practical "how do I use this"
guide — install command, the two or three lines that change in your
existing code, and what you get back. None of these pages cover *why*
each adapter is built the way it is, what real bugs were found
integrating it, or how its concurrency model was verified — that
detail lives in `docs/framework-integration-log.md` (internal, not
needed to use any of this).

Every snippet on these pages is a verified excerpt of a real pattern —
either run directly against the installed package while writing this
documentation, or drawn word-for-word from that adapter's own module
docstring and cross-checked against its real, passing test file
(`tests/test_<name>_adapter.py`). Where a framework needs real
model/tool scaffolding to run at all (most of them), the page shows
the Sevenbox-specific wiring and points to the test file for a complete,
runnable, network-free reference.

| Framework | Install extra | Page |
|---|---|---|
| LangGraph | `pip install "warrant-core[langgraph]"` | [langgraph.md](langgraph.md) |
| OpenAI Agents SDK | `pip install "warrant-core[openai-agents]"` | [openai-agents.md](openai-agents.md) |
| PydanticAI | `pip install "warrant-core[pydantic-ai]"` | [pydantic-ai.md](pydantic-ai.md) |
| CrewAI | `pip install "warrant-core[crewai]"` | [crewai.md](crewai.md) |
| Deep Agents | `pip install "warrant-core[deep-agents]"` | [deep-agents.md](deep-agents.md) |
| Microsoft Agent Framework | `pip install "warrant-core[agent-framework]"` | [agent-framework.md](agent-framework.md) |
| Mastra | — (log importer, no Python extra; see page) | [mastra.md](mastra.md) |
| OpenHands | `pip install "warrant-core[openhands]"` | [openhands.md](openhands.md) |
| LlamaIndex Agents | `pip install "warrant-core[llamaindex]"` | [llamaindex.md](llamaindex.md) |
| Smolagents | `pip install "warrant-core[smolagents]"` | [smolagents.md](smolagents.md) |
| Letta | `pip install "warrant-core[letta]"` | [letta.md](letta.md) |
| Google ADK | `pip install "warrant-core[google-adk]"` | [google-adk.md](google-adk.md) |
| AWS Strands Agents | `pip install "warrant-core[strands]"` | [strands.md](strands.md) |
| Haystack Agents | `pip install "warrant-core[haystack]"` | [haystack.md](haystack.md) |

No framework at all? See the [quickstart](../quickstart.md) — `sevenbox_sdk`'s
own decorators need no adapter and no extra install.
