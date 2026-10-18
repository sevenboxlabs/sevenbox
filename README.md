# Sevenbox

Sevenbox is causal audit infrastructure for AI agent decisions. A Warrant
records why a decision was made: the specific prior claims it depended on,
and whether acting on it can be undone. Every call your agent makes
becomes one, automatically, the moment you wrap it. When a belief your
agent acted on turns out to be wrong, Sevenbox finds exactly what still
needs fixing — not by searching logs, by walking the causal chain.

This repo is the open-core layer: the SDK, 14 framework adapters, the
local Operations Center UI, and examples. It depends on `warrant-core`,
a separate, compiled engine package — not on PyPI, distributed as a
GitHub Release of this repo (see [Install](#install) and
[License](#license)).

## Install

Neither package is on PyPI yet. Both come from this repo.

**1. Install the compiled `warrant-core` engine.** Go to this repo's
[Releases page](https://github.com/sevenboxlabs/sevenbox/releases/tag/v0.1.0)
and download the wheel matching your OS and Python version (3.10-3.13),
then:

```bash
pip install warrant_core-0.1.0-<your-platform-and-python-tag>.whl
```

Wheels are built for macOS (arm64), Linux (manylinux x86_64), and Windows
(AMD64) — e.g. `warrant_core-0.1.0-cp312-cp312-macosx_11_0_arm64.whl` for
macOS on Python 3.12, or `...-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
for Linux, or `...-win_amd64.whl` for Windows.

**2. Install `sevenbox` itself**, directly from this repo:

```bash
pip install git+https://github.com/sevenboxlabs/sevenbox.git
```

Or, if you've already cloned it:

```bash
pip install .
```

pip resolves `sevenbox`'s `warrant-core>=0.1.0` requirement against the
wheel you installed in step 1 and won't try to reach PyPI for it.

The first time you `import warrant_core`, you'll be shown its evaluation
terms and asked to press Enter to accept. For CI or non-interactive use,
set `SEVENBOX_ACCEPT_TERMS=1` instead.

## Quickstart

```python
from sevenbox_sdk import SevenboxSession, derive, irreversible, oracle, tool

@oracle
def infer_intent(ticket):
    return {"intent": "refund_request", "order_id": ticket["order_id"]}

@tool
def lookup_order(order_id):
    return {"status": "shipped"}

@derive
def eligibility(order):
    return {"eligible": order["status"] == "shipped"}

@irreversible
@derive
def refund_action(elig):
    return {"action": "refund_action(4471)"} if elig["eligible"] else None

with SevenboxSession() as session:
    ticket = session.axiom({"order_id": "4471"})
    action = refund_action(eligibility(lookup_order(infer_intent(ticket)["order_id"])))
    for w in session.frontier().live:
        print(w.content)
```

See [`docs/quickstart.md`](docs/quickstart.md) for the replay example and
a real-framework (LangGraph) walkthrough.

## Supported frameworks

Validated against 14 real agent frameworks, plus the case of running no
framework at all (`sevenbox_sdk`'s own decorators, directly — no adapter
needed): LangGraph, OpenAI Agents SDK, PydanticAI, CrewAI, Deep Agents,
Microsoft Agent Framework, Mastra, OpenHands, LlamaIndex Agents,
Smolagents, Letta, Google ADK, AWS Strands Agents, Haystack Agents. A
10-15 line usage page for each lives in [`docs/adapters/`](docs/adapters/).

## Validation

**206** tests in this repo, covering the SDK and all 14 adapters, pass
against the real installed framework packages — no network, no API keys.
The full suite, including `warrant_core`/`warrant_runtime` themselves
(private, not in this repo), is **394** tests.

## See it running

```bash
sevenbox init    # writes a real, runnable starter script
python my_agent.py
sevenbox ops     # opens a local Operations Center over whatever store you point it at
```

More at [sevenbox.dev](https://sevenbox.dev).

## License

The SDK, adapters, and demo in this repo are licensed under
[Apache 2.0](LICENSE). The `warrant-core` engine is distributed
separately as a compiled binary, under its own evaluation terms
(`TERMS.md`, included with that package) — not open source, not yet
licensed for redistribution.

## Contact

julian@sevenbox.dev
