# sevenbox-mastra-bridge

The TypeScript/Node-side half of Sevenbox's [Mastra](https://mastra.ai)
integration. Mastra is TypeScript-only (no Python SDK), so this
integration is a **log importer**, not a live in-process adapter like
every other framework Sevenbox supports — see
[`sevenbox_adapters/mastra_adapter.py`](../sevenbox_adapters/mastra_adapter.py)'s
module docstring for the full architecture decision and why.

## How it works

1. This package's `sevenboxStepLogger.mjs` observes a real Mastra `Agent` run
   via its native `onStepFinish` hook and writes a JSONL log of raw facts
   (tool name, arguments, result, step grouping) — no Warrant ids, no
   hashing, nothing calculus-shaped. It doesn't know what Sevenbox is.
2. The Python side (`sevenbox_adapters.mastra_adapter.import_mastra_log()`)
   reads that log afterward and constructs the real Warrants.

## Usage

Two additions to an existing Mastra agent:

```js
import { createSevenboxStepLogger } from "sevenbox-mastra-bridge/sevenboxStepLogger.mjs";

const sevenboxLog = createSevenboxStepLogger({ write: (line) => myLogStream.write(line + "\n") });
sevenboxLog.recordRunStart(userInput);
const result = await agent.generate(userInput, { onStepFinish: sevenboxLog.onStepFinish });
sevenboxLog.recordRunFinish(result.text);
```

Then, in Python:

```python
from sevenbox_adapters.mastra_adapter import import_mastra_log

result = import_mastra_log("path/to/log.jsonl")
print(result.store.all())
print(result.frontier)  # ids to parent a downstream Decision Warrant on
```

## Examples

`examples/kyc_scenario_parallel.mjs` and `kyc_scenario_sequential.mjs`
are real, network-free, runnable end-to-end scenarios (a hand-rolled
scripted model, no API key) — running them regenerates the exact fixture
logs `tests/fixtures/mastra_kyc_*_log.jsonl` that
`tests/test_mastra_adapter.py` reads.

```bash
cd mastra_bridge
npm install
node examples/kyc_scenario_parallel.mjs        # prints the JSONL log to stdout
node examples/kyc_scenario_parallel.mjs out.jsonl   # or writes it to a file
```

## Requirements

Node.js (verified against 24.15.0) and `@mastra/core@^1.57.0`. The
Python side (`sevenbox_adapters/mastra_adapter.py`) has no Node/npm
dependency at all — it only ever reads a JSONL file, so
`pytest tests/ -q` never needs Node installed.
