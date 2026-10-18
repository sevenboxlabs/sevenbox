# Mastra

No Python extra to install — Mastra is TypeScript-only (`@mastra/core`
on npm), so this adapter is a **log importer**, not a live in-process
adapter like the other thirteen.

Two pieces: a small TS logger (`mastra_bridge/sevenboxStepLogger.mjs`) hooks
into a real Mastra `Agent`'s `onStepFinish` and writes a JSONL log of
raw facts (tool name, arguments, result, step grouping) — no Warrant
ids, no hashes, nothing calculus-shaped computed on the TS side. Python
reads that log afterward and constructs the real Warrants.

```python
from sevenbox_adapters.mastra_adapter import import_mastra_log

# path, open text file, or any iterable of JSON-line strings
result = import_mastra_log("mastra_run_log.jsonl")

for warrant_id in result.frontier:
    print(result.store.get(warrant_id).content)
```

Verified by running this exact snippet against a real captured Mastra
log (`tests/fixtures/mastra_kyc_sequential_log.jsonl`).

**Real limitation, by design, not oversight:** this is not a live
adapter — there's no way for Sevenbox to influence a Mastra run while it's
happening (no cached-response short-circuiting mid-run), because
Warrant construction only happens after the fact, in Python, once.
