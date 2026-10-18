"""
sevenbox_demo.init_scaffold -- the content `sevenbox init` writes out.

The template below is a real, runnable script using nothing but the
public sevenbox_sdk/warrant_core surface -- the exact pattern verified
end-to-end against a real `sevenbox ops` server while scoping this feature
(instrument three toy steps, write to a local SqliteWarrantStore, open it
in the Operations Center). Not a special onboarding API: this is what an
external integrator's first file looks like today, generated instead of
hand-typed.
"""
from __future__ import annotations

SCAFFOLD_TEMPLATE = '''"""
Your first Sevenbox-instrumented agent.

Replace the three functions below with your real agent's own steps. Each
decorator tells Sevenbox what kind of decision it's watching:

    @oracle  -- a model/LLM call (non-deterministic)
    @tool    -- a call to something external (an API, a database, ...)
    @derive  -- a pure function of what you already have

(Two more exist for less common cases: @human for a step that takes human
input, and @irreversible/@compensable as modifiers marking a call's
effect as unable to be undone, or undoable via a real compensating
action -- see sevenbox_sdk.decorators for the full vocabulary. Most agents
only need oracle/tool/derive to get started.)

Every call you make through a decorated function becomes a Warrant --
Sevenbox's record of what happened and why, with the arguments/return value
you already use unchanged. No new control flow, no new data shapes.

Run this file, then:
    sevenbox ops
and use "Open System..." -> "Your own agent" on sevenbox_store.db to see
what you just ran.
"""
from sevenbox_sdk import SevenboxSession, oracle, tool, derive
from warrant_core import SqliteWarrantStore


@oracle
def classify_request(message):
    # Replace with your real model/LLM call.
    return {"intent": "example_intent"}


@tool
def call_external_system(intent):
    # Replace with your real API/database/tool call.
    return {"result": "example_result"}


@derive
def decide_next_step(result):
    # Replace with your real decision logic -- a pure function, no I/O.
    return {"decision": "example_decision"}


def run():
    with SqliteWarrantStore("sevenbox_store.db") as store, SevenboxSession(store=store) as session:
        message = session.axiom({"text": "hello from my agent"})
        intent = classify_request(message)
        result = call_external_system(intent)
        decide_next_step(result)
    print('Wrote sevenbox_store.db -- run `sevenbox ops` and open it via "Your own agent".')


if __name__ == "__main__":
    run()
'''
