# Quickstart

No framework, no adapter, no third-party SDK — just Sevenbox's own
`sevenbox_sdk` decorators. This is the fastest way to see what Sevenbox
actually does.

## Install

```bash
git clone <this repo> && cd warrant-core
./setup.sh
```

`setup.sh` creates `venv/` with a supported Python (3.10-3.13 --
see below), installs Sevenbox plus the LangGraph extra used by the real-
framework example further down, and verifies the environment. It never
imports `warrant_core` or runs `sevenbox` itself, so it never touches the
terms gate described just below -- accepting the terms stays a real,
deliberate step you take yourself. Run `./setup.sh none` for a core-only
install with no framework extra, or `./setup.sh openai-agents` (etc.) to
pick a different one. Prefer to see every step yourself instead of a
script running them?

```bash
git clone <this repo> && cd warrant-core
python3.12 -m venv venv && source venv/bin/activate   # Windows: py -3.12 -m venv venv && venv\Scripts\activate
pip install -e .
```

Either way works identically -- `setup.sh` just automates the version
check that would otherwise fail with a raw pip error. **Python 3.10-3.13
only, never 3.14**: `crewai`, one of Sevenbox's optional framework
integrations, declares `Requires-Python <3.14` itself (deliberately, not
"not yet" -- see `crewAIInc/crewAI#5109`), and this project's own
`pyproject.toml` inherits that ceiling so the whole install fails fast
and clearly on 3.14 rather than only once you reach for that one extra.

The first time you `import warrant_core` (not at `pip install` time),
you'll be shown Sevenbox's evaluation terms and asked to press Enter to
accept (any other key cancels). In a script or CI, set
`SEVENBOX_ACCEPT_TERMS=1` instead of answering interactively.

## The whole idea, in five function calls

A real, running example — decorate four plain functions, run them
inside a session, and every call becomes a **Warrant**: a record of
what was claimed, on what causal grounds, that can be replayed later
without re-running anything.

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
    if not elig["eligible"]:
        return None
    return {"action": f"refund_action(4471)"}


with SevenboxSession() as session:
    ticket = session.axiom({"order_id": "4471"})
    intent = infer_intent(ticket)
    order = lookup_order(intent["order_id"])
    elig = eligibility(order)
    action = refund_action(elig)

    for w in session.frontier().live:
        print(w.content)
```

Run it: `python examples/refund_scenario_sdk.py` (this is that exact
file — nothing was trimmed for this quickstart).

```
Frontier after execution:
  - {'order_id': '4471'}
  - {'intent': 'refund_request', 'order_id': '4471'}
  - {'status': 'shipped'}
  - {'eligible': True}
  - {'action': 'refund_action(4471)'}
```

Five Warrants. You never constructed one directly, never wrote a
`parent_ids` list, and never named a `source_kind` by hand except via
the decorators at the top of the file — `@oracle`/`@tool`/`@derive`
each carve out one epistemic category (a model's belief, an external
system's answer, a pure computation), and `Traced` values (what these
functions return) carry their own causal parent forward automatically
through ordinary Python data flow.

## The other half: replay

```python
with SevenboxSession(replay_from=session.store) as replay_session:
    ticket2 = replay_session.axiom({"order_id": "4471"})
    intent2 = infer_intent(ticket2)
    order2 = lookup_order(intent2["order_id"])
    elig2 = eligibility(order2)
    action2 = refund_action(elig2)

assert action2.value == action.value
```

Same code, a different session. `lookup_order` and `infer_intent`
never actually ran the second time — their answers were replayed from
the store captured the first time, and the result is provably
identical.

## Seeing it against a real agent framework

The `sevenbox_sdk` example above never touches a framework on purpose, to
show the underlying idea in isolation. If you already build agents with
something like LangGraph, this is closer to what adopting Sevenbox actually
looks like: two changes to a real graph, nothing else.

```bash
pip install -e ".[langgraph]"   # already done if you ran ./setup.sh
python examples/refund_scenario_langgraph.py
```

That file is a complete, ordinary LangGraph app — `infer_intent`,
`lookup_order`, `eligibility`, and `refund_action` are plain node
functions, unmodified. The only changes are wrapping each `add_node(...)`
call in `sevenbox_node(...)` and wrapping the `invoke()` call site in
`sevenbox_run()`. Running it prints:

```
LangGraph's own result (unchanged): {'ticket': {'order_id': '4471'}, 'intent': {'intent': 'refund_request', 'order_id': '4471'}, 'order': {'status': 'shipped'}, 'eligible': True, 'action': 'refund_action(4471)'}

What Sevenbox captured, that plain LangGraph would not give you:
  - [axiom  ] {'ticket': {'order_id': '4471'}}
  - [oracle ] {'intent': {'intent': 'refund_request', 'order_id': '4471'}}
  - [tool   ] {'order': {'status': 'shipped'}}
  - [derived] {'eligible': True}
  - [derived] {'action': 'refund_action(4471)'}

Compensation obligations found: 1
  - {'action': 'refund_action(4471)'} needs compensation

Replaying the same graph from the captured store:
Replay reproduced LangGraph's own result exactly, with no real node bodies executed for the effectful nodes.
```

Four things happened that plain LangGraph doesn't give you, all from the
same two-line change: every node's output got an immutable, content-
addressed record with a causal parent (the `[oracle]`/`[tool]`/`[derived]`
tags — `warrant_core`'s own classification, not this script's); the
refund was correctly recognized as **irreversible** and left with an open
**compensation obligation** the moment new evidence contradicted the
belief it was based on; and the whole graph replayed from that record
with the real node bodies for `infer_intent`/`lookup_order` never
executing a second time — the replayed result is asserted equal to the
original, not just printed and eyeballed.

Same pattern exists for the other 13 integrated frameworks — see
[`docs/adapters/`](adapters/README.md) for a 10-15 line usage page each.

## Next

- **Already have an agent in one of 14 frameworks?** [`docs/adapters/`](adapters/README.md) —
  each page is a 10-15 line diff to wire Sevenbox into what you already have.
- **Want to see it running with a UI?** `sevenbox init` writes a real,
  runnable starter script; `sevenbox ops` opens a browser view over
  whatever store you point it at.
- **Want the API surface (`Warrant`, `WarrantStore`, the core
  operations)?** [`docs/api-reference.md`](api-reference.md).
- **Want the formal grounding (axioms, theorems, proofs)?** Not in
  this repo yet -- shared directly with teams once they're evaluating
  Sevenbox, not published in advance of that.
