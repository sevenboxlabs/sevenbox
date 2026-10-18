# LangGraph

```bash
pip install "warrant-core[langgraph]"
```

Two changes to an existing LangGraph app: wrap each `add_node` call,
wrap the `.invoke()` call site. Node bodies are untouched.

```python
from langgraph.graph import END, START, StateGraph
from warrant_core import SourceKind
from sevenbox_adapters.langgraph_adapter import sevenbox_node, sevenbox_run


def lookup_order(state):
    return {"order": {"status": "shipped"}}


g = StateGraph(dict)
g.add_node("lookup_order", sevenbox_node(lookup_order, source_kind=SourceKind.TOOL))
g.add_edge(START, "lookup_order")
g.add_edge("lookup_order", END)
compiled = g.compile()

with sevenbox_run() as krun:
    compiled.invoke({})

for w in krun.session.frontier().live:
    print(w.content)
```

Verified by running this exact snippet against the installed
`langgraph` package. `source_kind` defaults to `SourceKind.TOOL`
(effectful) — pass `source_kind=SourceKind.DERIVED` for a node you know
is pure.

**Known limitation:** this is a linear-chain model. If LangGraph runs
two nodes concurrently from the same state, both get chained onto
whichever "latest" Warrant happened to land first — not a faithful
causal graph of a genuinely branching run.

Full example with replay and compensation:
[`examples/refund_scenario_langgraph.py`](../../examples/refund_scenario_langgraph.py).
Branching/fan-out: [`examples/loan_review_scenario_langgraph_branching.py`](../../examples/loan_review_scenario_langgraph_branching.py).
