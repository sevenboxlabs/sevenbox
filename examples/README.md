# Examples

Have your own agent already? Start here, not at the top of `README.md`.

| If you have... | Look at |
|---|---|
| Nothing yet, want the fastest path | `sevenbox init` (see the main `README.md`'s Quickstart) |
| A LangGraph graph | `refund_scenario_langgraph.py` — two changes to an existing graph (`sevenbox_node()` around each node, `sevenbox_run()` around `invoke()`), verified against a real installed `langgraph` |
| An OpenAI Agents SDK agent | `refund_scenario_openai_agents.py` — two changes to an existing agent (`@sevenbox_tool` under each `@function_tool`, `sevenbox_agent_run()` around `Runner.run()`), verified against a real installed `openai-agents`, including a real tool-calling loop (offline, via a real `Model` implementation — no API key needed to run this file) |
| No framework, plain Python functions | `refund_scenario_sdk.py` — the full `sevenbox_sdk` decorator vocabulary (`@oracle`/`@tool`/`@derive`/`@irreversible`) |
| Interest in what's underneath the SDK | `refund_scenario_runtime.py` (rule-based scheduling), then `refund_scenario.py` (Warrants constructed by hand, no SDK at all) |
| A second, unrelated domain for comparison | `kyc_scenario.py` — same vocabulary, proves nothing here is refund-specific |

All six run the same underlying scenario at increasing levels of the
stack, each verified independently (`pytest tests/ -q`). None of them are
tutorials that stop working if you skip ahead — every file is a complete,
runnable script:

```bash
python3 examples/refund_scenario_langgraph.py
pip install -e ".[openai-agents]"   # optional, for refund_scenario_openai_agents.py
python3 examples/refund_scenario_openai_agents.py
```

For seven more reference domains (security incident, procurement,
deployment, customer support, document review, plus refund/KYC) with
real conflicts, redactions, branching, and compensation already wired
into the Operations Center, run `sevenbox ops` and open any of them from
"Open System..." — no code required to see one.
