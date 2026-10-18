# CrewAI

```bash
pip install "warrant-core[crewai]"
```

One change to an existing crew: `krun.wrap(crew)` before
`crew.kickoff()`. Two layers of causal precision come for free —
Task-level (exact, read from CrewAI's own `Task.context` field) and,
opt-in via `@sevenbox_tool`, tool-call-level (approximate: CrewAI's hooks
fire per-call with no batch id).

```python
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from warrant_core import SourceKind
from sevenbox_adapters.crewai_adapter import sevenbox_agent_run, sevenbox_tool

@tool("check_credit_score")
@sevenbox_tool(source_kind=SourceKind.TOOL, name="check_credit_score")
def check_credit_score(applicant_id: str) -> str:
    return f"score:{applicant_id}:710"

credit_task = Task(description="Check the credit score", expected_output="...", agent=credit_agent)
summary_task = Task(description="Summarize the decision", expected_output="...",
                     agent=summary_agent, context=[credit_task])  # exact parent, native field
crew = Crew(agents=[credit_agent, summary_agent], tasks=[credit_task, summary_task],
            process=Process.sequential)

with sevenbox_agent_run() as krun:
    krun.wrap(crew)
    result = crew.kickoff()
    for w in krun.session.frontier().live:
        print(w.content)
```

Verified by running this pattern (with `credit_agent`/`summary_agent`
configured with a network-free scripted LLM) against the installed
`crewai` package.

**Real finding worth knowing:** CrewAI dispatches native tool calls via
a genuine `ThreadPoolExecutor` — true OS threads, not cooperative
concurrency — verified directly against the SDK's source, which also
contradicts that method's own docstring claiming otherwise.

Full example with replay and compensation:
[`examples/loan_review_crewai.py`](../../examples/loan_review_crewai.py).
