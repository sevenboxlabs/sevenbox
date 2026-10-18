"""
A two-task CrewAI crew (built against the real `crewai` package),
wrapped with sevenbox_adapters.crewai_adapter. Same "install in an
afternoon" comparison the other three framework examples make: one
change -- `krun.wrap(crew)` before `crew.kickoff()`.

Demonstrates BOTH layers this adapter provides:
  - Layer 1 (exact): two Tasks, the second reading the first's output
    via CrewAI's own native `Task.context` field -- parent_ids comes
    directly from that, zero inference.
  - Layer 2 (approximate, opt-in via @sevenbox_tool): a tool call inside a
    task, captured via CrewAI's before/after_tool_call hooks.

Uses a custom BaseLLM subclass (crewai's own extension point for
offline/deterministic testing) for a real, network-free run -- no API
key needed.

Run: PYTHONPATH=. python3 examples/loan_review_crewai.py
"""
from crewai import Agent, BaseLLM, Crew, Process, Task
from crewai.tools import tool

from warrant_core import SourceKind, find_transitive_obligations, supersede
from warrant_core import EpistemicClass, ReversibilityClass
from warrant_core import EffectRecord
from sevenbox_adapters.crewai_adapter import sevenbox_agent_run, sevenbox_tool


class _ScriptedLLM(BaseLLM):
    """Deterministic, network-free: a fixed sequence of responses, one
    per call()."""
    responses: list = []
    _step: int = 0

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
              from_task=None, from_agent=None, response_model=None):
        response = self.responses[self._step]
        self._step += 1
        return response

    async def acall(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def supports_stop_words(self) -> bool:
        return False

    def supports_function_calling(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192


@tool("check_credit_score")
@sevenbox_tool(source_kind=SourceKind.TOOL, name="check_credit_score")
def check_credit_score(applicant_id: str) -> str:
    """Look up an applicant's credit score."""
    return f"score:{applicant_id}:710"


def build_crew():
    credit_llm = _ScriptedLLM(model="fake", responses=[
        [{"id": "call_1", "function": {"name": "check_credit_score", "arguments": {"applicant_id": "APP-1001"}}}],
        "Credit score is 710 -- good standing.",
    ])
    summary_llm = _ScriptedLLM(model="fake", responses=[
        "Based on the credit check, this applicant qualifies for a standard-rate loan.",
    ])

    credit_agent = Agent(role="Credit Analyst", goal="check credit", backstory="...",
                          llm=credit_llm, tools=[check_credit_score])
    summary_agent = Agent(role="Loan Officer", goal="summarize", backstory="...", llm=summary_llm)

    credit_task = Task(
        description="Check the credit score for applicant APP-1001",
        expected_output="the credit score and standing",
        agent=credit_agent,
    )
    summary_task = Task(
        description="Summarize the loan decision based on the credit check",
        expected_output="a one-sentence loan decision summary",
        agent=summary_agent,
        context=[credit_task],  # Layer 1: explicit, exact dependency
    )

    crew = Crew(agents=[credit_agent, summary_agent], tasks=[credit_task, summary_task],
                process=Process.sequential)
    return crew, credit_task, summary_task


def main():
    crew, credit_task, summary_task = build_crew()

    with sevenbox_agent_run() as krun:
        krun.session.start("Review loan application APP-1001")
        krun.wrap(crew)
        result = crew.kickoff()

    print("CrewAI's own result (unchanged):", result.raw)

    print(f"\nWarrant graph ({len(krun.session.store)} warrants):")
    for w in krun.session.store.all():
        parents = [p[:8] for p in w.parent_ids] or ["(none)"]
        content = w.content if isinstance(w.content, str) else str(w.content)[:70]
        print(f"  id={w.id[:8]}  [{w.source_kind.value:6s}]  parents={parents}  {content}")

    # Layer 1: exact parent from Task.context, not inferred
    summary_w_id = krun.session._task_id_to_warrant_id[summary_task.id]
    credit_w_id = krun.session._task_id_to_warrant_id[credit_task.id]
    summary_w = next(w for w in krun.session.store.all() if w.id == summary_w_id)
    assert summary_w.parent_ids == (credit_w_id,)
    print(f"\nLayer 1 confirmed: summary task's sole parent is the credit task's "
          f"Warrant, read directly from Task.context -- zero inference.")

    # New evidence arrives -- same compensation pattern as every other example
    credit_tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    supersede(
        krun.session.store, target_id=credit_tool_w.id,
        content="score:APP-1001:590 (corrected -- prior score was for a different applicant)",
        source_kind=SourceKind.TOOL, epistemic=EpistemicClass.VERIFIED,
        reversibility=ReversibilityClass.PURE,
        effect_record=EffectRecord(request="bureau re-check", response="score corrected"),
    )
    obligations = find_transitive_obligations(krun.session.store)
    print(f"\nCompensation obligations found after correction: {len(obligations)}")

    print("\nReplaying Layer 2 (tool calls) from the captured store:")
    crew2, credit_task2, summary_task2 = build_crew()
    with sevenbox_agent_run() as replay_run:
        replay_run.session.start("Review loan application APP-1001")
        replay_run.wrap(crew2, replay_from=krun.session.store)
        crew2.kickoff()
    replay_tool_w = next(w for w in replay_run.session.store.all() if w.source_kind == SourceKind.TOOL)
    print(f"Replayed tool Warrant content: {replay_tool_w.content!r}")
    print("(the real check_credit_score body did not run again for this call --")
    print(" see sevenbox_adapters/crewai_adapter.py's module docstring for the exact,")
    print(" honest scope: Layer 2 tool calls replay for real, Layer 1 Task/LLM")
    print(" calls do not -- there is no native CrewAI cache for those.)")


if __name__ == "__main__":
    main()
