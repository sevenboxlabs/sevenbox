"""
Real-crewai integration tests for sevenbox_adapters.crewai_adapter.

Uses a custom BaseLLM subclass (crewai's own extension point for
offline/deterministic testing -- no mock of crewai itself) to drive
real Agent/Task/Crew objects through crewai.kickoff(), including
genuinely parallel native tool calls dispatched by CrewAI's own real
ThreadPoolExecutor (crewai.agents.crew_agent_executor), not a
simulation of it.
"""
import threading

import pytest

crewai = pytest.importorskip("crewai")
from crewai import Agent, BaseLLM, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

from warrant_core import EpistemicClass, ReversibilityClass, SourceKind, compute_frontier  # noqa: E402
from sevenbox_adapters.crewai_adapter import sevenbox_agent_run, sevenbox_tool  # noqa: E402


class _FakeLLM(BaseLLM):
    """Deterministic, network-free LLM: a scripted sequence of responses,
    one per call() invocation. Native tool calling requires
    supports_function_calling() -- verified empirically to be required
    (see this file's assumption-pinning test); its absence silently
    routes crewai through a different, non-native execution loop that
    never dispatches structured tool calls at all."""

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


def _tool_call(call_id: str, name: str, **args) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": args}}


# --- Assumption-pinning test: CrewAI's own documented-vs-actual gap ---

def test_native_tool_calling_requires_supports_function_calling_capability():
    """Pins the empirically-discovered requirement this whole adapter's
    tests depend on: an LLM must implement supports_function_calling()
    -> True, or CrewAI never dispatches structured tool calls at all --
    the raw tool-call-shaped list crewai's own executor would otherwise
    have parsed instead becomes the task's raw output directly, and
    since TaskOutput.raw is typed str, CrewAI's own pydantic validation
    rejects it with a hard ValidationError (not a graceful text
    fallback, as might be assumed without checking). If a future crewai
    version changes this, this test fails here, not as a mysterious
    empty-tool-call-list bug in the adapter's own tests."""
    @tool("noop_tool")
    def noop_tool(x: str) -> str:
        """A tool that never actually gets called in this test."""
        return x

    class _NoFunctionCallingLLM(_FakeLLM):
        def supports_function_calling(self) -> bool:
            return False

    llm = _NoFunctionCallingLLM(model="fake", responses=[
        [_tool_call("call_1", "noop_tool", x="unused")],  # a real tool-call-shaped list
    ])
    agent = Agent(role="R", goal="G", backstory="B", llm=llm, tools=[noop_tool])
    task = Task(description="d", expected_output="o", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)

    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="raw"):
        crew.kickoff()


# --- Layer 1: Task-level, exact -----------------------------------------

def _fake_task_agent(*responses: str) -> Agent:
    return Agent(role="R", goal="G", backstory="B", llm=_FakeLLM(model="fake", responses=list(responses)))


def test_task_called_outside_sevenbox_agent_run_is_fine_but_uncaptured():
    """Unlike the other three adapters, CrewAI's Layer 1 is opt-in via
    krun.wrap(crew), not a decorator raising RuntimeError -- a crew run
    without wrap() just isn't captured, since there is no per-call
    injection point analogous to @sevenbox_tool/@sevenbox_node to raise from."""
    agent = _fake_task_agent("done")
    task = Task(description="d", expected_output="o", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    assert result.raw == "done"  # plain crewai, unaffected either way


def test_not_specified_context_implicitly_chains_every_prior_task():
    """Layer 1's core claim: Task.context left at its NOT_SPECIFIED
    default means the task implicitly depends on EVERY previously
    completed task -- verified against crewai's own Crew._get_context,
    not assumed."""
    agent = _fake_task_agent("out1", "out2", "out3")
    task1 = Task(description="t1", expected_output="o", agent=agent)
    task2 = Task(description="t2", expected_output="o", agent=agent)  # NOT_SPECIFIED
    task3 = Task(description="t3", expected_output="o", agent=agent)  # NOT_SPECIFIED
    crew = Crew(agents=[agent], tasks=[task1, task2, task3], process=Process.sequential)

    with sevenbox_agent_run() as krun:
        krun.session.start("kickoff")
        krun.wrap(crew)
        crew.kickoff()

    w1 = krun.session._task_id_to_warrant_id[task1.id]
    w2 = krun.session._task_id_to_warrant_id[task2.id]
    w3 = krun.session._task_id_to_warrant_id[task3.id]
    warrants = {w.id: w for w in krun.session.store.all()}

    assert warrants[w2].parent_ids == (w1,)              # task2: only task1 completed so far
    assert set(warrants[w3].parent_ids) == {w1, w2}       # task3: both priors, implicit


def test_explicit_context_list_restricts_to_exactly_those_tasks():
    """A task explicitly declaring context=[task1] must NOT also pick up
    task2's output, even though task2 completed earlier -- explicit
    context overrides the implicit accumulate-everything default."""
    agent = _fake_task_agent("out1", "out2", "out3")
    task1 = Task(description="t1", expected_output="o", agent=agent)
    task2 = Task(description="t2", expected_output="o", agent=agent)
    task3 = Task(description="t3", expected_output="o", agent=agent, context=[task1])  # explicit, excludes task2

    crew = Crew(agents=[agent], tasks=[task1, task2, task3], process=Process.sequential)

    with sevenbox_agent_run() as krun:
        krun.session.start("kickoff")
        krun.wrap(crew)
        crew.kickoff()

    w1 = krun.session._task_id_to_warrant_id[task1.id]
    w3 = krun.session._task_id_to_warrant_id[task3.id]
    warrants = {w.id: w for w in krun.session.store.all()}
    assert warrants[w3].parent_ids == (w1,)  # NOT task2, despite running before task3


def test_explicit_empty_context_gets_only_the_axiom():
    """context=[] (explicit, falsy) is CrewAI's own "no context at all"
    signal -- Crew._get_context returns "" for this case. Layer 1
    matches: parent is the axiom only, not the accumulated frontier."""
    agent = _fake_task_agent("out1", "out2")
    task1 = Task(description="t1", expected_output="o", agent=agent)
    task2 = Task(description="t2", expected_output="o", agent=agent, context=[])

    crew = Crew(agents=[agent], tasks=[task1, task2], process=Process.sequential)

    with sevenbox_agent_run() as krun:
        krun.session.start("kickoff")
        krun.wrap(crew)
        crew.kickoff()

    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    w2 = krun.session._task_id_to_warrant_id[task2.id]
    warrants = {w.id: w for w in krun.session.store.all()}
    assert warrants[w2].parent_ids == (axiom.id,)


def test_context_on_a_not_yet_completed_task_raises_rather_than_guessing():
    """A task graph where context references a task the session hasn't
    recorded yet (e.g. built out of execution order) must raise, not
    silently produce an incomplete or wrong parent set."""
    from warrant_core import WarrantStore
    from sevenbox_adapters.crewai_adapter import CrewSession

    agent = _fake_task_agent("out1")
    task1 = Task(description="t1", expected_output="o", agent=agent)
    task2 = Task(description="t2", expected_output="o", agent=agent)

    session = CrewSession(store=WarrantStore())
    session.start("kickoff")
    task2.context = [task1]  # task1 was never recorded

    class _FakeOutput:
        raw = "out"

    with pytest.raises(ValueError, match="have not completed yet"):
        session.record_task_output(task=task2, output=_FakeOutput())


# --- Layer 2: tool-call, honestly approximate ----------------------------

def test_single_sequential_tool_call_chains_correctly():
    @tool("lookup_order")
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="lookup_order")
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return f"shipped:{order_id}"

    llm = _FakeLLM(model="fake", responses=[
        [_tool_call("call_1", "lookup_order", order_id="4471")],
        "Order 4471 has shipped.",
    ])
    agent = Agent(role="Support", goal="G", backstory="B", llm=llm, tools=[lookup_order])
    task = Task(description="Look up order 4471", expected_output="status", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)

    with sevenbox_agent_run() as krun:
        krun.session.start("Look up order 4471")
        krun.wrap(crew)
        result = crew.kickoff()

    assert result.raw == "Order 4471 has shipped."
    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    tool_w = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL)
    assert tool_w.parent_ids == (axiom.id,)
    assert tool_w.effect_record.source_name == "lookup_order"


def test_genuine_parallel_batch_produces_the_documented_false_dependency():
    """THE key Layer-2 finding, asserted explicitly rather than left
    implicit: CrewAI's real ThreadPoolExecutor dispatches both calls in
    this batch concurrently, and because no public hook/event exposes a
    batch/turn identifier (verified during design -- see module
    docstring), the second call to be RECORDED chains onto the first,
    not onto their true shared parent. This is the same class of
    imprecision sevenbox_adapters.langgraph_adapter.GraphSession documents
    for its own ambient-frontier case -- pinned here so it is never
    silently "fixed" into a false claim of precision without a design
    change (e.g. wrapping tools to negotiate a shared batch id
    themselves, which CrewAI's public API does not support today)."""
    @tool("check_watchlist")
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="check_watchlist")
    def check_watchlist(applicant_id: str) -> str:
        """Check the watchlist."""
        return f"watchlist_clear:{applicant_id}"

    @tool("verify_document")
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="verify_document")
    def verify_document(applicant_id: str) -> str:
        """Verify a document."""
        return f"document_verified:{applicant_id}"

    llm = _FakeLLM(model="fake", responses=[
        [
            _tool_call("call_1", "check_watchlist", applicant_id="A-991"),
            _tool_call("call_2", "verify_document", applicant_id="A-991"),
        ],
        "KYC complete.",
    ])
    agent = Agent(role="KYC", goal="G", backstory="B", llm=llm, tools=[check_watchlist, verify_document])
    task = Task(description="Run KYC on A-991", expected_output="status", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)

    with sevenbox_agent_run() as krun:
        krun.session.start("Run KYC on A-991")
        krun.wrap(crew)
        crew.kickoff()

    axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
    tool_warrants = [w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL]
    assert len(tool_warrants) == 2

    parent_sets = [set(w.parent_ids) for w in tool_warrants]
    one_cites_axiom = {axiom.id} in parent_sets
    other_cites_sibling = any(
        w.parent_ids and w.parent_ids[0] not in (axiom.id,) and
        w.parent_ids[0] in {sib.id for sib in tool_warrants if sib.id != w.id}
        for w in tool_warrants
    )
    assert one_cites_axiom
    assert other_cites_sibling, (
        "expected the documented false-dependency chain (one tool call's "
        "parent is its sibling, not the shared axiom) -- if this now "
        "fails, CrewAI may have started exposing batch information "
        "publicly, which would be real news, not a broken test"
    )


def test_concurrent_tool_calls_do_not_corrupt_the_store_under_heavier_load():
    """A real thread-safety stress test through CrewAI's own
    ThreadPoolExecutor (max_workers = min(8, N)), not a synthetic
    thread test of warrant_core alone -- 16 genuinely parallel native
    tool calls per trial, 30 trials (480 total concurrent inserts),
    with sys.setswitchinterval lowered to force
    more frequent thread-switching than default scheduling would
    produce. Confirms the store ends up with exactly the right warrant
    count and no duplicate or corrupted descendant-index entries every
    single time -- real evidence this is safe under genuine OS-thread
    concurrency driven by an actual framework, not just synthetic
    Python-threading pressure against warrant_core directly."""
    import sys
    n_tools = 16
    n_trials = 30
    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        def make_tool(i):
            @tool(f"tool_{i}")
            @sevenbox_tool(source_kind=SourceKind.TOOL, name=f"tool_{i}")
            def _t(x: str) -> str:
                """A stress-test tool."""
                return f"result_{i}:{x}"
            _t.__name__ = f"tool_{i}"
            return _t

        tools = [make_tool(i) for i in range(n_tools)]

        for trial in range(n_trials):
            llm = _FakeLLM(model="fake", responses=[
                [_tool_call(f"call_{i}", f"tool_{i}", x="v") for i in range(n_tools)],
                "all done",
            ])
            agent = Agent(role="R", goal="G", backstory="B", llm=llm, tools=tools)
            task = Task(description=f"trial {trial}", expected_output="o", agent=agent)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)

            with sevenbox_agent_run() as krun:
                krun.session.start(f"trial {trial}")
                krun.wrap(crew)
                crew.kickoff()

            tool_warrants = [w for w in krun.session.store.all() if w.source_kind == SourceKind.TOOL]
            assert len(tool_warrants) == n_tools, f"trial {trial}: expected {n_tools} tool warrants, got {len(tool_warrants)}"
            axiom = next(w for w in krun.session.store.all() if w.source_kind == SourceKind.AXIOM)
            descendant_ids = krun.session.store.descendants(axiom.id)
            assert len(descendant_ids) == len(set(descendant_ids)), f"trial {trial}: duplicate descendant ids"
    finally:
        sys.setswitchinterval(original_interval)


# --- Replay ---------------------------------------------------------------

def test_replay_from_skips_real_tool_execution_for_layer_2():
    """The real mechanism, now implemented: wrap(crew, replay_from=store)
    pre-populates every agent's native tool-result cache from the
    reference store's captured effect_records, matching CrewAI's own
    (tool_name, json.dumps(args)) cache key exactly. Verified via a
    call_log the real tool body appends to -- it must NOT grow during
    the replay run."""
    call_log = []

    @tool("lookup_order")
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="lookup_order")
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        call_log.append(order_id)
        return f"shipped:{order_id}"

    def build_crew():
        llm = _FakeLLM(model="fake", responses=[
            [_tool_call("call_1", "lookup_order", order_id="4471")],
            "Order 4471 has shipped.",
        ])
        agent = Agent(role="Support", goal="G", backstory="B", llm=llm, tools=[lookup_order])
        task = Task(description="Look up order 4471", expected_output="status", agent=agent)
        return Crew(agents=[agent], tasks=[task], process=Process.sequential)

    with sevenbox_agent_run() as live_run:
        live_run.session.start("Look up order 4471")
        crew1 = build_crew()
        live_run.wrap(crew1)
        crew1.kickoff()

    assert call_log == ["4471"]

    with sevenbox_agent_run() as replay_run:
        replay_run.session.start("Look up order 4471")
        crew2 = build_crew()
        replay_run.wrap(crew2, replay_from=live_run.session.store)
        crew2.kickoff()

    assert call_log == ["4471"], "the real tool body must NOT run again during replay"
    assert {w.id for w in replay_run.session.store.all()} == {w.id for w in live_run.session.store.all()}


def test_replay_from_does_not_extend_to_layer_1_task_llm_calls():
    """Pins the exact, honest boundary documented in the module
    docstring: replay_from gives real non-execution replay for Layer 2
    (tool calls) ONLY. There is no native CrewAI cache for a Task's own
    LLM call, so the LLM is invoked again during a "replay" run -- this
    test uses an llm_call_log (distinct from the tool's call_log) to
    prove the LLM really was re-invoked, not skipped."""
    llm_call_log = []
    tool_call_log = []

    @tool("lookup_order")
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="lookup_order")
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        tool_call_log.append(order_id)
        return f"shipped:{order_id}"

    class _CountingLLM(_FakeLLM):
        def call(self, *args, **kwargs):
            llm_call_log.append(1)
            return super().call(*args, **kwargs)

    def build_crew():
        llm = _CountingLLM(model="fake", responses=[
            [_tool_call("call_1", "lookup_order", order_id="4471")],
            "Order 4471 has shipped.",
        ])
        agent = Agent(role="Support", goal="G", backstory="B", llm=llm, tools=[lookup_order])
        task = Task(description="Look up order 4471", expected_output="status", agent=agent)
        return Crew(agents=[agent], tasks=[task], process=Process.sequential)

    with sevenbox_agent_run() as live_run:
        live_run.session.start("Look up order 4471")
        crew1 = build_crew()
        live_run.wrap(crew1)
        crew1.kickoff()

    live_llm_calls = len(llm_call_log)
    assert live_llm_calls > 0
    assert tool_call_log == ["4471"]

    with sevenbox_agent_run() as replay_run:
        replay_run.session.start("Look up order 4471")
        crew2 = build_crew()
        replay_run.wrap(crew2, replay_from=live_run.session.store)
        crew2.kickoff()

    # Layer 2 (tool): did NOT run again.
    assert tool_call_log == ["4471"]
    # Layer 1 (LLM/Task): DID run again -- exactly as many times as before,
    # proving the LLM is invoked fresh each run, not skipped.
    assert len(llm_call_log) == live_llm_calls * 2


def test_replay_from_falls_back_to_real_execution_on_a_cache_miss():
    """Another honest, verified difference from the other three
    adapters: CrewAI's native cache falls through to REAL execution
    silently on a miss, rather than raising loudly the way
    ReplayEffectExecutor.invoke() does elsewhere in this project. This
    test uses a reference store missing the one tool call the replay run
    actually needs, and confirms the tool body DOES run (no exception,
    no silent wrong answer -- graceful degradation to live execution)."""
    from warrant_core import WarrantStore

    call_log = []

    @tool("lookup_order")
    @sevenbox_tool(source_kind=SourceKind.TOOL, name="lookup_order")
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        call_log.append(order_id)
        return f"shipped:{order_id}"

    llm = _FakeLLM(model="fake", responses=[
        [_tool_call("call_1", "lookup_order", order_id="4471")],
        "Order 4471 has shipped.",
    ])
    agent = Agent(role="Support", goal="G", backstory="B", llm=llm, tools=[lookup_order])
    task = Task(description="Look up order 4471", expected_output="status", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)

    empty_reference_store = WarrantStore()  # no cached entries at all

    with sevenbox_agent_run() as run:
        run.session.start("Look up order 4471")
        run.wrap(crew, replay_from=empty_reference_store)
        result = crew.kickoff()

    assert call_log == ["4471"]  # ran for real -- no matching cache entry existed
    assert result.raw == "Order 4471 has shipped."


def test_an_explicitly_passed_empty_store_is_not_silently_discarded():
    from warrant_core import WarrantStore
    from sevenbox_adapters.crewai_adapter import CrewSession

    my_store = WarrantStore()
    session = CrewSession(store=my_store)
    assert session.store is my_store


# --- chromadb guard (docs/security-review.md section 1) -------------------
#
# CVE-2026-45829: an unpatched, CVSS-9.3 pre-auth RCE in chromadb's HTTP
# server. Not this project's dependency -- pulled in transitively by
# crewai's Knowledge feature, which imports crewai.rag.chromadb
# unconditionally. These tests exist so that a future session enabling
# Knowledge on a real Crew hits a loud, explicit RuntimeError naming the
# CVE, instead of silently reintroducing the vulnerable surface with no
# record anyone ever knew it was there. They also pin the opposite fact
# (memory=True does NOT trigger this, verified against the installed
# crewai's own default storage) so this guard can't silently drift into
# being over-broad either.

def test_wrap_rejects_crew_level_knowledge_sources_without_acknowledgement():
    from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

    agent = _fake_task_agent("done")
    task = Task(description="d", expected_output="o", agent=agent)
    crew = Crew(
        agents=[agent], tasks=[task], process=Process.sequential,
        knowledge_sources=[StringKnowledgeSource(content="some reference text")],
    )
    with sevenbox_agent_run() as krun:
        krun.session.start("kickoff")
        with pytest.raises(RuntimeError, match="CVE-2026-45829"):
            krun.wrap(crew)


def test_wrap_rejects_agent_level_knowledge_without_acknowledgement():
    from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

    agent = _fake_task_agent("done")
    agent.knowledge_sources = [StringKnowledgeSource(content="some reference text")]
    task = Task(description="d", expected_output="o", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    with sevenbox_agent_run() as krun:
        krun.session.start("kickoff")
        with pytest.raises(RuntimeError, match="CVE-2026-45829"):
            krun.wrap(crew)


def test_wrap_allows_knowledge_sources_once_explicitly_acknowledged():
    from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

    agent = _fake_task_agent("done")
    task = Task(description="d", expected_output="o", agent=agent)
    crew = Crew(
        agents=[agent], tasks=[task], process=Process.sequential,
        knowledge_sources=[StringKnowledgeSource(content="some reference text")],
    )
    with sevenbox_agent_run() as krun:
        krun.session.start("kickoff")
        krun.wrap(crew, acknowledge_chromadb_risk=True)  # must not raise


def test_wrap_does_not_reject_plain_memory_true():
    """Pins the other half of the finding: memory=True alone must NOT
    trip this guard -- crewai's own default memory storage is lancedb,
    not chromadb (verified in _reject_if_chromadb_backed_features_enabled's
    docstring). A guard that fired on this would be a false positive, not
    caution -- this test protects against the check silently drifting
    into that."""
    agent = _fake_task_agent("done")
    task = Task(description="d", expected_output="o", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, memory=True)
    with sevenbox_agent_run() as krun:
        krun.session.start("kickoff")
        krun.wrap(crew)  # must not raise
