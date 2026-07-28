from ploshcha_sim.adapters import FakeLlm, FakeToolbox, PresetEffort, profile_router, single_model_router
from ploshcha_sim.adapters.planner_skeleton import MAX_GATHER, SkeletonPlanner
from ploshcha_sim.adapters.tools_fake import DEFAULT_TOOLS, FakeToolbox as Box, Tool
from ploshcha_sim.agents.orchestrator import LinearPlanner, Orchestrator
from ploshcha_sim.domain.task import Budget, TaskPlan, TaskStep

DUP = '{"tool": "lookup_fact", "entity": "Іван Мазепа"}'
FINAL = '{"tool": "final_answer", "text": "Гетьман"}'


def test_plan_progress_and_advance():
    plan = TaskPlan(steps=[TaskStep(id="a", goal="x"), TaskStep(id="b", goal="y")])
    assert plan.progress() == (0, 2)
    assert plan.current().id == "a"
    plan.advance()
    assert plan.progress() == (1, 2) and plan.current().id == "b"
    plan.advance()
    assert plan.current() is None and plan.progress() == (2, 2)


def test_default_planner_has_no_plan():
    assert LinearPlanner().plan("t", FakeToolbox()) is None
    assert LinearPlanner().replan("t", None) is None


def test_skeleton_plan_is_derived_from_tools_not_task_text():
    plan_a = SkeletonPlanner().plan("будь-який текст", FakeToolbox())
    plan_b = SkeletonPlanner().plan("зовсім інший текст", FakeToolbox())
    assert plan_a.model_dump() == plan_b.model_dump()
    assert [s.kind for s in plan_a.steps] == ["select", "decide"]


def test_skeleton_hints_the_only_data_tool():
    single = Box(tools=[t for t in DEFAULT_TOOLS if t.name in ("lookup_fact", "final_answer")])
    plan = SkeletonPlanner().plan("t", single)
    assert plan.steps[0].tool_hint == "lookup_fact"
    assert SkeletonPlanner().plan("t", FakeToolbox()).steps[0].tool_hint is None


def test_skeleton_without_data_tools_plans_only_the_answer():
    only_final = Box(tools=[t for t in DEFAULT_TOOLS if t.name == "final_answer"])
    plan = SkeletonPlanner().plan("t", only_final)
    assert [s.kind for s in plan.steps] == ["decide"]


def test_next_kind_follows_the_plan_cursor():
    planner = SkeletonPlanner()
    plan = planner.plan("t", FakeToolbox())

    class S:
        pass

    state = S()
    state.plan = plan
    assert planner.next_kind(state) == "select"
    plan.advance()
    assert planner.next_kind(state) == "decide"
    plan.advance()
    assert planner.next_kind(state) == "decide"


def test_replan_widens_gather_and_keeps_done_steps():
    planner = SkeletonPlanner()

    class S:
        pass

    state = S()
    state.plan = planner.plan("t", FakeToolbox())
    state.task = "t"
    state.plan.advance()
    fresh = planner.replan("t", state)
    assert [s.kind for s in fresh.steps] == ["select", "select", "decide"]
    assert fresh.steps[0].done and not fresh.steps[1].done


def test_replan_stops_widening_at_the_cap():
    planner = SkeletonPlanner(gather=MAX_GATHER)

    class S:
        pass

    state = S()
    state.plan = planner.plan("t", FakeToolbox())
    assert planner.replan("t", state) is None


def test_replan_needs_an_existing_plan():
    class S:
        pass

    state = S()
    state.plan = None
    assert SkeletonPlanner().replan("t", state) is None


def test_orchestrator_routes_answer_step_to_the_stronger_model():
    weak = FakeLlm([DUP], model="lapa", strict=True)
    strong = FakeLlm([FINAL], model="mamay", strict=True)
    orch = Orchestrator(profile_router(weak, strong), PresetEffort(), FakeToolbox(),
                        planner=SkeletonPlanner(), verifier=False)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.answer == "Гетьман"
    assert len(weak.calls) == 1, "крок здобуття — на дешевій моделі"
    assert len(strong.calls) == 1, "крок відповіді — на сильнішій"


def test_plan_advances_only_on_successful_tool_call():
    bad = '{"tool": "calc", "expr": "import os"}'
    llm = FakeLlm([bad, DUP, FINAL], model="m", strict=True)
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                        planner=SkeletonPlanner(), verifier=False, recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=6))
    assert r.answer == "Гетьман"
    assert r.incidents == ["tool_error"]


def test_plan_adds_the_replan_rung_before_giving_up():
    llm = FakeLlm([DUP] * 6, model="m", strict=True)
    with_plan = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                             planner=SkeletonPlanner(), verifier=False, recovery=True)
    r = with_plan.run("t", seed=1, budget=Budget(max_steps=8))

    llm2 = FakeLlm([DUP] * 6, model="m", strict=True)
    no_plan = Orchestrator(single_model_router(llm2), PresetEffort(), FakeToolbox(),
                           verifier=False, recovery=True)
    r2 = no_plan.run("t", seed=1, budget=Budget(max_steps=8))

    assert r2.incidents == ["dup_call"] * 3, "без плану: nudge, nudge, partial"
    assert len(r.incidents) == 4, "з планом додається щабель replan перед здачею"
    assert r.partial and r2.partial
