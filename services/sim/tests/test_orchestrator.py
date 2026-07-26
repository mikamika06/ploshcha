import json

import pytest

from ploshcha_sim.adapters import FakeLlm, FakeToolbox, InMemoryTrace, PresetEffort, single_model_router
from ploshcha_sim.domain.task import Budget
from ploshcha_sim.agents import LinearPlanner, Orchestrator


def tc(tool, **args):
    return json.dumps({"tool": tool, **args})


def orch(script, verifier=True, trace=None):
    router = single_model_router(FakeLlm(script, model="m"))
    return Orchestrator(router, PresetEffort(), FakeToolbox(), verifier=verifier, trace=trace)


def test_linear_planner_always_select():
    assert LinearPlanner().next_kind(None) == "select"


def test_tool_then_final_then_verify_accept():
    o = orch([tc("lookup_fact", entity="Шевченко"), tc("final_answer", text="готово"),
              json.dumps({"accepted": True, "reason": "ok"})])
    r = o.run("питання", seed=1)
    assert r.answer == "готово" and r.accepted is True and not r.degraded
    assert len(r.scratch) == 1 and r.scratch[0]["call"]["tool"] == "lookup_fact"


def test_multistep_chain_accumulates_scratch():
    o = orch([tc("lookup_fact", entity="Крути"), tc("check_date", year=1918, event="Битва під Крутами"),
              tc("final_answer", text="1918"), json.dumps({"accepted": True, "reason": "ok"})])
    r = o.run("питання", seed=1)
    assert len(r.scratch) == 2 and r.answer == "1918"


def test_verifier_reject_marks_degraded():
    o = orch([tc("final_answer", text="хибне"), json.dumps({"accepted": False, "reason": "анахронізм"})])
    r = o.run("питання")
    assert r.accepted is False and r.degraded is True and r.verdict_reason == "анахронізм"


def test_verifier_off_accepts_without_judge():
    o = orch([tc("final_answer", text="готово")], verifier=False)
    r = o.run("питання")
    assert r.accepted is True and not r.degraded


def test_budget_stop_without_final_is_degraded():
    o = orch([tc("lookup_fact", entity=f"a{i}") for i in range(10)], verifier=False)
    r = o.run("питання", budget=Budget(max_steps=3))
    assert r.answer is None and r.degraded is True and r.steps == 3


def test_parse_fail_breaks_and_degrades():
    o = orch(["не JSON зовсім"], verifier=False)
    r = o.run("питання")
    assert r.degraded is True and r.answer is None


def test_unknown_tool_breaks():
    o = orch([tc("delete_all", x=1)], verifier=False)
    r = o.run("питання")
    assert r.degraded is True


def test_determinism_same_seed_same_result():
    a = orch([tc("lookup_fact", entity="X"), tc("final_answer", text="R"),
              json.dumps({"accepted": True, "reason": "ok"})]).run("q", seed=7)
    b = orch([tc("lookup_fact", entity="X"), tc("final_answer", text="R"),
              json.dumps({"accepted": True, "reason": "ok"})]).run("q", seed=7)
    assert a.model_dump() == b.model_dump()


def test_trace_records_each_step_with_seed():
    trace = InMemoryTrace()
    o = orch([tc("lookup_fact", entity="X"), tc("final_answer", text="R"),
              json.dumps({"accepted": True, "reason": "ok"})], trace=trace)
    o.run("q", seed=5)
    stages = [r.stage for r in trace.records]
    assert stages == ["select", "select", "judge"]
    assert all(r.seed == 5 for r in trace.records)


def test_seed_reaches_llm():
    llm = FakeLlm([tc("final_answer", text="R")], model="m")
    router = single_model_router(llm)
    Orchestrator(router, PresetEffort(), FakeToolbox(), verifier=False).run("q", seed=9)
    assert llm.calls[0]["seed"] == 9


def test_result_reports_step_count():
    o = orch([tc("lookup_fact", entity="X"), tc("final_answer", text="R")], verifier=False)
    r = o.run("q")
    assert r.steps == 2


def test_repeated_call_breaks_early():
    o = orch([tc("lookup_fact", entity="X")] * 6, verifier=False)
    r = o.run("q", budget=Budget(max_steps=10))
    assert r.degraded is True and r.steps == 2 and len(r.scratch) == 1
