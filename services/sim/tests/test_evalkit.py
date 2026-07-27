from evalkit.checks import check, run_checks
from evalkit.harness import (
    EvalItem,
    load_items,
    orchestrator_runner,
    run_eval,
    single_call_runner,
)
from evalkit.report import aggregate, format_report
from ploshcha_sim.adapters.llm_fake import FakeLlm
from ploshcha_sim.adapters.router_profile import PresetEffort, single_model_router
from ploshcha_sim.adapters.tools_fake import FakeToolbox
from ploshcha_sim.agents.orchestrator import Orchestrator
from ploshcha_sim.domain.task import TaskResult


def _res(answer=None, scratch=None, accepted=False, degraded=False):
    return TaskResult(answer=answer, scratch=scratch or [], accepted=accepted, degraded=degraded)


def test_answer_contains_casefold():
    r = _res(answer="Результат: 309524.")
    assert check({"kind": "answer_contains", "value": "309524"}, r)
    assert not check({"kind": "answer_contains", "value": "999"}, r)


def test_used_tool_and_multi_hop():
    r = _res(scratch=[{"call": {"tool": "calc"}}, {"call": {"tool": "check_date"}}])
    assert check({"kind": "used_tool", "tool": "calc"}, r)
    assert check({"kind": "multi_hop", "n": 2}, r)
    assert not check({"kind": "multi_hop", "n": 3}, r)


def test_abstain_and_no_data_tool():
    r = _res(answer="Привіт!", scratch=[])
    assert check({"kind": "abstain"}, r)
    assert check({"kind": "no_data_tool"}, r)
    r2 = _res(answer="x", scratch=[{"call": {"tool": "lookup_fact"}}])
    assert not check({"kind": "abstain"}, r2)


def test_unknown_check_raises():
    try:
        check({"kind": "nope"}, _res())
        assert False
    except ValueError:
        pass


def test_run_checks_all_keys():
    r = _res(answer="600", scratch=[{"call": {"tool": "calc"}}])
    out = run_checks([{"kind": "used_tool", "tool": "calc"}, {"kind": "answer_contains", "value": "600"}], r)
    assert all(out.values())
    assert len(out) == 2


def test_run_eval_and_report_math():
    items = [
        EvalItem(id="a", category="x", task="t", checks=[{"kind": "answer_contains", "value": "ok"}]),
        EvalItem(id="b", category="y", task="t", checks=[{"kind": "answer_contains", "value": "ok"}]),
    ]
    good = lambda task, seed: _res(answer="ok")  # noqa: E731
    flaky = lambda task, seed: _res(answer="ok" if seed == 0 else "no")  # noqa: E731
    results = run_eval(items, {"good": good, "flaky": flaky}, seeds=[0, 1])
    reps = {r["condition"]: r for r in aggregate(results)}
    assert reps["good"]["success_rate"] == 1.0
    assert reps["good"]["pass_k"] == 1.0
    assert reps["flaky"]["success_rate"] == 0.5
    assert reps["flaky"]["pass_k"] == 0.0
    assert "condition" in format_report(results)


def test_load_starter_items():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "evalkit", "items", "starter.jsonl")
    items = load_items(path)
    assert len(items) >= 12
    assert {i.category for i in items} >= {"tool_needed", "multi_hop", "abstain", "grounding"}


def test_single_call_runner():
    llm = FakeLlm(["309524"])
    run = single_call_runner(llm)
    r = run("скільки 347*892", 0)
    assert r.answer == "309524"
    assert r.scratch == []


def test_orchestrator_runner_integration():
    def make_orch():
        llm = FakeLlm([
            '{"tool": "calc", "expr": "347*892"}',
            '{"tool": "final_answer", "text": "309524"}',
        ])
        return Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(), verifier=False)

    run = orchestrator_runner(make_orch)
    r = run("скільки 347*892", 0)
    assert check({"kind": "used_tool", "tool": "calc"}, r)
    assert check({"kind": "answer_contains", "value": "309524"}, r)
