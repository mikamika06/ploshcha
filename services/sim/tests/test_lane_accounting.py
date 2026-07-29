from evalkit.cost import lane_share
from evalkit.harness import EvalItem, orchestrator_runner, run_eval
from ploshcha_sim.adapters import FakeLlm, FakeToolbox, PresetEffort, profile_router
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.domain.task import Budget

FINAL = '{"tool": "final_answer", "text": "1918"}'


def _orch(router, **kw):
    return Orchestrator(router, PresetEffort(), FakeToolbox(), verifier=False, **kw)


def test_budget_copies_do_not_share_the_lane_dict():
    """Знайдено живим прогоном: поверхнева копія ділила словник між усіма прогонами."""
    template = Budget(max_steps=4)
    first = template.model_copy(deep=True)
    first.spend(100, "lapa")
    second = template.model_copy(deep=True)
    second.spend(50, "mamay")
    assert first.tokens_by_lane == {"lapa": 100}
    assert second.tokens_by_lane == {"mamay": 50}
    assert template.tokens_by_lane == {}


def test_the_shallow_copy_really_was_the_bug():
    template = Budget(max_steps=4)
    shallow = template.model_copy()
    shallow.spend(100, "lapa")
    assert template.tokens_by_lane == {"lapa": 100}, "поверхнева копія ділить словник — саме це й було"


def test_two_runs_through_one_runner_are_independent():
    router = single_model_router(FakeLlm([FINAL, FINAL], strict=True), lane="mamay")
    runner = orchestrator_runner(lambda: _orch(router), budget=Budget(max_steps=4))
    first = runner("задача 1", 1)
    second = runner("задача 2", 1)
    assert first.tokens_by_lane == second.tokens_by_lane, "другий прогін не має нести токени першого"
    assert sum(first.tokens_by_lane.values()) == first.tokens + first.aux_tokens


def test_hetero_router_attributes_both_lanes():
    lapa, mamay = FakeLlm([FINAL] * 4), FakeLlm([FINAL] * 4)
    router = profile_router(lapa, mamay)
    assert router.lane("select") == "lapa" and router.lane("decide") == "mamay"
    assert single_model_router(lapa).lane("select") == "unknown"


def test_lane_totals_match_the_run_totals():
    router = profile_router(FakeLlm([FINAL] * 4), FakeLlm([FINAL] * 4))
    result = orchestrator_runner(lambda: _orch(router), budget=Budget(max_steps=4))("t", 1)
    assert sum(result.tokens_by_lane.values()) == result.tokens + result.aux_tokens


def test_eval_result_carries_the_breakdown_and_share():
    router = single_model_router(FakeLlm([FINAL] * 8), lane="lapa")
    runner = orchestrator_runner(lambda: _orch(router), budget=Budget(max_steps=4))
    items = [EvalItem(id="a", category="x", task="t",
                      checks=[{"kind": "answer_contains", "value": "1918"}])]
    rows = run_eval(items, {"cond": runner}, [1])
    assert rows[0].tokens_by_lane and "lapa" in rows[0].tokens_by_lane
    assert lane_share(rows)["cond"] == {"lapa": 1.0}
