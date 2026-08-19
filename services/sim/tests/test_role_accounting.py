import pytest

from evalkit.cost import (
    ROLE_OF_STAGE,
    format_roles,
    role_attribution_gap,
    role_cost,
    role_lane_by_condition,
    role_of,
    roles_by_condition,
)
from evalkit.harness import EvalResult
from ploshcha_sim.adapters.router_profile import LAPA_KINDS, MAMAY_KINDS
from ploshcha_sim.domain.task import Budget
from ploshcha_sim.ports.router import STEP_KINDS


def test_budget_attributes_stage_and_lane_together():
    b = Budget()
    b.spend(100, "lapa", 80, stage="select")
    b.spend(50, "mamay", 30, stage="decide")
    b.spend_aux(20, "mamay", 10, stage="judge")

    assert b.tokens_by_stage == {"select": 100, "decide": 50, "judge": 20}
    assert b.prompt_by_stage == {"select": 80, "decide": 30, "judge": 10}
    assert b.tokens_by_stage_lane == {"select|lapa": 100, "decide|mamay": 50, "judge|mamay": 20}
    assert b.prompt_by_stage_lane == {"select|lapa": 80, "decide|mamay": 30, "judge|mamay": 10}


def test_sum_over_stages_equals_sum_over_lanes_equals_total():
    b = Budget()
    b.spend(100, "lapa", 80, stage="select")
    b.spend(50, "mamay", 30, stage="decide")
    b.spend_aux(20, "mamay", 10, stage="judge")

    total = b.tokens_used + b.aux_tokens
    assert sum(b.tokens_by_stage.values()) == total
    assert sum(b.tokens_by_lane.values()) == total
    assert sum(b.tokens_by_stage_lane.values()) == total
    assert sum(b.prompt_by_stage.values()) == sum(b.prompt_by_lane.values())


def test_spend_without_stage_keeps_old_behaviour():
    b = Budget()
    b.spend(42, "lapa", 40)
    assert b.tokens_by_lane == {"lapa": 42}
    assert b.tokens_by_stage == {"unknown": 42}
    assert b.tokens_by_stage_lane == {"unknown|lapa": 42}


def test_every_step_kind_has_a_role():
    for kind in STEP_KINDS:
        assert role_of(kind) != "other", f"kind {kind} без ролі"


def test_role_mapping_agrees_with_router_profile():
    for kind in LAPA_KINDS:
        assert role_of(kind) == "executor", kind
    for kind in MAMAY_KINDS:
        assert role_of(kind) in {"orchestrator", "verifier"}, kind
    assert role_of("judge") == "verifier"


def test_unknown_stage_falls_into_other_and_is_not_lost():
    b = Budget()
    b.spend(7, "lapa", 5, stage="дивна-стадія")
    assert "дивна-стадія" not in ROLE_OF_STAGE
    r = _result("c", b)
    assert roles_by_condition([r]) == {"c": {"other": 7}}
    assert sum(roles_by_condition([r])["c"].values()) == 7


def _result(condition: str, budget: Budget) -> EvalResult:
    return EvalResult(
        item_id="i", category="cat", condition=condition, seed=0, success=True, checks={},
        tokens=budget.tokens_used, aux_tokens=budget.aux_tokens,
        tokens_by_lane=dict(budget.tokens_by_lane), prompt_by_lane=dict(budget.prompt_by_lane),
        tokens_by_stage=dict(budget.tokens_by_stage), prompt_by_stage=dict(budget.prompt_by_stage),
        tokens_by_stage_lane=dict(budget.tokens_by_stage_lane),
        prompt_by_stage_lane=dict(budget.prompt_by_stage_lane),
    )


@pytest.fixture
def hetero_results():
    b = Budget()
    b.spend(1000, "lapa", 850, stage="select")
    b.spend(1000, "lapa", 800, stage="parse")
    b.spend(300, "mamay", 170, stage="decide")
    b.spend(200, "mamay", 100, stage="synthesize")
    b.spend_aux(100, "mamay", 60, stage="judge")
    return [_result("hetero", b)]


def test_roles_fold_stages(hetero_results):
    roles = roles_by_condition(hetero_results)["hetero"]
    assert roles == {"executor": 2000, "orchestrator": 500, "verifier": 100}


def test_role_lane_cross_is_exact_not_distributed(hetero_results):
    cells = role_lane_by_condition(hetero_results)["hetero"]
    assert cells == {("executor", "lapa"): 2000,
                     ("orchestrator", "mamay"): 500,
                     ("verifier", "mamay"): 100}


def test_executor_on_mamay_is_reported_as_mamay():
    b = Budget()
    b.spend(500, "mamay", 400, stage="select")
    cells = role_lane_by_condition([_result("all-mamay", b)])["all-mamay"]
    assert cells == {("executor", "mamay"): 500}


def test_role_cost_matches_lane_cost_total(hetero_results):
    from evalkit.cost import lane_cost

    by_role = sum(role_cost(hetero_results)["hetero"].values())
    r = hetero_results[0]
    by_lane = lane_cost(r.tokens_by_lane, r.prompt_by_lane)
    assert by_role == pytest.approx(by_lane, rel=1e-9)


def test_no_attribution_gap_when_stages_are_threaded(hetero_results):
    assert role_attribution_gap(hetero_results) == {}


def test_attribution_gap_is_reported_when_stage_missing():
    r = EvalResult(item_id="i", category="c", condition="broken", seed=0, success=True,
                   checks={}, tokens=100, aux_tokens=0, tokens_by_stage={})
    assert role_attribution_gap([r]) == {"broken": 100}
    assert "НЕ ВІДНЕСЕНО" in format_roles([r] )


def test_format_roles_mentions_every_role(hetero_results):
    out = format_roles(hetero_results)
    for role in ("orchestrator", "executor", "verifier"):
        assert role in out
    assert "lapa" in out and "mamay" in out
