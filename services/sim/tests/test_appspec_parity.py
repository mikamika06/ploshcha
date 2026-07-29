import json
from pathlib import Path

import pytest

from evalkit.conditions import CONDITIONS, grid, runner_for
from evalkit.cost import cost_of_results, cost_usd, rate_bounds, usd_per_success
from evalkit.harness import EvalResult, load_items
from ploshcha_sim.domain.spec import AppSpec
from rule_llm import RuleLlm

FIXTURE = Path(__file__).parent / "fixtures_appspec_parity.json"
ITEMS = Path(__file__).parents[1] / "evalkit" / "items" / "starter.jsonl"
WANT = ("calc-01", "date-01", "hop-01", "abstain-01")
SEEDS = (1, 2)
DROP = {"latency_ms"}

STUCK = {
    "stuck-mamay@8": AppSpec().with_(routing="mamay", max_steps=8),
    "stuck-mamay+rec@8": AppSpec().with_(routing="mamay", max_steps=8, recovery=True),
    "stuck-hetero+rec@8": AppSpec().with_(max_steps=8, recovery=True),
    "stuck-hetero-nov@8": AppSpec().with_(max_steps=8, verifier=False),
}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in DROP}
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    return obj


def _items():
    return [i for i in load_items(str(ITEMS)) if i.id in WANT]


def _frozen_names(expected):
    """Паритет стосується умов, які існували на момент заморозки; нові умови в фікстурі відсутні."""
    return sorted({k.split("|", 1)[0] for k in expected})


def _observed(names):
    lapa, mamay = RuleLlm("lapa"), RuleLlm("mamay")
    stuck = RuleLlm("stuck", repeat=True)
    runners = {}
    for name in names:
        if name in STUCK:
            runners[name] = runner_for(STUCK[name], lapa=stuck, mamay=stuck)
        else:
            runners.update(grid([name], lapa=lapa, mamay=mamay))
    out = {}
    for name, runner in runners.items():
        for item in _items():
            for seed in SEEDS:
                out[f"{name}|{item.id}|{seed}"] = _clean(runner(item.task, seed).model_dump())
    return out


@pytest.fixture(scope="module")
def expected():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_covers_the_grid_it_froze(expected):
    names = _frozen_names(expected)
    assert len(expected) == len(names) * len(WANT) * len(SEEDS)
    gone = [n for n in names if n not in CONDITIONS and n not in STUCK]
    assert not gone, f"заморожені умови зникли з решітки: {gone}"


def test_every_condition_reproduces_the_frozen_run(expected):
    observed = _observed(_frozen_names(expected))
    assert sorted(observed) == sorted(expected), "склад умов змінився"
    diff = [k for k in expected if observed[k] != expected[k]]
    assert not diff, f"специфікація змінила поведінку в {len(diff)} клітинках: {diff[:5]}"


def test_the_fixture_actually_discriminates(expected):
    """Паритет нічого не варт, якщо фікстура вироджена."""
    steps = {v["steps"] for v in expected.values()}
    calls = {len(v["scratch"]) for v in expected.values()}
    assert len(steps) >= 3, "усі умови дають однакову кількість кроків"
    assert calls != {0}, "жодна умова не викликала інструмент"
    assert any(v["incidents"] for v in expected.values()), "драбина відновлення не задіяна"
    assert any(v["partial"] for v in expected.values()), "часткова відповідь не задіяна"


def test_recovery_is_visible_in_the_fixture(expected):
    off = expected["stuck-mamay@8|date-01|1"]
    on = expected["stuck-mamay+rec@8|date-01|1"]
    assert off["steps"] < on["steps"] and not off["incidents"] and on["incidents"]
    assert on["partial"] and not off["partial"]


def test_spec_sha_is_stable_and_order_independent():
    a = AppSpec(routing="mamay", max_steps=8)
    b = AppSpec(max_steps=8, routing="mamay")
    assert a.sha256 == b.sha256
    assert a.sha256 != AppSpec(routing="mamay", max_steps=5).sha256


def test_spec_is_frozen():
    with pytest.raises(Exception):
        AppSpec().max_steps = 9


def test_named_conditions_have_distinct_specs():
    shas = {name: spec.sha256 for name, spec in CONDITIONS.items()}
    dupes = [n for n, s in shas.items() if list(shas.values()).count(s) > 1]
    assert not dupes, f"різні назви з однаковою специфікацією: {dupes}"


def test_gate_direct_model_is_an_axis():
    """K7 заміряв 0.923 проти 0.808 між цими двома — отже це справжня вісь, не дубль."""
    assert (CONDITIONS["gate-notools-mamay"].sha256
            != CONDITIONS["gate-notools-lapa"].sha256)


def test_hetero_cost_is_an_interval_not_a_guess():
    lo, hi = rate_bounds("hetero")
    assert lo < hi, "гетеро без tokens_by_tier не має точкової ціни"
    assert rate_bounds("lapa")[0] == rate_bounds("lapa")[1]
    assert cost_usd(1_000_000, "lapa") == (0.10, 0.10)


def test_cost_ranks_lapa_below_mamay():
    rows = [EvalResult(item_id="x", category="c", condition=c, seed=1, success=True,
                       checks={}, tokens=1000)
            for c in ("single-lapa", "single-mamay")]
    cost = cost_of_results(rows, CONDITIONS)
    assert cost["single-lapa"][1] < cost["single-mamay"][0]
    per = usd_per_success(rows, CONDITIONS)
    assert per["single-lapa"][0] < per["single-mamay"][0]
