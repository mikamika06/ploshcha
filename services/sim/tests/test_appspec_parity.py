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


def _project(observed: dict, template: dict) -> dict:
    """Порівнюємо лише те, що існувало на момент заморозки.

    Інваріант паритету — «нічого зі старого не змінилось», а не «дамп побайтово той самий»: інакше
    будь-яке ДОДАВАННЯ поля (ADR-0004 дозволяє лише додавати) ламало б фікстуру, і її довелось би
    перезнімати — тобто втратити властивість «знято ДО рефакторингу». Зникнення поля тут дає KeyError.
    """
    return {k: _project(observed[k], v) if isinstance(v, dict)
            else _project_list(observed[k], v) if isinstance(v, list)
            else observed[k]
            for k, v in template.items()}


def _project_list(observed: list, template: list) -> list:
    """Списки теж проєктуються поелементно: інакше додане поле ВСЕРЕДИНІ елемента (напр. у `scratch`)
    ламає паритет, хоч ADR-0004 саме додавання й дозволяє."""
    if len(observed) != len(template):
        return observed
    return [_project(o, t) if isinstance(t, dict) else o for o, t in zip(observed, template, strict=True)]


SANCTIONED = {
    # Поле: чому його зміна дозволена. Фікстура лишається знятою ДО рефакторингу K4.5 — замість
    # перезаморозки (яка вбила б цю властивість) навмисні зміни поведінки оголошуються тут.
    "incidents": "K7f — інцидент фіксується без драбини; фікстура зняла стару сліпоту",
}

# Оптовий виняток на `degraded` зняв би сторожа з поля, від якого залежить половина інваріантів.
# Тому дозволено РІВНО одне розходження: відкидання верифікатором більше не є збоєм машинерії
# (K9 — через `degraded` предикат `answered` міряв думку судді, і страта чесних відмов читалась 0/8).
def _verifier_reject_no_longer_degrades(observed: dict, expected: dict) -> bool:
    return (expected.get("degraded") is True and observed.get("degraded") is False
            and observed.get("accepted") is False and observed.get("verdict_kind") is not None)


OLD_PARTIAL = "Часткова відповідь на основі здобутого:"
NEW_PARTIAL = "Завершити не вдалося."


def _partial_rung_no_longer_dumps(observed: dict, expected: dict) -> bool:
    """Рунга `partial` більше не вивалює сирий payload; текст змінився лише в ній (K9/борг 45)."""
    was, now = expected.get("answer") or "", observed.get("answer") or ""
    return was.startswith(OLD_PARTIAL) and now.startswith(NEW_PARTIAL)


SANCTIONED_WHEN = {
    "degraded": _verifier_reject_no_longer_degrades,
    "answer": _partial_rung_no_longer_dumps,
}


def _apply_sanctioned_when(observed: dict, expected: dict) -> dict:
    out = dict(observed)
    for field, allowed in SANCTIONED_WHEN.items():
        if out.get(field) != expected.get(field) and allowed(observed, expected):
            out[field] = expected[field]
    return out


SANCTIONED_CONDITIONS = {
    # Умова була НЕПРАВИЛЬНО налаштована на момент заморозки: `toolset="none"` з тул-агентним
    # промптом, який перелічує інструменти, яких немає. K7 міряв її під `lang/plain`; після K4.5
    # решітка успадкувала `agent/v2`, і заголовкове число K7 перестало відтворюватись.
    "gate-notools-mamay": "K7-repro — промпт повернуто на lang/plain (структурне правило слота)",
    "gate-notools-lapa": "K7-repro — те саме",
}


def _without_sanctioned(cell: dict) -> dict:
    return {k: v for k, v in cell.items() if k not in SANCTIONED}


def test_every_condition_reproduces_the_frozen_run(expected):
    observed = _observed(_frozen_names(expected))
    assert sorted(observed) == sorted(expected), "склад умов змінився"
    diff = [k for k in expected
            if k.split("|", 1)[0] not in SANCTIONED_CONDITIONS
            and _project(_without_sanctioned(_apply_sanctioned_when(observed[k], expected[k])),
                         _without_sanctioned(expected[k]))
            != _without_sanctioned(expected[k])]
    assert not diff, f"специфікація змінила поведінку в {len(diff)} клітинках: {diff[:5]}"


def test_each_sanctioned_condition_really_diverges(expected):
    """Виняток рівня умови мусить відповідати реальному розходженню, інакше він мертвий."""
    observed = _observed(_frozen_names(expected))
    for cond, why in SANCTIONED_CONDITIONS.items():
        cells = [k for k in expected if k.startswith(cond + "|")]
        assert cells, f"{cond} немає у фікстурі — виняток зайвий"
        assert any(observed[k] != expected[k] for k in cells), (
            f"виняток «{cond}» ({why}) більше не потрібен — прибрати")


def test_each_sanctioned_exemption_is_actually_used(expected):
    """Виняток без розходження — мертвий виняток, який тихо приховує наступну зміну."""
    observed = _observed(_frozen_names(expected))
    for field, why in SANCTIONED.items():
        changed = [k for k in expected if observed[k][field] != expected[k][field]]
        assert changed, f"виняток «{field}» ({why}) більше не потрібен — прибрати"


def test_parity_still_notices_a_dropped_or_changed_field(expected):
    """Проєкція не має перетворити паритет на формальність."""
    key = next(iter(expected))
    tampered = dict(expected[key], steps=expected[key]["steps"] + 1)
    assert _project(tampered, expected[key]) != expected[key]
    with pytest.raises(KeyError):
        _project({k: v for k, v in expected[key].items() if k != "steps"}, expected[key])


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


def test_hetero_cost_is_an_interval_when_lanes_are_unknown():
    lo, hi = rate_bounds("hetero")
    assert lo < hi, "без розкладки за ярусом гетеро не має точкової ціни"
    assert rate_bounds("lapa")[0] == rate_bounds("lapa")[1]
    lo_lapa, _ = cost_usd(1_000_000, "lapa")
    assert lo_lapa == pytest.approx(0.05 * 0.8 + 0.15 * 0.2), "ціна з ринкового проксі, не з голови"


def test_hetero_cost_becomes_exact_once_lanes_are_attributed():
    """Борг 25: розкладка за ярусом перетворює інтервал на число."""
    from evalkit.cost import attributed, lane_cost, lane_share

    rows = [EvalResult(item_id="x", category="c", condition="hetero@8", seed=1, success=True,
                       checks={}, tokens=1_000_000,
                       tokens_by_lane={"lapa": 600_000, "mamay": 400_000},
                       prompt_by_lane={"lapa": 600_000, "mamay": 400_000})]
    lo, hi = cost_of_results(rows, CONDITIONS)["hetero@8"]
    assert lo == hi == pytest.approx(0.6 * 0.05 + 0.4 * 0.08), "усе промпт — лише вхідні ставки"
    assert lane_share(rows)["hetero@8"] == {"lapa": pytest.approx(0.6), "mamay": pytest.approx(0.4)}

    blind = [rows[0].model_copy(update={"tokens_by_lane": {"unknown": 1_000_000}})]
    blo, bhi = cost_of_results(blind, CONDITIONS)["hetero@8"]
    assert blo < bhi, "«unknown» не є атрибуцією — інтервал мусить лишитись"
    assert not attributed({"unknown": 5}) and attributed({"lapa": 5})
    assert lane_cost({"lapa": 1_000_000}, {"lapa": 1_000_000}) == pytest.approx(0.05)
    assert lane_cost({"lapa": 1_000_000}, {}) == pytest.approx(0.15), "нуль промпту = усе вихід"


def test_cost_ranks_lapa_below_mamay():
    rows = [EvalResult(item_id="x", category="c", condition=c, seed=1, success=True,
                       checks={}, tokens=1000)
            for c in ("single-lapa", "single-mamay")]
    cost = cost_of_results(rows, CONDITIONS)
    assert cost["single-lapa"][1] < cost["single-mamay"][0]
    per = usd_per_success(rows, CONDITIONS)
    assert per["single-lapa"][0] < per["single-mamay"][0]


def test_the_conditional_exemption_is_really_used(expected):
    """Умовний виняток без реального розходження — мертвий сторож, який приховає наступну зміну."""
    observed = _observed(_frozen_names(expected))
    for field, allowed in SANCTIONED_WHEN.items():
        hit = [k for k in expected if allowed(observed[k], expected[k])]
        assert hit, f"умовний виняток «{field}» більше не потрібен — прибрати"


def test_the_partial_exemption_does_not_hide_a_real_answer_change(expected):
    assert not _partial_rung_no_longer_dumps({"answer": "1918"}, {"answer": "1919"})
    assert not _partial_rung_no_longer_dumps({"answer": NEW_PARTIAL}, {"answer": "1918"})


def test_the_conditional_exemption_does_not_hide_other_flips(expected):
    """Дозволено лише перевертання True→False під відкиданням судді; решта мусить падати."""
    assert not _verifier_reject_no_longer_degrades(
        {"degraded": True, "accepted": True, "verdict_kind": "supported"}, {"degraded": False})
    assert not _verifier_reject_no_longer_degrades(
        {"degraded": False, "accepted": False, "verdict_kind": None}, {"degraded": True})
