import pytest

from evalkit.conditions import CONDITIONS, judge_warnings
from ploshcha_sim.compose import build_router
from ploshcha_sim.domain.spec import AppSpec

INVARIANT = "ПОРУШЕННЯ"


class _Stub:
    def __init__(self, name: str):
        self.model = name


def _router(**kw):
    return build_router(AppSpec(**kw), lapa=_Stub("lapa"), mamay=_Stub("mamay"))


@pytest.mark.parametrize("routing,expect", [("hetero", "mamay"), ("mamay", "mamay"), ("lapa", "lapa")])
def test_auto_keeps_the_old_behaviour(routing, expect):
    """`auto` = як було: ярус судді визначає `routing`. Інакше зсунулись би заморожені числа."""
    assert _router(routing=routing).lane("judge") == expect


@pytest.mark.parametrize("routing", ["hetero", "mamay", "lapa"])
@pytest.mark.parametrize("judge", ["mamay", "lapa"])
def test_the_judge_lane_overrides_routing(routing, judge):
    router = _router(routing=routing, judge_lane=judge)
    assert router.lane("judge") == judge
    assert router.route("judge").model == judge


def test_overriding_the_judge_does_not_touch_the_answering_lanes():
    router = _router(routing="lapa", judge_lane="mamay")
    assert router.route("judge").model == "mamay"
    for kind in ("select", "generate", "synthesize", "ground"):
        assert router.route(kind).model == "lapa", kind


def test_self_judging_is_detected_not_assumed():
    assert _router(routing="mamay").self_judging() is True
    assert _router(routing="lapa").self_judging() is True
    assert _router(routing="lapa", judge_lane="mamay").self_judging() is False
    assert _router(routing="mamay", judge_lane="lapa").self_judging() is False


def test_hetero_is_also_self_judging_because_mamay_generates():
    """У гетеро відповідь генерує Mamay, тож Mamay-суддя — це теж самосуд, хоч набір яругів мішаний."""
    assert _router(routing="hetero").self_judging() is True


def test_the_cross_lane_cells_exist_and_are_clean():
    """Дві порожні клітинки хреста — це і є замір: чи інфлює самосуд."""
    warnings = judge_warnings()
    for name in ("lex-am-jl", "lex-al-jm"):
        assert name in CONDITIONS
        assert name not in warnings, f"{name} мусить бути перехресним"


def test_the_lapa_invariant_violations_are_named_not_hidden():
    """Інваріант «ніколи Lapa-судить-Lapa» ніде не був примусовим — 9 умов його порушують.

    Тест не забороняє їх (частина потрібна саме як контроль самосуду), а вимагає, щоб порушення
    було ГУЧНИМ: мовчазне порушення оголошеного інваріанта гірше за оголошене.
    """
    warnings = judge_warnings()
    violations = {n for n, why in warnings.items() if INVARIANT in why}
    assert violations, "детектор перестав ловити — інваріант знову невидимий"
    for name in violations:
        spec = CONDITIONS[name]
        assert spec.routing == "lapa" and spec.judge_lane in ("auto", "lapa")


def test_every_self_judging_condition_is_flagged():
    """Не можна тихо додати умову з самосудом: попередження рівня конфігурації, як `shape_warnings`."""
    warnings = judge_warnings()
    for name, spec in CONDITIONS.items():
        if not spec.verifier:
            continue
        router = build_router(spec, lapa=_Stub("lapa"), mamay=_Stub("mamay"))
        assert (name in warnings) == router.self_judging(), name


def test_the_judge_lane_changes_the_spec_hash():
    """Ярус судді — частина умови прогону, тому мусить бути в `sha256`, інакше кеш звітів збреше."""
    base = AppSpec(routing="lapa")
    assert base.sha256 != AppSpec(routing="lapa", judge_lane="mamay").sha256
