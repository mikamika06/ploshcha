import pytest

from evalkit.conditions import CONDITIONS, shape_warnings
from ploshcha_sim.adapters.skillbox import ANSWER_CAPABILITY, SkillBox
from ploshcha_sim.compose import TOOLSETS, build_skillbox, build_toolbox
from ploshcha_sim.domain.gate import FINAL_TOOL
from ploshcha_sim.domain.skill import (
    ITERATION_CEILING,
    SkillSpec,
    aggregate_skills,
    collection_skills,
    iteration_load,
    needs_fanout,
    scope,
    shape_notes,
    untrusted_skills,
    write_skills,
)
from ploshcha_sim.domain.spec import AppSpec
from ploshcha_sim.ports.tool import ToolCall

TOOLSET_NAMES = ("default", "ua", "registry", "registry_teach", "registry_agg")


def _box(toolset: str) -> SkillBox:
    return build_skillbox(AppSpec().with_(toolset=toolset))


@pytest.mark.parametrize("toolset", TOOLSET_NAMES)
def test_skillbox_is_a_toolport_identical_to_the_plain_toolbox(toolset):
    """Скіл — обгортка з декларацією; оркестратор не має відчути різниці."""
    spec = AppSpec().with_(toolset=toolset)
    assert [s.name for s in _box(toolset).specs()] == [s.name for s in build_toolbox(spec).specs()]
    assert [s.params for s in _box(toolset).specs()] == [s.params for s in build_toolbox(spec).specs()]


def test_the_call_result_matches_the_plain_toolbox():
    call = ToolCall(tool="check_date", args={"event": "Битва під Крутами", "year": 1918})
    skilled = _box("default").call(call)
    plain = build_toolbox(AppSpec()).call(call)
    assert skilled.ok == plain.ok and skilled.value == plain.value


def test_unknown_tool_stays_loud():
    result = _box("default").call(ToolCall(tool="немає", args={}))
    assert not result.ok and result.error == "unknown_tool"


@pytest.mark.parametrize("toolset", TOOLSET_NAMES)
def test_every_tool_carries_a_declaration(toolset):
    specs = _box(toolset).skill_specs()
    assert len(specs) == len(TOOLSETS[toolset])
    assert not [s for s in specs if s.capability == "unknown"], "інструмент без оголошення"


def test_declarations_reproduce_the_k7c_finding():
    """Форма даних — не орнамент: та сама різниця, що дала 0.125 проти 0.625."""
    collection = _box("registry").skill_specs()
    aggregate = _box("registry_agg").skill_specs()

    assert [s.name for s in collection_skills(collection)] == ["список_записів"]
    assert collection_skills(aggregate) == []
    assert [s.name for s in aggregate_skills(aggregate)] == ["записи_села"]

    from ploshcha_sim.adapters.registry_kb import VILLAGES, ids_for
    biggest = max(len(ids_for(v)) for v in VILLAGES)
    assert iteration_load(collection) == biggest, "max_items беруться з КБ, не вписуються руками"
    assert needs_fanout(collection), f"{biggest} > стелі без покриття (2)"
    assert not needs_fanout(aggregate)
    assert shape_notes(collection) == [f"skills:collection=список_записів×{biggest}"]
    assert shape_notes(aggregate) == []


def test_the_ceiling_is_the_measured_one():
    assert ITERATION_CEILING == 2, "без покриття: 2 виклики в 104 прогонах з 104 (K7c+K7d)"
    small = [SkillSpec(name="s", capability="c", shape="collection", max_items=2)]
    assert not needs_fanout(small), "рівно стеля — ще не потребує fan-out"
    assert needs_fanout([SkillSpec(name="s", capability="c", shape="collection", max_items=3)])


def test_the_ceiling_depends_on_the_configuration():
    """K7e виправив K7c: 2 — властивість циклу БЕЗ відстеження залишку, не системи."""
    from ploshcha_sim.domain.skill import ITERATION_CEILING_COVERAGE, ceiling_for

    assert ceiling_for() == 2 and ceiling_for(coverage=True) == ITERATION_CEILING_COVERAGE == 8
    eight = [SkillSpec(name="s", capability="c", shape="collection", max_items=8)]
    twenty = [SkillSpec(name="s", capability="c", shape="collection", max_items=20)]

    assert needs_fanout(eight), "без покриття 8 елементів цикл не обходить"
    assert not needs_fanout(eight, coverage=True), "з покриттям 8/8 викликів спостережено"
    assert needs_fanout(twenty, coverage=True), "на 20 обходу не спостерігалось жодного"
    assert shape_notes(eight, coverage=True) == [] and shape_notes(eight) != []


def test_scoping_keeps_the_answer_capability():
    box = _box("registry")
    scoped = box.scoped(("registry.record",))
    names = [s.name for s in scoped.specs()]
    assert FINAL_TOOL in names, "суб-агент без завершення не може віддати результат"
    assert "список_записів" not in names and "запис" in names
    assert ANSWER_CAPABILITY in scoped.capabilities()


def test_scoping_is_pure_at_the_domain_level():
    specs = [SkillSpec(name="a", capability="x"), SkillSpec(name="b", capability="y")]
    assert [s.name for s in scope(specs, ("y",))] == ["b"]
    assert scope(specs, ()) == []


def test_side_effect_and_trust_default_to_the_safe_side():
    for toolset in TOOLSET_NAMES:
        specs = _box(toolset).skill_specs()
        assert write_skills(specs) == [], "жоден наявний скіл не пише"
        assert untrusted_skills(specs) == [], "чужого тексту ще не читаємо — це K10"


COLLECTION_TOOLSETS = {"registry", "registry_teach", "registry_sum", "registry_reduce", "docs"}


def test_conditions_with_a_collection_are_flagged_and_aggregate_ones_are_not():
    flagged = shape_warnings()
    for name in ("chain-schema@16", "chain-text@16", "docs-schema@16", "docs-text@16"):
        assert name in flagged, name
    for name in ("chain-agg-schema@16", "chain-agg-text@16",
                 "docs-agg-schema@16", "docs-agg-text@16"):
        assert name not in flagged, name
    off = [n for n in flagged if CONDITIONS[n].toolset not in COLLECTION_TOOLSETS]
    assert not off, f"позначено умову без колекційного скіла: {off}"


def test_the_warning_is_loud_but_not_corrective():
    """Реєстр НЕ підміняє колекцію агрегатом: там, де колекція неминуча, підміна б це сховала."""
    box = _box("registry")
    assert [s.name for s in box.specs()] == [t.name for t in TOOLSETS["registry"]]
    from ploshcha_sim.adapters.registry_kb import VILLAGES, ids_for
    biggest = max(len(ids_for(v)) for v in VILLAGES)
    assert box.notes() == [f"skills:collection=список_записів×{biggest}"]
