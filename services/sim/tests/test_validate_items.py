import pytest

from evalkit.harness import EvalItem, load_items
from evalkit.validate import (
    ITEMS_DIR,
    coverage,
    format_report,
    item_sets,
    synth_result,
    validate_file,
    validate_item,
)

GOLD_JSON = '{"твердження":"Битва під Крутами відбулася 1919 року.","вердикт":"хибним","правильний_рік":1918}'
FOIL_JSON = '{"вердикт":"правильно","правильний_рік":1919}'


def _item(checks, **kw):
    return EvalItem(id="x", category="c", task="t", checks=checks,
                    gold=[GOLD_JSON], foil=[FOIL_JSON], gold_tools=["check_date"], **kw)


def test_gate_catches_a_hardcoded_word_form():
    """Дефект №4: чек вимагав «хибно», еталон каже «хибним»."""
    strict = _item([{"kind": "answer_contains", "value": "хибно"}])
    report = validate_item(strict)
    assert report.gold_failed, "надто суворий чек мусить бути виявлений"

    family = _item([{"kind": "answer_contains_any",
                     "values": ["хибн", "неправд", "невірн"]}])
    assert validate_item(family).gold_failed == []


def test_gate_catches_a_single_synonym():
    """Дефект №5: модель каже «Неправда», чек вимагає «хибн»."""
    item = EvalItem(id="x", category="c", task="t",
                    checks=[{"kind": "answer_contains", "value": "хибн"}],
                    gold=['{"вердикт":"Неправда","правильний_рік":1648}'],
                    foil=['{"вердикт":"правильно"}'])
    assert validate_item(item).gold_failed


def test_gate_catches_a_hardcoded_tool_name():
    """Дефект №8: чек вимагає латинську назву, змонтовано український набір."""
    latin = EvalItem(id="x", category="c", task="t",
                     checks=[{"kind": "used_tool", "tool": "check_date"}],
                     gold=["будь-що"], gold_tools=["перевірити_дату"])
    assert validate_item(latin).gold_failed

    agnostic = latin.model_copy(update={
        "checks": [{"kind": "used_tool_any", "tools": ["check_date", "перевірити_дату"]}]})
    assert validate_item(agnostic).gold_failed == []


def test_gate_catches_a_vacuous_check():
    """Чек, який пропускає навіть свідомо хибну відповідь, нічого не міряє."""
    vacuous = _item([{"kind": "answered"}])
    report = validate_item(vacuous)
    assert report.foil_vacuous, "хибна відповідь не має проходити всі чеки"


def test_hygiene_checks_do_not_block_the_gold():
    """Гігієна не є результатом — еталон не мусить її задовольняти."""
    item = _item([{"kind": "answer_contains_any", "values": ["хибн"]},
                  {"kind": "abstain"}])
    assert validate_item(item).gold_failed == [], "abstain — гігієна, не результат"


def test_blank_answer_is_not_an_answer():
    """Дефект №9: `answer is not None` пропускав порожній рядок як відповідь."""
    from evalkit.checks import check
    assert not check({"kind": "answered"}, synth_result(""))
    assert not check({"kind": "answered"}, synth_result("   "))
    assert check({"kind": "answered"}, synth_result("1648"))
    assert not check({"kind": "abstain"}, synth_result(""))


def test_an_item_whose_only_check_is_hygiene_is_unwinnable():
    """Дефект №10: після переносу abstain у гігієну такі айтеми ніколи не проходять."""
    from evalkit.checks import split_checks

    outcome, hygiene = split_checks([{"kind": "abstain"}], synth_result("Привіт!"))
    assert not outcome and hygiene, "лишилась тільки гігієна — результат не міряється"

    fixed = [{"kind": "answered"}, {"kind": "abstain"}]
    outcome, _ = split_checks(fixed, synth_result("Привіт!"))
    assert outcome and all(outcome.values())


def test_every_item_has_at_least_one_outcome_check():
    from evalkit.checks import split_checks

    for name in item_sets():
        for item in load_items(str(ITEMS_DIR / f"{name}.jsonl")):
            outcome, _ = split_checks(item.checks, synth_result("x"))
            assert outcome, f"{name}/{item.id}: лише гігієна — айтем непроходжуваний"


def test_synth_result_shapes_scratch_like_the_orchestrator():
    r = synth_result("текст", ["check_date", "calc"])
    assert [x["call"]["tool"] for x in r.scratch] == ["check_date", "calc"]
    assert r.answer == "текст" and r.steps == 3


def test_audit_set_is_fully_covered_and_clean():
    items = load_items(str(ITEMS_DIR / "audit.jsonl"))
    assert coverage(items) == 1.0, "форма флагмана мусить бути покрита еталонами повністю"
    reports = validate_file("audit")
    assert all(r.ok for r in reports), format_report("audit", reports)
    assert all(r.has_foil for r in reports), "кожен айтем потребує й хибної відповіді"


@pytest.mark.parametrize("name", item_sets())
def test_every_item_with_a_gold_answer_passes_its_own_checks(name):
    reports = validate_file(name)
    broken = [r for r in reports if not r.ok]
    assert not broken, format_report(name, reports)


def test_report_names_the_defect():
    strict = _item([{"kind": "answer_contains", "value": "хибно"}])
    text = format_report("t", [validate_item(strict)])
    assert "еталон НЕ проходить" in text
