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


def test_gate_catches_a_predicate_unsatisfiable_under_another_toolset():
    """Дефект №13 (= повторення №8): чек вимагає `запис`, а в агрегатному наборі його немає."""
    hardcoded = EvalItem(id="x", category="c", task="t",
                         checks=[{"kind": "used_tool_any", "tools": ["запис"]}],
                         gold=["372"], foil=["391"])
    report = validate_item(hardcoded, ("registry", "registry_agg"))
    assert report.unsatisfiable, "нездійсненний чек мусить бути виявлений"
    assert "registry_agg" in report.unsatisfiable[0]
    assert validate_item(hardcoded, ("registry",)).unsatisfiable == []

    agnostic = hardcoded.model_copy(update={
        "checks": [{"kind": "used_tool_any", "tools": ["запис", "записи_села"]}]})
    assert validate_item(agnostic, ("registry", "registry_agg")).unsatisfiable == []


DUMP = ('Часткова відповідь на основі здобутого:\n'
        'список_записів: {"відомо": true, "село": "Сухий Яр", "записи": ["сх-1904-01"]}\n'
        'запис: {"відомо": true, "майстер": "Остап Заячук", "сума": 91}')


def test_gate_catches_a_raw_dump_passing_content_checks():
    """Дефект №14: рунга `partial` вивалює payload інструмента, і чек ловить значення в дампі."""
    loose = EvalItem(id="x", category="c", task="t",
                     checks=[{"kind": "answer_contains_any", "values": ["Остап", "Заячук"]},
                             {"kind": "answer_contains_any", "values": ["91"]}],
                     gold=["Остап Заячук, 91 карбованець."], foil=[DUMP])
    assert validate_item(loose).foil_passed, "дамп не має проходити змістові чеки"

    tight = loose.model_copy(update={"checks": loose.checks + [{"kind": "answer_no_json"}]})
    assert validate_item(tight).ok
    assert validate_item(tight).gold_failed == []


def test_dump_predicate_reads_the_actual_payload():
    from evalkit.checks import check
    payload = {"відомо": True, "майстер": "Остап Заячук", "сума": 91}
    dumped = synth_result("ось що здобуто: " + __import__("json").dumps(payload, ensure_ascii=False))
    dumped.scratch = [{"call": {"tool": "запис"}, "result": payload}]
    assert not check({"kind": "answer_not_dumped"}, dumped)
    clean = synth_result("Остап Заячук, 91 карбованець.")
    clean.scratch = [{"call": {"tool": "запис"}, "result": payload}]
    assert check({"kind": "answer_not_dumped"}, clean)


def test_every_foil_must_fail_not_merely_one():
    """Дефект №15: один добрий антиеталон маскував інший, що проходив усі чеки."""
    item = EvalItem(id="x", category="c", task="t",
                    checks=[{"kind": "answer_contains_any", "values": ["178"]}],
                    gold=["178"], foil=["зовсім інше", "усе-таки 178"])
    report = validate_item(item)
    assert report.foil_passed == ["усе-таки 178"], "другий антиеталон мусить бути виявлений"


def test_gate_catches_phantom_gold_tools():
    """`gold_tools` монтує фейкову трасу — вигаданий інструмент робить її брехнею."""
    item = EvalItem(id="x", category="c", task="t", checks=[{"kind": "answered"}],
                    gold=["будь-що"], gold_tools=["такого_немає"])
    assert validate_item(item, ("registry",)).unsatisfiable


def test_hygiene_is_allowed_to_be_unsatisfiable():
    """`tool_calls_at_least` саме й міряє, чи був обхід — під агрегатом він законно провалюється."""
    item = EvalItem(id="x", category="c", task="t",
                    checks=[{"kind": "answer_contains_any", "values": ["372"]},
                            {"kind": "tool_calls_at_least", "n": 9}],
                    gold=["372"], foil=["391"])
    assert validate_item(item, ("registry", "registry_agg")).ok


def test_every_item_set_declares_its_toolsets():
    from evalkit.validate import ITEM_SET_TOOLSETS, TOOLSETS
    missing = [n for n in item_sets() if n not in ITEM_SET_TOOLSETS]
    assert not missing, f"набір без оголошених наборів інструментів: {missing}"
    unknown = {ts for names in ITEM_SET_TOOLSETS.values() for ts in names} - set(TOOLSETS)
    assert not unknown, f"невідомий набір інструментів: {unknown}"


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


def test_every_refusal_item_uses_the_shared_vocabulary():
    """Словник відмови розходився двічі: у `lexis` бракувало «немає тлумачення», у `audit` —
    «немає інформації», і обидва рази ПРАВИЛЬНА відповідь читалась як провал. Один список на всі
    набори, інакше наступне формулювання знову проґавимо."""
    from evalkit.refusal import ADMIT

    wrong = []
    for name in item_sets():
        for item in load_items(str(ITEMS_DIR / f"{name}.jsonl")):
            if "absent" not in item.id and "abstain" not in item.id:
                continue
            for spec in item.checks:
                if spec["kind"] != "answer_contains_any":
                    continue
                if any(v in ADMIT for v in spec["values"]) and spec["values"] != ADMIT:
                    wrong.append(f"{name}/{item.id}: власний список замість спільного")
    assert not wrong, wrong
