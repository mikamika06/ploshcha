import pytest

from evalkit.checks import split_checks
from evalkit.harness import load_items
from evalkit.validate import ITEMS_DIR, synth_result
from ploshcha_sim.adapters.registry_kb import MISSING
from ploshcha_sim.compose import build_toolbox
from ploshcha_sim.domain.spec import AppSpec
from ploshcha_sim.ports.tool import ToolCall

VILLAGE_AS_WRITTEN = {
    "sum-03": "Сухому Яру", "max-03": "Сухому Яру", "craft-03": "Сухому Яру",
    "sum-04": "Гайворонці", "missing-04": "Гайворонці",
    "sum-08": "Вербівці", "max-08": "Вербівці", "count-08": "Вербівці",
    "sum-20": "Великому Лузі",
}


@pytest.fixture(scope="module")
def toolbox():
    return build_toolbox(AppSpec().with_(toolset="registry"))


@pytest.fixture(scope="module")
def items():
    return load_items(str(ITEMS_DIR / "chain.jsonl"))


def _walk(toolbox, village_as_written):
    """Оракул: рівно те, що мусив би зробити ідеальний агент, і нічого більше."""
    listed = toolbox.call(ToolCall(tool="список_записів", args={"село": village_as_written}))
    assert listed.ok and listed.value["відомо"], f"реєстр не знайшов «{village_as_written}»"
    ids = listed.value["записи"]
    records = []
    for rid in ids:
        got = toolbox.call(ToolCall(tool="запис", args={"ідентифікатор": rid}))
        assert got.ok and got.value["відомо"], f"запис {rid} недосяжний"
        records.append(got.value)
    return ids, records


def test_the_registry_resolves_every_inflected_village(toolbox, items):
    """Вісь U6: у задачі село в місцевому відмінку, у реєстрі — у називному."""
    for item in items:
        ids, _ = _walk(toolbox, VILLAGE_AS_WRITTEN[item.id])
        assert ids, f"{item.id}: порожній список записів"


def test_the_toolbox_can_actually_produce_the_gold_answer(toolbox, items):
    """Дефект 6 з V6: еталон проходив предикати, але інструменти НЕ могли його дати."""
    by_id = {i.id: i for i in items}

    for iid in ("sum-03", "sum-04", "sum-08", "sum-20"):
        _, recs = _walk(toolbox, VILLAGE_AS_WRITTEN[iid])
        total = sum(r["сума"] for r in recs if r["сума"] != MISSING)
        outcome, _ = split_checks(by_id[iid].checks, synth_result(str(total),
                                                                 by_id[iid].gold_tools))
        assert all(outcome.values()), f"{iid}: сума з інструментів {total} не проходить чеки"

    for iid in ("max-03", "max-08"):
        _, recs = _walk(toolbox, VILLAGE_AS_WRITTEN[iid])
        best = max((r for r in recs if r["сума"] != MISSING), key=lambda r: r["сума"])
        answer = f"{best['майстер']}, {best['сума']}"
        outcome, _ = split_checks(by_id[iid].checks, synth_result(answer, by_id[iid].gold_tools))
        assert all(outcome.values()), f"{iid}: «{answer}» не проходить чеки"

    ids, recs = _walk(toolbox, VILLAGE_AS_WRITTEN["missing-04"])
    holes = [(rid, r["майстер"]) for rid, r in zip(ids, recs) if r["сума"] == MISSING]
    assert len(holes) == 1, "айтем передбачає РІВНО одну дірку"
    answer = f"{holes[0][0]} — {holes[0][1]}"
    outcome, _ = split_checks(by_id["missing-04"].checks,
                              synth_result(answer, by_id["missing-04"].gold_tools))
    assert all(outcome.values()), f"missing-04: «{answer}» не проходить чеки"

    ids, recs = _walk(toolbox, VILLAGE_AS_WRITTEN["count-08"])
    answer = f"всього={len(ids)}, із сумою={sum(1 for r in recs if r['сума'] != MISSING)}"
    outcome, _ = split_checks(by_id["count-08"].checks,
                              synth_result(answer, by_id["count-08"].gold_tools))
    assert all(outcome.values()), f"count-08: «{answer}» не проходить чеки"

    _, recs = _walk(toolbox, VILLAGE_AS_WRITTEN["craft-03"])
    answer = ", ".join(dict.fromkeys(r["ремесло"] for r in recs))
    outcome, _ = split_checks(by_id["craft-03"].checks,
                              synth_result(answer, by_id["craft-03"].gold_tools))
    assert all(outcome.values()), f"craft-03: «{answer}» не проходить чеки"


def test_declared_chain_length_matches_the_registry(toolbox, items):
    """Мінімальний ланцюг = список + N записів + обчислення + відповідь."""
    for item in items:
        ids, _ = _walk(toolbox, VILLAGE_AS_WRITTEN[item.id])
        assert item.chain_len == len(ids) + 3, (
            f"{item.id}: заявлено {item.chain_len}, реєстр дає {len(ids) + 3}")


def test_the_set_grades_chain_length(items):
    lengths = sorted({i.chain_len for i in items})
    assert lengths == [6, 7, 11, 23], f"сходинки довжини зникли: {lengths}"
    assert lengths[-1] > 2 * lengths[-2], "потрібен набір, помітно довший за стелю з покриттям"


def test_the_distractor_is_not_reachable_through_listing(toolbox):
    """Дистрактор карає вгадування ідентифікаторів, але не ловиться при чесному обході."""
    ids, _ = _walk(toolbox, "Вербівці")
    assert "зп-1901-10" not in ids
    guessed = toolbox.call(ToolCall(tool="запис", args={"ідентифікатор": "зп-1901-10"}))
    assert guessed.ok and guessed.value["відомо"], "дистрактор мусить існувати при прямому запиті"


def test_unknown_village_and_record_are_loud_not_silent(toolbox):
    nowhere = toolbox.call(ToolCall(tool="список_записів", args={"село": "Полтава"}))
    assert nowhere.ok and nowhere.value["відомо"] is False and nowhere.value["записи"] == []
    nothing = toolbox.call(ToolCall(tool="запис", args={"ідентифікатор": "зп-9999-99"}))
    assert nothing.ok and nothing.value["відомо"] is False
