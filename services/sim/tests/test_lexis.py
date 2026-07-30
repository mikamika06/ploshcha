import json
from pathlib import Path

import pytest

from ploshcha_sim.adapters.lexicon_kb import DATA, NOT_FOUND, article, words
from ploshcha_sim.adapters.tools_lexis import LEXIS_TOOLS, СловникArgs, _словник
from ploshcha_sim.compose import TOOLSETS

ITEMS = Path(__file__).resolve().parents[1] / "evalkit" / "items" / "lexis.jsonl"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def items() -> list[dict]:
    return [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_the_source_and_licence_are_recorded(payload):
    """Дані з чужого джерела без атрибуції — це борг, а не набір. SYS-KBSRC вимагає це в метаданих."""
    assert payload["ліцензія"] == "CC BY-SA 4.0"
    assert "wiktionary" in payload["джерело"].casefold()
    assert payload["здобуто"]


def test_every_entry_has_a_sense_an_example_and_a_source(payload):
    for entry in payload["статті"]:
        assert entry["значення"], entry["слово"]
        assert entry["приклади"], entry["слово"]
        assert entry["страт"] in ("rare", "common"), entry["слово"]


def test_the_example_actually_contains_the_word(payload):
    """Приклад, у якому слова немає, не може бути входом задачі про це слово."""
    for entry in payload["статті"]:
        stem = entry["слово"][:3].casefold()
        assert any(stem in ex.casefold() for ex in entry["приклади"]), entry["слово"]


def test_the_article_is_material_not_a_verdict():
    """Стаття мусить давати тлумачення й приклад, а не готову відповідь на задачу."""
    text = article("ботей")
    assert text is not None and "отара" in text
    assert "Приклад слововжитку" in text and "Джерело" in text


def test_an_absent_word_is_a_loud_miss():
    assert article("автомобіль") is None
    out = _словник(СловникArgs(слово="автомобіль"))
    assert out["відомо"] is False and out["стаття"] == NOT_FOUND


def test_an_inflected_form_resolves_to_the_nearest_word_not_the_first_match():
    """«ватри» не має тягнути «ватралка»: обидва мають той самий стем, і тихий збіг дав би не те."""
    assert article("ватри").startswith("ватра —")
    assert article("ватралки").startswith("ватралка —")
    assert article("ВАТРА").startswith("ватра —")


def test_both_strata_are_present_and_the_rare_one_is_bigger():
    rare, common = words("rare"), words("common")
    assert len(rare) >= 16 and len(common) >= 8
    assert not set(rare) & set(common)


def test_the_toolset_is_wired_with_exactly_the_reference_tool():
    assert TOOLSETS["lexis"] is LEXIS_TOOLS
    assert {t.name for t in LEXIS_TOOLS} == {"словник", "final_answer"}


def test_the_item_set_has_both_strata_and_stable_ids(items):
    assert len(items) == 24
    strata = {i["category"] for i in items}
    assert strata == {"lexis_rare", "lexis_common"}
    assert len({i["id"] for i in items}) == len(items)


def test_no_accepted_key_is_present_in_the_task_sentence(items):
    """Ехо-ключ дає прохід простим копіюванням речення — саме так «колодязь» ламав цямрину."""
    bad = []
    for item in items:
        values = next(c["values"] for c in item["checks"] if c["kind"] == "answer_contains_any")
        task = item["task"].casefold()
        bad += [f"{item['id']}:{v}" for v in values if v.casefold() in task]
    assert not bad, bad


def test_every_accepted_key_is_grounded_in_the_source(payload, items):
    """Ключ, якого немає ні в значенні, ні в списку оголошених синонімів, означав би вигаданий еталон."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_lexis_items import ALSO, PLAN

    senses = {e["слово"]: " ".join(e["значення"]).casefold() for e in payload["статті"]}
    for item in items:
        word = item["id"].split("-", 2)[2]
        values = next(c["values"] for c in item["checks"] if c["kind"] == "answer_contains_any")
        stems, extra = PLAN[word][1], ALSO.get(word, [])
        assert values == stems + extra, item["id"]
        for stem in stems:
            assert stem.casefold() in senses[word], f"{word}: «{stem}» немає у значенні"


def test_the_answer_check_rejects_the_foil(items):
    for item in items:
        values = next(c["values"] for c in item["checks"] if c["kind"] == "answer_contains_any")
        for foil in item["foil"]:
            assert not any(v.casefold() in foil.casefold() for v in values), item["id"]


def test_the_set_declares_the_paired_toolsets(items):
    for item in items:
        assert item["toolsets"] == ["none", "lexis"], item["id"]


def test_no_item_demands_a_tool_because_the_baseline_has_none(items):
    """Той самий айтем ганяється БЕЗ інструментів — тул-предикат зробив би базу нездійсненною."""
    for item in items:
        kinds = {c["kind"] for c in item["checks"]}
        assert not kinds & {"used_tool", "used_tool_any", "tool_calls_at_least"}, item["id"]
