import pytest

from evalkit.checks import split_checks
from evalkit.harness import load_items
from evalkit.validate import ITEMS_DIR, synth_result
from ploshcha_sim.adapters.docs_kb import crafts_in, has_year, mentions
from ploshcha_sim.compose import build_toolbox
from ploshcha_sim.domain.spec import AppSpec
from ploshcha_sim.ports.tool import ToolCall

DOC_OF = {
    "crafts-03": "літопис-1", "noyear-03": "літопис-1",
    "year-count-05": "літопис-2", "crafts-05": "літопис-2",
    "year-count-08": "літопис-3", "noyear-08": "літопис-3",
    "who-08": "літопис-3", "total-08": "літопис-3",
}


@pytest.fixture(scope="module")
def box():
    return build_toolbox(AppSpec().with_(toolset="docs"))


@pytest.fixture(scope="module")
def agg():
    return build_toolbox(AppSpec().with_(toolset="docs_agg"))


@pytest.fixture(scope="module")
def items():
    return {i.id: i for i in load_items(str(ITEMS_DIR / "docs.jsonl"))}


def _walk(box, document):
    listed = box.call(ToolCall(tool="список_абзаців", args={"документ": document}))
    assert listed.ok and listed.value["відомо"], f"документ «{document}» недосяжний"
    ids = listed.value["абзаци"]
    texts = []
    for pid in ids:
        got = box.call(ToolCall(tool="абзац", args={"ідентифікатор": pid}))
        assert got.ok and got.value["відомо"], f"абзац {pid} недосяжний"
        texts.append(got.value["текст"])
    return ids, texts


def _passes(item, answer):
    outcome, _ = split_checks(item.checks, synth_result(answer, item.gold_tools))
    return all(outcome.values()), outcome


def test_both_toolsets_reach_the_same_paragraphs(box, agg):
    """Колекційний і агрегатний набори мусять давати ІДЕНТИЧНІ дані, інакше порівняння форм — фікція."""
    for doc in ("літопис-1", "літопис-2", "літопис-3"):
        ids, texts = _walk(box, doc)
        whole = agg.call(ToolCall(tool="абзаци_документа", args={"документ": doc}))
        assert whole.ok and whole.value["відомо"]
        assert [p["ідентифікатор"] for p in whole.value["абзаци"]] == ids
        assert [p["текст"] for p in whole.value["абзаци"]] == texts


def test_the_toolbox_can_produce_every_gold_answer(box, items):
    counts = {}
    for iid, doc in DOC_OF.items():
        ids, texts = _walk(box, doc)
        counts[doc] = (ids, texts)

    ids, texts = counts["літопис-1"]
    crafts = sorted({c for t in texts for c in crafts_in(t)})
    ok, outcome = _passes(items["crafts-03"], ", ".join(crafts))
    assert ok, outcome
    no_year = [p for p, t in zip(ids, texts) if not has_year(t)]
    ok, outcome = _passes(items["noyear-03"], " і ".join(no_year))
    assert ok, outcome

    ids, texts = counts["літопис-2"]
    ok, outcome = _passes(items["year-count-05"],
                          f"з роком={sum(1 for t in texts if has_year(t))}")
    assert ok, outcome
    crafts = sorted({c for t in texts for c in crafts_in(t)})
    ok, outcome = _passes(items["crafts-05"], f"ремесел={len(crafts)}")
    assert ok, outcome

    ids, texts = counts["літопис-3"]
    with_year = sum(1 for t in texts if has_year(t))
    ok, outcome = _passes(items["year-count-08"], f"з роком={with_year}")
    assert ok, outcome
    ok, outcome = _passes(items["noyear-08"], f"без року={len(ids) - with_year}")
    assert ok, outcome
    ok, outcome = _passes(items["who-08"], " і ".join(mentions("літопис-3", "Ломачк")))
    assert ok, outcome
    ok, outcome = _passes(items["total-08"], f"всього={len(ids)}, з роком={with_year}")
    assert ok, outcome


def test_declared_chain_length_matches_the_documents(box, items):
    for iid, doc in DOC_OF.items():
        ids, _ = _walk(box, doc)
        assert items[iid].chain_len == len(ids) + 2, (
            f"{iid}: заявлено {items[iid].chain_len}, документ дає {len(ids) + 2}")


def test_the_set_grades_length(items):
    assert sorted({i.chain_len for i in items.values()}) == [5, 7, 10]


def test_unknown_document_and_paragraph_are_loud(box):
    nowhere = box.call(ToolCall(tool="список_абзаців", args={"документ": "літопис-9"}))
    assert nowhere.ok and nowhere.value["відомо"] is False
    nothing = box.call(ToolCall(tool="абзац", args={"ідентифікатор": "аб-9-99"}))
    assert nothing.ok and nothing.value["відомо"] is False


def test_crafts_are_found_through_lemmas_not_substrings():
    """У тексті «ткалю», у переліку «ткаля» — вісь U6 всередині оракула."""
    assert "ткаля" in crafts_in("Про ткалю Одарку Кривоніс писано")
    assert "ткаля" not in "Про ткалю".casefold()
