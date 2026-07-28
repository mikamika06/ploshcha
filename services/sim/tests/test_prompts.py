import re

import pytest

from evalkit.prompts import REGISTRY, PromptVariant, TableRow, load_prompts, resolve

LATIN = re.compile(r"[A-Za-z][A-Za-z_]*")
TOOL_NAMES = {"check_date", "year", "event", "lookup_fact", "entity", "calc", "expr",
              "final_answer", "text"}


@pytest.fixture(scope="module")
def registry():
    return load_prompts(REGISTRY)


def test_registry_loads_and_ids_are_unique(registry):
    assert len(registry) >= 7
    assert all(pid.startswith("agent/") for pid in registry)


def test_every_sha_matches_its_text(registry):
    for pid, variant in registry.items():
        assert variant.sha256 == variant.digest(), f"{pid}: текст розійшовся зі sha"


def test_exactly_one_frozen_orchestrator_prompt(registry):
    frozen = [v for v in registry.values() if v.status == "frozen" and v.slot == "orchestrator"]
    assert len(frozen) == 1
    assert frozen[0].id == "agent/v2"


def test_frozen_and_candidate_variants_pass_all_invariants(registry):
    for pid, variant in registry.items():
        if variant.status == "rejected":
            continue
        assert variant.violations() == [], f"{pid}: {variant.violations()}"


def test_validator_catches_the_v0_deadlock(registry):
    v0 = registry["agent/v0"]
    problems = v0.violations()
    assert any(p.startswith("deadlock:(True") for p in problems), problems
    assert "deadlock" in (v0.defect or "")


def test_validator_catches_the_v1_premature_finish(registry):
    v1 = registry["agent/v1"]
    problems = v1.violations()
    assert any(p.startswith("premature_finish") for p in problems), problems
    assert "premature_finish" in (v1.defect or "")


def test_rejected_variants_are_kept_with_a_defect(registry):
    rejected = [v for v in registry.values() if v.status == "rejected"]
    assert len(rejected) >= 3
    for v in rejected:
        assert v.defect, f"{v.id}: відкинутий варіант мусить пояснювати чому"
        assert v.defect_class in ("structural", "semantic"), v.id


def test_structural_defects_are_caught_by_the_validator(registry):
    for v in registry.values():
        if v.defect_class == "structural":
            assert v.violations(), f"{v.id}: структурний дефект мусить ловитись валідатором"


def test_semantic_defects_document_the_validator_blind_spot(registry):
    """Текст може пройти всі структурні перевірки й не реалізовувати власну таблицю."""
    semantic = [v for v in registry.values() if v.defect_class == "semantic"]
    assert semantic, "має бути хоч один задокументований приклад сліпої плями"
    for v in semantic:
        assert v.violations() == [], f"{v.id}: семантичний дефект структурно НЕ видно — у цьому й суть"
        assert "заміряно" in (v.defect or "").lower() or "0.875" in (v.defect or ""), \
            f"{v.id}: семантичний дефект мусить спиратись на вимір, не на думку"


def test_no_latin_outside_tool_names(registry):
    for pid, variant in registry.items():
        words = set(LATIN.findall(variant.head + " " + variant.tail))
        assert words <= TOOL_NAMES, f"{pid}: чужа латиниця {sorted(words - TOOL_NAMES)}"


def test_tail_requires_a_declared_rule():
    bad = PromptVariant(id="x", head="h", tail="нагадування", rules=["no-repeat"],
                        placement="both", table=_full_table())
    assert "tail_without_rule" in bad.violations()
    good = bad.model_copy(update={"tail_rule": "no-repeat"})
    assert good.violations() == []
    undeclared = good.model_copy(update={"tail_rule": "no-such-rule"})
    assert "tail_rule_undeclared:no-such-rule" in undeclared.violations()


def test_tail_without_placement_is_flagged():
    v = PromptVariant(id="x", head="h", tail="t", tail_rule="r", rules=["r"],
                      placement="head", table=_full_table())
    assert "tail_set_but_placement_head" in v.violations()


def test_uncovered_state_is_flagged():
    rows = _full_table()[:-1]
    v = PromptVariant(id="x", head="h", rules=["r"], table=rows)
    assert any(p.startswith("uncovered_state") for p in v.violations())


def test_contradiction_and_unknown_action_are_flagged():
    rows = _full_table()
    rows[0] = rows[0].model_copy(update={"allow": ["final_answer", "teleport"],
                                         "deny": ["final_answer"]})
    v = PromptVariant(id="x", head="h", rules=["r"], table=rows)
    problems = v.violations()
    assert any(p.startswith("contradiction") for p in problems)
    assert any(p.startswith("unknown_action") for p in problems)


def test_call_after_completion_is_flagged():
    rows = _full_table()
    rows[0] = rows[0].model_copy(update={"allow": ["final_answer", "new_call"]})
    v = PromptVariant(id="x", head="h", rules=["r"], table=rows)
    assert any(p.startswith("call_after_completion") for p in v.violations())


def test_duplicate_state_is_flagged():
    rows = _full_table()
    rows.append(rows[0])
    v = PromptVariant(id="x", head="h", rules=["r"], table=rows)
    assert any(p.startswith("duplicate_state") for p in v.violations())


def test_resolve_finds_and_rejects():
    assert resolve("agent/v2").status == "frozen"
    with pytest.raises(KeyError):
        resolve("agent/nope")


def test_frozen_prompt_text_is_the_one_measured(registry):
    head = registry["agent/v2"].head
    assert "НЕ повторюй виклик" in head
    assert "ВСІ частини задачі" in head
    assert "лише після перевірки" not in head


def _full_table():
    return [
        TableRow(a=True, allow=["final_answer"], deny=["new_call"]),
        TableRow(a=False, b=True, allow=["other_call", "final_if_no_alt"], deny=["repeat"]),
        TableRow(a=False, c=True, allow=["other_args", "final_if_no_alt"], deny=["repeat"]),
        TableRow(a=False, allow=["new_call"], deny=["final_answer"]),
    ]
