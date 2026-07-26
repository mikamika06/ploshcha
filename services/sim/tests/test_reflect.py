"""Reflect: порог, дві фази, і головне — чи цитує модель РЕАЛЬНІ памʼяті."""

import json

from ploshcha_sim.adapters import FakeLlm, InMemoryTrace
from ploshcha_sim.agents import reflect, should_reflect
from ploshcha_sim.agents.reflect import (
    EMPTY,
    TRUNCATED,
    EVIDENCE_INVALID,
    REFLECTION_FLOOR,
    SCHEMA_INVALID,
    accumulated_importance,
    last_reflection_tick,
)
from ploshcha_sim.domain import MemoryItem


def observations(world, agent_id: str, n: int, importance: int = 5):
    world.agents[agent_id].memory = [
        MemoryItem(id=f"m{i}", tick=i, kind="observation", text=f"подія {i}", importance=importance)
        for i in range(n)
    ]


QUESTIONS = json.dumps({"questions": ["Чи можу я довіряти Оксані?"]})


def insights(evidence: list) -> str:
    return json.dumps({"insights": [{"text": "Оксана дбає про мене", "evidence": evidence}]})


# ── порог ────────────────────────────────────────────────────────────────────


def test_threshold_gates_reflection(world):
    observations(world, "koval", 3, importance=5)
    assert not should_reflect(world.agents["koval"], threshold=25)
    assert should_reflect(world.agents["koval"], threshold=10)


def test_accumulation_restarts_after_a_reflection(world):
    observations(world, "koval", 6, importance=5)
    agent = world.agents["koval"]
    assert accumulated_importance(agent, last_reflection_tick(agent)) == 30
    agent.memory.append(
        MemoryItem(id="r0", tick=4, kind="reflection", text="висновок", importance=7)
    )
    # рахуються лише памʼяті ПІСЛЯ останньої рефлексії
    assert accumulated_importance(agent, last_reflection_tick(agent)) == 5


def test_no_reflection_yet_means_everything_counts(world):
    observations(world, "koval", 2, importance=4)
    assert last_reflection_tick(world.agents["koval"]) == -1
    assert accumulated_importance(world.agents["koval"], -1) == 8


# ── дві фази ─────────────────────────────────────────────────────────────────


def test_happy_path_produces_reflection_with_real_evidence(world):
    observations(world, "koval", 4)
    llm = FakeLlm([QUESTIONS, insights([1, 2])])
    r = reflect(world, "koval", llm)
    assert r.questions == ["Чи можу я довіряти Оксані?"]
    assert len(r.items) == 1 and r.items[0].kind == "reflection"
    assert r.items[0].evidence and all(e.startswith("m") for e in r.items[0].evidence)
    assert r.dropped_citations == 0 and r.reject_reason is None


def test_reflection_importance_has_a_floor(world):
    observations(world, "koval", 4)
    r = reflect(world, "koval", FakeLlm([QUESTIONS, insights([1])]))
    assert r.items[0].importance >= REFLECTION_FLOOR


def test_provenance_marks_the_agent_as_author(world):
    observations(world, "koval", 4)
    r = reflect(world, "koval", FakeLlm([QUESTIONS, insights([1])]))
    assert r.items[0].provenance == "agent:koval"


def test_fabricated_citation_is_dropped_and_counted(world):
    """Номер поза діапазоном = галюцинація посилання. Це метрика, не крах."""
    observations(world, "koval", 3)
    r = reflect(world, "koval", FakeLlm([QUESTIONS, insights([1, 99])]))
    assert r.dropped_citations == 1
    assert r.reject_reason == EVIDENCE_INVALID
    assert len(r.items[0].evidence) == 1


def test_non_integer_citation_is_dropped(world):
    observations(world, "koval", 3)
    r = reflect(world, "koval", FakeLlm([QUESTIONS, insights(["перша"])]))
    assert r.dropped_citations == 1 and r.items[0].evidence == []


def test_broken_questions_stop_the_step(world):
    observations(world, "koval", 3)
    r = reflect(world, "koval", FakeLlm(["не JSON"]))
    assert r.items == [] and r.reject_reason == SCHEMA_INVALID and r.llm_calls == 1


def test_broken_insights_yield_no_items_but_keep_questions(world):
    observations(world, "koval", 3)
    r = reflect(world, "koval", FakeLlm([QUESTIONS, "не JSON"]))
    assert r.questions and r.items == [] and r.reject_reason == SCHEMA_INVALID


def test_empty_memory_is_reported_not_crashed(world):
    r = reflect(world, "koval", FakeLlm([]))
    assert r.reject_reason == EMPTY and r.llm_calls == 0


def test_question_count_is_capped(world):
    observations(world, "koval", 4)
    many = json.dumps({"questions": ["а", "б", "в", "г", "д"]})
    llm = FakeLlm([many] + [insights([1])] * 5)
    r = reflect(world, "koval", llm, questions=2)
    assert len(r.questions) == 2 and r.llm_calls == 3  # 1 питання + 2 висновки


def test_usage_is_aggregated_across_calls(world):
    observations(world, "koval", 4)
    llm = FakeLlm([QUESTIONS, insights([1])])
    r = reflect(world, "koval", llm)
    assert r.llm_calls == 2 and r.usage.total > 0


def test_both_phases_and_recall_appear_in_trace(world):
    observations(world, "koval", 4)
    trace = InMemoryTrace()
    reflect(world, "koval", FakeLlm([QUESTIONS, insights([1])]), trace=trace, run_id="r1")
    stages = [r.stage for r in trace.records]
    assert stages == ["reflect.questions", "recall", "reflect.insights"]
    assert all(r.run_id == "r1" for r in trace.records)


def test_insight_schema_is_grammar_compilable(world):
    observations(world, "koval", 4)
    llm = FakeLlm([QUESTIONS, insights([1])])
    reflect(world, "koval", llm)
    item = llm.calls[1]["schema"]["properties"]["insights"]["items"]
    assert item["required"] == ["text", "evidence"]
    assert item["additionalProperties"] is False


def test_window_limits_what_goes_into_the_prompt(world):
    observations(world, "koval", 20)
    llm = FakeLlm([QUESTIONS, insights([1])])
    reflect(world, "koval", llm, window=3)
    assert llm.calls[0]["prompt"].count("- подія") == 3


# ── дедуплікація й обрізання ─────────────────────────────────────────────────


def test_near_duplicate_insights_are_dropped(world):
    """Різні питання тягнуть ті самі згадки — модель повторює думку майже дослівно."""
    observations(world, "koval", 4)
    twice = json.dumps({
        "insights": [
            {"text": "Свирид попросив позичити сокиру", "evidence": [1]},
            {"text": "Свирид попросив позичити сокиру.", "evidence": [1]},
        ]
    })
    r = reflect(world, "koval", FakeLlm([QUESTIONS, twice]))
    assert len(r.items) == 1 and r.duplicates_dropped == 1


def test_distinct_insights_are_kept(world):
    observations(world, "koval", 4)
    both = json.dumps({
        "insights": [
            {"text": "Свирид хворий", "evidence": [1]},
            {"text": "У неділю весілля в Ганни", "evidence": [2]},
        ]
    })
    r = reflect(world, "koval", FakeLlm([QUESTIONS, both]))
    assert len(r.items) == 2 and r.duplicates_dropped == 0


def test_dedup_can_be_switched_off(world):
    observations(world, "koval", 4)
    twice = json.dumps({
        "insights": [
            {"text": "Свирид хворий", "evidence": [1]},
            {"text": "Свирид хворий", "evidence": [1]},
        ]
    })
    r = reflect(world, "koval", FakeLlm([QUESTIONS, twice]), dedup_threshold=1.01)
    assert len(r.items) == 2


def test_truncated_questions_are_not_a_format_error(world):
    observations(world, "koval", 3)
    llm = FakeLlm(['{"questions":["Чи зможу я'], finish_reason="length")
    r = reflect(world, "koval", llm)
    assert r.reject_reason == TRUNCATED and r.items == []
