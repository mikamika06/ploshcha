import json

import pytest

from ploshcha_sim.adapters import FakeLlm, FakeToolbox, InMemoryTrace, PresetEffort
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.adapters.tools_lexis import LEXIS_TOOLS
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.agents.graph import FANOUT_NOTE, MISSING_HEADER, NO_FANOUT_NOTE, AgentGraph
from ploshcha_sim.domain.finding import Finding, merge_evidence, merge_outcome, missing_tasks
from ploshcha_sim.domain.graph import child_budget, plan_split, quoted_items
from ploshcha_sim.domain.task import Budget, TaskResult
from ploshcha_sim.ports.agent import AgentPort

TASK4 = "Поясни слова «ботей», «будз», «ватра» і «мешти» одним реченням кожне."


class Stub(AgentPort):
    """Дитина без моделі: віддає заготовлений результат за підзадачею."""

    def __init__(self, table: dict[str, TaskResult], default: TaskResult | None = None):
        self.table = table
        self.default = default
        self.seen: list[tuple[str, int]] = []

    def run(self, task, seed=0, budget=None):
        self.seen.append((task, budget.max_steps if budget else 0))
        # Матчимо СЛОВО ФОКУСА, а не весь текст: підзадача містить усю початкову задачу, тому
        # пошук підрядком віддавав заготовку першого слова всім чотирьом дітям.
        focus = task.split("ЛИШЕ на «")[-1].split("»")[0] if "ЛИШЕ на «" in task else task
        for key, value in self.table.items():
            if key == focus:
                return value.model_copy(deep=True)
        if self.default is None:
            raise AssertionError(f"немає заготовки для: {task[:60]}")
        return self.default.model_copy(deep=True)


def _ok(text, evidence=True, steps=3, tokens=100):
    return TaskResult(answer=text, accepted=True, outcome="answer", evidence=evidence,
                      steps=steps, tokens=tokens)


def _absent(steps=3, tokens=80):
    return TaskResult(answer="даних немає", accepted=True, outcome="abstain", evidence=False,
                      steps=steps, tokens=tokens)


def _graph(child, **kw):
    return AgentGraph(lambda budget: child, run_id="g", **kw)


# ── домен ───────────────────────────────────────────────────────────────────────────────────────
def test_items_are_taken_from_quotes_deterministically():
    assert quoted_items(TASK4) == ["ботей", "будз", "ватра", "мешти"]
    assert quoted_items("те саме «слово» і ще раз «слово»") == ["слово"]
    assert quoted_items("немає лапок узагалі") == []


def test_fanout_is_declined_when_there_is_nothing_to_split():
    assert plan_split("одне питання", Budget(), template="{task}{item}") is None
    assert plan_split("лише «одне»", Budget(), template="{task}{item}") is None


def test_depth_gate_stands_before_parsing():
    assert plan_split(TASK4, Budget(), template="{task}{item}", depth=3, max_depth=2) is None


def test_the_budget_is_divided_not_copied():
    """N дітей із батьковою стелею означали б, що «масштаб» — це просто «дорожче»."""
    parent = Budget(max_steps=16, max_tokens=8000)
    child = child_budget(parent, 4)
    assert child.max_steps == 4 and child.max_tokens == 2000


def test_a_child_never_gets_a_budget_too_small_to_act():
    assert child_budget(Budget(max_steps=3), 8).max_steps == 2


def test_spent_tokens_are_not_handed_out_twice():
    parent = Budget(max_tokens=1000)
    parent.tokens_used = 600
    assert child_budget(parent, 2).max_tokens == 200


@pytest.mark.parametrize("kinds,want", [
    ((True, True), True), ((False, False), False), ((False, True), True),
    ((None, None), None), ((None, False), False), ((None, True), True),
])
def test_evidence_merges_by_the_same_rule_as_a_single_run(kinds, want):
    """Підйом на рівень вище не має змінювати правило: інакше «даних немає» стає «не знаю, чи є»."""
    findings = [Finding(task="t", evidence=e) for e in kinds]
    assert merge_evidence(findings) is want


def test_one_usable_child_outweighs_two_refusals():
    findings = [Finding(task="a", claim="зміст", kind="answer"),
                Finding(task="b", kind="abstain"), Finding(task="c", kind="abstain")]
    assert merge_outcome(findings) == "answer"
    assert missing_tasks(findings) == ["b", "c"]


def test_all_refusals_stay_a_refusal_not_a_failure():
    assert merge_outcome([Finding(task="a", kind="abstain"),
                          Finding(task="b", kind="abstain")]) == "abstain"


def test_an_empty_claim_is_not_usable():
    assert not Finding(task="a", claim="   ", kind="answer").usable
    assert merge_outcome([Finding(task="a", claim="", kind="answer")]) == "failure"


# ── граф ────────────────────────────────────────────────────────────────────────────────────────
def test_the_graph_fans_out_one_child_per_quoted_item():
    child = Stub({}, default=_ok("тлумачення"))
    result = _graph(child).run(TASK4, budget=Budget(max_steps=16))
    assert len(child.seen) == 4
    assert FANOUT_NOTE in result.notes and "width=4" in result.notes
    assert result.answer.count("—") == 4


def test_without_fanout_the_task_goes_to_a_single_child_untouched():
    child = Stub({}, default=_ok("проста відповідь"))
    result = _graph(child).run("одне питання без лапок")
    assert child.seen[0][0] == "одне питання без лапок"
    assert NO_FANOUT_NOTE in result.notes and result.answer == "проста відповідь"


def test_the_missing_children_are_named_in_the_synthesis():
    """Борг K9: якщо дірку не назвати, супељвізор видасть часткове за повне."""
    child = Stub({"ботей": _ok("отара"), "будз": _absent(), "ватра": _absent(),
                  "мешти": _ok("черевики")})
    result = _graph(child).run(TASK4, budget=Budget(max_steps=16))
    assert MISSING_HEADER in result.answer and "2 з 4" in result.answer
    assert sum(1 for n in result.notes if n.startswith("missing=")) == 2


def test_the_child_evidence_state_reaches_the_parent():
    child = Stub({}, default=_absent())
    result = _graph(child).run(TASK4, budget=Budget(max_steps=16))
    assert result.evidence is False and result.outcome == "abstain"
    assert all(entry["found"] is False for entry in result.scratch)


def test_a_single_find_lifts_the_merged_evidence():
    child = Stub({"ботей": _ok("отара")}, default=_absent())
    result = _graph(child).run(TASK4, budget=Budget(max_steps=16))
    assert result.evidence is True and result.outcome == "answer"


def test_children_costs_are_summed_not_lost():
    child = Stub({}, default=_ok("х", steps=2, tokens=50))
    result = _graph(child).run(TASK4, budget=Budget(max_steps=16))
    assert result.steps == 8 and result.tokens == 200
    assert result.tokens_by_lane == {"child": 200}


def test_parallel_output_is_ordered_by_index_not_by_completion():
    """Той самий seed мусить давати той самий вивід, інакше `pass^k` міряє шум планувальника."""
    import time

    class Slow(AgentPort):
        def run(self, task, seed=0, budget=None):
            word = task.split("«")[-1].split("»")[0]
            time.sleep(0.02 if word == "ботей" else 0.0)
            return _ok(f"[{word}]")

    result = _graph(Slow(), workers=4).run(TASK4, budget=Budget(max_steps=16))
    assert result.answer.index("[ботей]") < result.answer.index("[будз]")
    serial = _graph(Slow(), workers=1).run(TASK4, budget=Budget(max_steps=16))
    assert serial.answer == result.answer


def test_too_wide_a_split_falls_back_instead_of_exploding():
    task = " ".join(f"«с{i}»" for i in range(12))
    child = Stub({}, default=_ok("х"))
    result = _graph(child, max_width=6).run(task)
    assert NO_FANOUT_NOTE in result.notes and len(child.seen) == 1


def test_the_trace_is_hierarchical():
    trace = InMemoryTrace()
    child = Stub({}, default=_ok("х"))
    AgentGraph(lambda b: child, trace=trace, run_id="g").run(TASK4, budget=Budget(max_steps=16))
    spans = [r for r in trace.records if r.agent == "subagent"]
    assert len(spans) == 4
    assert {r.parent_run_id for r in spans} == {"g"}
    assert sorted(r.span for r in spans) == [f"g/{i}" for i in range(4)]


def test_the_graph_composes_with_a_real_orchestrator_child():
    """Дитина — звичайний оркестратор, тож граф не потребує власної мови."""
    def make(budget):
        script = [json.dumps({"tool": "словник", "слово": "ботей"}, ensure_ascii=False),
                  json.dumps({"tool": "final_answer", "text": "отара"}, ensure_ascii=False)]
        llm = FakeLlm(script * 4, model="m")
        return Orchestrator(single_model_router(llm, lane="lapa"), PresetEffort(),
                            FakeToolbox(tools=LEXIS_TOOLS), verifier=False)

    result = AgentGraph(make, run_id="g").run(TASK4, budget=Budget(max_steps=16))
    assert result.outcome == "answer" and result.answer.count("отара") == 4
    assert result.evidence is True
