from ploshcha_sim.adapters import FakeLlm, FakeToolbox, InMemoryTrace, PresetEffort, single_model_router
from ploshcha_sim.adapters.memory_notebook import DROPPED_SHOWN, NotebookMemory
from ploshcha_sim.adapters.planner_skeleton import SkeletonPlanner
from ploshcha_sim.agents.orchestrator import VERBATIM_WINDOW, Orchestrator
from ploshcha_sim.domain.task import Budget

MAZEPA = '{"tool": "lookup_fact", "entity": "Іван Мазепа"}'
KRUTY = '{"tool": "lookup_fact", "entity": "Битва під Крутами"}'
SHEVCHENKO = '{"tool": "lookup_fact", "entity": "Тарас Шевченко"}'
CALC = '{"tool": "calc", "expr": "200*3"}'
DATE = '{"tool": "check_date", "year": 1918, "event": "Битва під Крутами"}'
FINAL = '{"tool": "final_answer", "text": "готово"}'


def _orch(responses, **kw):
    llm = FakeLlm(responses, strict=True)
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                        verifier=False, **kw)
    return orch, llm


def test_notebook_dedups_identical_notes():
    nb = NotebookMemory()
    assert nb.note("той самий текст", 0) is not None
    assert nb.note("той самий текст", 1) is None
    assert len(nb.items) == 1


def test_notebook_ignores_blank_notes():
    nb = NotebookMemory()
    assert nb.note("   ", 0) is None
    assert nb.items == []


def test_notebook_importance_depends_on_sort():
    nb = NotebookMemory()
    fact = nb.note("факт", 0, sort="fact")
    err = nb.note("помилка", 0, sort="error")
    assert fact.importance > err.importance


def test_recall_reports_candidates_and_dropped():
    nb = NotebookMemory(k=1)
    for i in range(5):
        nb.note(f"запис про Мазепу номер {i}", i)
    recall = nb.recall("Мазепа", 5)
    assert recall.candidates == 5
    assert len(recall.hits) == 1
    assert len(recall.dropped) == DROPPED_SHOWN
    trace = recall.as_trace()
    assert trace["candidates"] == 5
    assert len(trace["retrieved"]) == 1 and len(trace["dropped"]) == DROPPED_SHOWN


def test_recall_keeps_score_components_separate():
    nb = NotebookMemory()
    nb.note("Іван Мазепа був гетьманом", 0)
    hit = nb.recall("Мазепа", 3).hits[0]
    assert set(hit.model_dump()) >= {"recency", "importance", "relevance", "total"}
    assert hit.relevance > 0, "один запис не має давати нульову релевантність"
    assert hit.importance > 0
    assert hit.total > hit.relevance, "склад скору лишається сумою з вагами"


def test_single_note_keeps_signal_without_minmax():
    one = NotebookMemory().recall("Мазепа", 0)
    nb = NotebookMemory()
    nb.note("Іван Мазепа був гетьманом", 0)
    solo = nb.recall("Мазепа", 0).hits[0]
    nb.note("Битва під Крутами відбулася 1918", 1)
    pair = nb.recall("Мазепа", 1).hits[0]
    assert one.hits == []
    assert solo.relevance == pair.relevance, "скор не залежить від розміру блокнота"


def test_recall_on_empty_notebook_is_safe():
    recall = NotebookMemory().recall("будь-що", 0)
    assert recall.hits == [] and recall.candidates == 0 and recall.dropped == []


def test_notebook_off_emits_no_memory_records():
    trace = InMemoryTrace()
    orch, _ = _orch([FINAL], trace=trace)
    orch.run("t", seed=1, budget=Budget(max_steps=3))
    assert all(r.stage not in ("mem_read", "mem_write") for r in trace.records)


def test_notebook_on_emits_read_and_write():
    trace = InMemoryTrace()
    orch, _ = _orch([MAZEPA, FINAL], trace=trace, notebook=NotebookMemory)
    orch.run("хто такий Іван Мазепа", seed=1, budget=Budget(max_steps=4))
    reads = [r for r in trace.records if r.stage == "mem_read"]
    writes = [r for r in trace.records if r.stage == "mem_write"]
    assert len(reads) == 2, "recall перед кожним кроком"
    assert len(writes) == 1, "запис лише після успішного інструмента"
    assert writes[0].parsed["sort"] == "fact"
    assert "candidates" in reads[0].parsed and "dropped" in reads[0].parsed


def test_write_record_only_when_actually_stored():
    trace = InMemoryTrace()
    orch, _ = _orch([MAZEPA, MAZEPA, FINAL], trace=trace, notebook=NotebookMemory, recovery=True)
    orch.run("t", seed=1, budget=Budget(max_steps=5))
    writes = [r for r in trace.records if r.stage == "mem_write"]
    sorts = [w.parsed["sort"] for w in writes]
    assert sorts == ["fact", "incident"], "дубль факту не пишеться, інцидент пишеться"


def test_incident_lands_in_the_notebook_text():
    trace = InMemoryTrace()
    orch, _ = _orch([MAZEPA, MAZEPA, FINAL], trace=trace, notebook=NotebookMemory, recovery=True)
    orch.run("t", seed=1, budget=Budget(max_steps=5))
    incident = [r for r in trace.records if r.stage == "mem_write"][1]
    assert "збій dup_call" in incident.parsed["text"]


def test_verbatim_window_trims_old_steps_from_the_prompt():
    orch, llm = _orch([MAZEPA, KRUTY, SHEVCHENKO, FINAL], notebook=NotebookMemory)
    orch.run("три факти", seed=1, budget=Budget(max_steps=6))
    last_prompt = llm.calls[3]["prompt"]
    assert last_prompt.count("Виклик:") == VERBATIM_WINDOW
    assert "Пригадане:" in last_prompt


def test_recalled_items_carry_anchors():
    orch, llm = _orch([MAZEPA, KRUTY, FINAL], notebook=NotebookMemory)
    orch.run("t", seed=1, budget=Budget(max_steps=5))
    prompt = llm.calls[2]["prompt"]
    assert "n0 ·" in prompt or "n1 ·" in prompt


def test_notebook_off_prompt_is_unchanged():
    a, llm_a = _orch([MAZEPA, KRUTY, FINAL])
    a.run("t", seed=1, budget=Budget(max_steps=5))
    b, llm_b = _orch([MAZEPA, KRUTY, FINAL])
    b.run("t", seed=1, budget=Budget(max_steps=5))
    assert [c["prompt"] for c in llm_a.calls] == [c["prompt"] for c in llm_b.calls]
    assert all("Пригадане:" not in c["prompt"] for c in llm_a.calls)


def test_notebook_is_task_scoped_not_shared_between_runs():
    orch, llm = _orch([MAZEPA, FINAL, MAZEPA, FINAL], notebook=NotebookMemory)
    orch.run("перша", seed=1, budget=Budget(max_steps=3))
    orch.run("друга", seed=1, budget=Budget(max_steps=3))
    assert "Пригадане:" not in llm.calls[2]["prompt"], "другий прогін починає з чистого блокнота"


def test_notebook_run_is_deterministic():
    def run_once():
        trace = InMemoryTrace()
        orch, _ = _orch([MAZEPA, KRUTY, FINAL], trace=trace, notebook=NotebookMemory)
        orch.run("t", seed=4, budget=Budget(max_steps=5))
        return [r.model_dump(exclude={"latency_ms"}) for r in trace.records]

    assert run_once() == run_once()


def test_recall_query_uses_plan_goal_and_last_result():
    trace = InMemoryTrace()
    llm = FakeLlm([DATE, CALC, FINAL], strict=True)
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                        planner=SkeletonPlanner(), verifier=False, notebook=NotebookMemory,
                        trace=trace)
    r = orch.run("перевір дату і порахуй", seed=1, budget=Budget(max_steps=5))
    assert r.answer == "готово"
    reads = [x for x in trace.records if x.stage == "mem_read"]
    assert len(reads) == 3
