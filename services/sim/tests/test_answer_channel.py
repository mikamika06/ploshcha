from evalkit.prompts import resolve
from ploshcha_sim.adapters import FakeLlm, FakeToolbox, InMemoryTrace, PresetEffort, single_model_router
from ploshcha_sim.adapters.planner_skeleton import SkeletonPlanner
from ploshcha_sim.agents.orchestrator import Orchestrator
from ploshcha_sim.domain.task import Budget

LOOKUP = '{"tool": "lookup_fact", "entity": "Іван Мазепа"}'
FINAL = '{"tool": "final_answer", "text": "готово"}'
FREE = "Іван Мазепа був гетьманом Війська Запорозького."
INSTR = resolve("answer/plain").render_system()


def _orch(responses, **kw):
    llm = FakeLlm(responses, strict=True)
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                        verifier=False, **kw)
    return orch, llm


def test_default_channel_is_schema_and_unchanged():
    a, llm_a = _orch([LOOKUP, FINAL], planner=SkeletonPlanner())
    ra = a.run("t", seed=1, budget=Budget(max_steps=5))
    b, llm_b = _orch([LOOKUP, FINAL], planner=SkeletonPlanner())
    rb = b.run("t", seed=1, budget=Budget(max_steps=5))
    assert ra.model_dump() == rb.model_dump()
    assert ra.answer == "готово"
    assert all(c["structured"] for c in llm_a.calls), "усі кроки під схемою"


def test_text_channel_answers_without_the_action_schema():
    orch, llm = _orch([LOOKUP, FREE], planner=SkeletonPlanner(),
                      answer_channel="text", answer_instruction=INSTR)
    r = orch.run("хто такий Іван Мазепа", seed=1, budget=Budget(max_steps=5))
    assert r.answer == FREE
    assert llm.calls[0]["structured"] is True, "крок здобуття — під схемою"
    assert llm.calls[1]["structured"] is False, "крок відповіді — вільний текст"
    assert llm.calls[1]["schema"] is None


def test_text_channel_costs_no_extra_call():
    schema_orch, schema_llm = _orch([LOOKUP, FINAL], planner=SkeletonPlanner())
    schema_orch.run("t", seed=1, budget=Budget(max_steps=5))
    text_orch, text_llm = _orch([LOOKUP, FREE], planner=SkeletonPlanner(),
                                answer_channel="text", answer_instruction=INSTR)
    r = text_orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert len(text_llm.calls) == len(schema_llm.calls) == 2
    assert r.steps == 2, "інваріант: один виклик = один крок"


def test_answer_step_gets_its_own_instruction_line():
    orch, llm = _orch([LOOKUP, FREE], planner=SkeletonPlanner(),
                      answer_channel="text", answer_instruction=INSTR)
    orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert llm.calls[0]["prompt"].rstrip().endswith("final_answer):")
    assert llm.calls[1]["prompt"].rstrip().endswith(INSTR)


def test_text_channel_reacts_to_the_fact_of_final_answer_not_to_the_plan():
    """Модель завершує коли хоче, а не коли план дозволяє — тригер мусить бути на факті."""
    orch, llm = _orch([FINAL, FREE], answer_channel="text", answer_instruction=INSTR)
    r = orch.run("t", seed=1, budget=Budget(max_steps=4))
    assert r.answer == FREE, "текст із final_answer відкинуто, відповідь перегенеровано вільно"
    assert llm.calls[0]["structured"] is True
    assert llm.calls[1]["structured"] is False


def test_early_finish_costs_one_extra_call():
    schema_orch, schema_llm = _orch([FINAL])
    schema_orch.run("t", seed=1, budget=Budget(max_steps=4))
    text_orch, text_llm = _orch([FINAL, FREE], answer_channel="text",
                                answer_instruction=INSTR)
    r = text_orch.run("t", seed=1, budget=Budget(max_steps=4))
    assert len(schema_llm.calls) == 1
    assert len(text_llm.calls) == 2, "коли модель завершує одразу — платимо +1 виклик"
    assert r.steps == 2, "інваріант: кожен виклик моделі інкрементує steps"


def test_free_answer_step_is_traced_as_synthesize():
    trace = InMemoryTrace()
    orch, _ = _orch([FINAL, FREE], trace=trace, answer_channel="text",
                    answer_instruction=INSTR)
    orch.run("t", seed=1, budget=Budget(max_steps=4))
    last = trace.records[-1]
    assert last.stage == "synthesize" and last.ablation["tier"] == "none"


def test_answer_step_is_traced_with_tier_none():
    trace = InMemoryTrace()
    orch, _ = _orch([LOOKUP, FREE], planner=SkeletonPlanner(), trace=trace,
                    answer_channel="text", answer_instruction=INSTR)
    orch.run("t", seed=1, budget=Budget(max_steps=5))
    last = trace.records[-1]
    assert last.ablation["tier"] == "none", "крок відповіді без схеми — це видно в трасі"
    assert last.parsed is None and last.schema_valid is False


def test_text_channel_is_deterministic():
    def once():
        orch, _ = _orch([LOOKUP, FREE], planner=SkeletonPlanner(),
                        answer_channel="text", answer_instruction=INSTR)
        return orch.run("t", seed=3, budget=Budget(max_steps=5)).model_dump()

    assert once() == once()


def test_answer_step_strips_whitespace():
    orch, _ = _orch([LOOKUP, "  відповідь із пробілами  \n"], planner=SkeletonPlanner(),
                    answer_channel="text", answer_instruction=INSTR)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.answer == "відповідь із пробілами"


def test_gathering_still_uses_tools_under_text_channel():
    orch, _ = _orch([LOOKUP, FREE], planner=SkeletonPlanner(),
                    answer_channel="text", answer_instruction=INSTR)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert [x["call"]["tool"] for x in r.scratch] == ["lookup_fact"]
    assert "Гетьман" in str(r.scratch[0]["result"])
