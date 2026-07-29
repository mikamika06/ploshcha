from ploshcha_sim.adapters import (
    FakeLlm,
    FakeToolbox,
    InMemoryTrace,
    PresetEffort,
    profile_router,
    single_model_router,
)
from ploshcha_sim.adapters.router_profile import ProfileRouter
from ploshcha_sim.agents.orchestrator import Orchestrator
from ploshcha_sim.domain.task import Budget
from ploshcha_sim.ports.router import escalated

DUP = '{"tool": "lookup_fact", "entity": "Іван Мазепа"}'
FINAL = '{"tool": "final_answer", "text": "Гетьман Війська Запорозького"}'
NEAR = '{"tool": "lookup_fact", "entity": "Мазепа"}'
JUDGE = '{"accepted": true, "reason": "ok"}'


def _orch(responses, *, recovery, verifier=False, trace=None, strict=True):
    llm = FakeLlm(responses, strict=strict)
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                        verifier=verifier, trace=trace, recovery=recovery)
    return orch, llm


def test_without_recovery_the_incident_is_still_RECORDED():
    """Дефект 18: раніше тут стояло `incidents == []`, тобто тест ФІКСУВАВ сліпоту як норму.

    Спостереження не залежить від втручання: без драбини збій усе одно видно, просто ніхто його
    не лікує. Плюс тихий вихід із циклу тепер називає себе (`no_final_answer`).
    """
    orch, llm = _orch([DUP, DUP], recovery=False)
    r = orch.run("хто такий Іван Мазепа", seed=1, budget=Budget(max_steps=5))
    assert r.answer is None and r.degraded and not r.partial
    assert r.incidents == ["dup_call", "no_final_answer"]
    assert r.notes == [], "нічого не лікували — драбина вимкнена"
    assert len(llm.calls) == 2


def test_duplicate_with_recovery_nudges_and_finishes():
    orch, llm = _orch([DUP, DUP, FINAL], recovery=True)
    r = orch.run("хто такий Іван Мазепа", seed=1, budget=Budget(max_steps=5))
    assert r.answer == "Гетьман Війська Запорозького"
    assert not r.degraded
    assert r.incidents == ["dup_call"]
    assert "Заверши через final_answer" in llm.calls[2]["prompt"]
    assert "Підказка:" in llm.calls[2]["prompt"]


def test_hint_is_consumed_after_one_step():
    orch, llm = _orch([DUP, DUP, DUP, FINAL], recovery=True)
    orch.run("t", seed=1, budget=Budget(max_steps=6))
    assert llm.calls[2]["prompt"].count("Підказка:") == 1
    assert llm.calls[3]["prompt"].count("Підказка:") == 1


def test_near_duplicate_detected_only_with_recovery():
    orch, llm = _orch([DUP, NEAR, FINAL], recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.incidents == ["near_dup_call"]
    assert r.answer == "Гетьман Війська Запорозького"

    orch2, llm2 = _orch([DUP, NEAR, FINAL], recovery=False)
    r2 = orch2.run("t", seed=1, budget=Budget(max_steps=5))
    assert r2.incidents == []
    assert len(r2.scratch) == 2


def test_nudge_cap_then_partial_answer():
    orch, llm = _orch([DUP] + [DUP] * 5, recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=8))
    assert r.partial and r.degraded
    assert r.answer is not None and "lookup_fact" in r.answer
    assert r.incidents == ["dup_call", "dup_call", "dup_call"]
    assert r.accepted is False


def test_parse_failure_retightens_schema_then_recovers():
    trace = InMemoryTrace()
    orch, llm = _orch(["не json", FINAL], recovery=True, trace=trace)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.incidents == ["not_json"]
    assert r.answer == "Гетьман Війська Запорозького"
    assert "oneOf" in str(llm.calls[1]["schema"])


def test_parse_failure_without_recovery_breaks_but_is_visible():
    orch, llm = _orch(["не json"], recovery=False)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.degraded and r.answer is None
    assert r.incidents == ["not_json", "no_final_answer"]


def test_empty_output_classified_not_as_parse_error():
    orch, llm = _orch(["", FINAL], recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.incidents == ["empty_output"]


def test_tool_unknown_adds_hint_without_extra_step():
    unknown = '{"tool": "lookup_fact", "entity": "Невідома Особа"}'
    orch, llm = _orch([unknown, FINAL], recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.notes == ["tool_unknown"], "«інструмент не знає» — нотатка, не збій"
    assert r.incidents == []
    assert "Підказка:" in llm.calls[1]["prompt"]
    assert r.steps == 2


def test_tool_error_hint_carries_detail():
    bad = '{"tool": "calc", "expr": "import os"}'
    orch, llm = _orch([bad, FINAL], recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.incidents == ["tool_error"]
    assert "disallowed characters" in llm.calls[1]["prompt"]


def test_reasking_after_unknown_is_not_a_near_duplicate():
    unknown_a = '{"tool": "lookup_fact", "entity": "Пилип Орлик"}'
    unknown_b = '{"tool": "lookup_fact", "entity": "гетьман Пилип Орлик"}'
    orch, llm = _orch([unknown_a, unknown_b, FINAL], recovery=True)
    r = orch.run("хто такий Пилип Орлик", seed=1, budget=Budget(max_steps=6))
    assert r.notes == ["tool_unknown", "tool_unknown"], \
        "переформулювання після 'не знаю' — це пошук, не топтання"
    assert r.incidents == []
    assert r.answer == "Гетьман Війська Запорозького"
    assert not r.partial


def test_soft_hints_do_not_starve_a_later_real_incident():
    unknown = '{"tool": "lookup_fact", "entity": "Пилип Орлик"}'
    unknown2 = '{"tool": "lookup_fact", "entity": "гетьман Орлик"}'
    orch, llm = _orch([unknown, unknown2, DUP, DUP, FINAL], recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=8))
    assert r.notes == ["tool_unknown", "tool_unknown"]
    assert r.incidents == ["dup_call"]
    assert not r.partial, "дві мʼякі підказки не мають доводити до здачі"
    assert r.answer == "Гетьман Війська Запорозького"


def test_steps_equal_llm_calls_invariant():
    orch, llm = _orch([DUP, DUP, FINAL], recovery=True, verifier=False)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.steps == len(llm.calls) == 3


def test_recovery_cap_limits_total_attempts():
    orch, llm = _orch([DUP] * 8, recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=4))
    assert r.incidents == ["dup_call", "dup_call", "dup_call"]
    assert r.partial and r.answer is not None
    assert len(llm.calls) == 4


def test_partial_skips_verifier():
    llm = FakeLlm([DUP] * 6, strict=True)
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                        verifier=True, recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=8))
    assert r.partial and r.accepted is False and r.verdict_reason is None
    assert all('"accepted"' not in c["prompt"] for c in llm.calls)


def test_verifier_tokens_land_in_aux_not_main():
    llm = FakeLlm([FINAL, JUDGE], strict=True)
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                        verifier=True, recovery=True)
    r = orch.run("хто такий Іван Мазепа", seed=1, budget=Budget(max_steps=5))
    assert r.accepted and r.steps == 1
    assert r.aux_tokens > 0
    assert r.tokens > 0


def test_recovery_off_is_byte_identical_to_baseline():
    a, _ = _orch([DUP, DUP], recovery=False)
    b, _ = _orch([DUP, DUP], recovery=False)
    ra = a.run("t", seed=7, budget=Budget(max_steps=5))
    rb = b.run("t", seed=7, budget=Budget(max_steps=5))
    assert ra.model_dump() == rb.model_dump()


def test_recovery_on_is_deterministic_across_runs():
    a, _ = _orch([DUP, DUP, FINAL], recovery=True)
    b, _ = _orch([DUP, DUP, FINAL], recovery=True)
    ra = a.run("t", seed=3, budget=Budget(max_steps=6))
    rb = b.run("t", seed=3, budget=Budget(max_steps=6))
    assert ra.model_dump() == rb.model_dump()


def test_trace_identical_modulo_latency():
    def trace_of():
        tr = InMemoryTrace()
        orch, _ = _orch([DUP, DUP, FINAL], recovery=True, trace=tr)
        orch.run("t", seed=5, budget=Budget(max_steps=6))
        return [r.model_dump(exclude={"latency_ms"}) for r in tr.records]

    assert trace_of() == trace_of()


def test_partial_guard_holds_on_empty_scratch():
    orch, llm = _orch(["не json", "теж не json", "і це не json"], recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=6))
    assert r.incidents == ["not_json", "not_json", "not_json", "no_final_answer"]
    assert r.partial is False and r.answer is None and r.degraded


def test_parse_ladder_retightens_on_weak_model_before_escalating():
    weak = FakeLlm(["не json", "теж не json"], model="lapa", strict=True)
    strong = FakeLlm([FINAL], model="mamay", strict=True)
    orch = Orchestrator(profile_router(weak, strong), PresetEffort(), FakeToolbox(),
                        verifier=False, recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.incidents == ["not_json", "not_json"]
    assert len(weak.calls) == 2, "retighten мусить лишитись на дешевій моделі"
    assert len(strong.calls) == 1, "escalate мусить піти на сильнішу"
    assert "oneOf" in str(weak.calls[1]["schema"]), "retighten вже дає strict"
    assert "oneOf" in str(strong.calls[0]["schema"]), "escalate тримає strict"
    assert r.answer == "Гетьман Війська Запорозького"


def test_escalate_unavailable_falls_through_to_next_rung():
    llm = FakeLlm(["не json", "теж не json"], model="only-select", strict=True)
    orch = Orchestrator(ProfileRouter({"select": llm}), PresetEffort(), FakeToolbox(),
                        verifier=False, recovery=True)
    r = orch.run("t", seed=1, budget=Budget(max_steps=5))
    assert r.incidents == ["not_json", "not_json", "no_final_answer"]
    assert len(llm.calls) == 2
    assert r.degraded and r.answer is None


def test_escalated_kind_map_is_one_way():
    assert escalated("select") == "decide"
    assert escalated("judge") is None
    assert escalated("decide") == "synthesize"
