import inspect
from pathlib import Path

from evalkit.checks import check
from evalkit.harness import EvalItem, EvalResult, run_eval
from evalkit.prompts import resolve
from evalkit.report import by_prompt, format_prompt_report, sensitivity
from ploshcha_sim.adapters import FakeLlm, FakeToolbox, InMemoryTrace, PresetEffort, single_model_router
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm
from ploshcha_sim.agents.orchestrator import Orchestrator
from ploshcha_sim.domain.task import Budget, TaskResult

FINAL = '{"tool": "final_answer", "text": "ok"}'
DUP = '{"tool": "lookup_fact", "entity": "Іван Мазепа"}'


def _res(**kw):
    return TaskResult(**kw)


def test_no_tools_parameter_anywhere_in_the_adapter():
    source = inspect.getsource(OpenAICompatLlm)
    assert "tools" not in source, "Gemma-3 не має tool-токенів; шлюз tools тихо викидає"
    assert "tool_choice" not in source


def test_serve_local_uses_jinja_template():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "serve_local.sh").read_text()
    assert "--jinja" in script, "без --jinja llama.cpp рендерить system іншою гілкою"


def test_scripts_hold_no_prompt_literals():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("eval_local.py", "probe_orchestrator.py"):
        text = (scripts / name).read_text(encoding="utf-8")
        assert "Ти агент з інструментами" not in text, f"{name}: літерал промпту замість реєстру"
        assert "resolve(" in text


def test_tail_lands_as_the_last_prompt_line():
    variant = resolve("agent/v2-tail")
    llm = FakeLlm([FINAL], strict=True)
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(), verifier=False,
                        system=variant.render_system(), tail=variant.tail,
                        prompt_id=variant.id, prompt_sha=variant.sha256)
    orch.run("t", seed=1, budget=Budget(max_steps=3))
    assert llm.calls[0]["prompt"].splitlines()[-1] == variant.tail
    assert llm.calls[0]["system"] == variant.head


def test_no_tail_keeps_the_instruction_last():
    llm = FakeLlm([FINAL], strict=True)
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(), verifier=False)
    orch.run("t", seed=1, budget=Budget(max_steps=3))
    assert llm.calls[0]["prompt"].splitlines()[-1].startswith("Наступний крок")


def test_prompt_id_and_tier_land_in_the_trace():
    trace = InMemoryTrace()
    variant = resolve("agent/v2")
    llm = FakeLlm([FINAL], strict=True)
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(), verifier=False,
                        trace=trace, prompt_id=variant.id, prompt_sha=variant.sha256)
    orch.run("t", seed=1, budget=Budget(max_steps=3))
    ablation = trace.records[0].ablation
    assert ablation["prompt"] == "agent/v2"
    assert ablation["prompt_sha"] == variant.sha256
    assert ablation["tier"] in ("strict", "wire", "none")


def test_eval_result_carries_prompt_id():
    items = [EvalItem(id="a", category="x", task="t",
                      checks=[{"kind": "answer_contains", "value": "ok"}])]
    runner = lambda task, seed: _res(answer="ok")  # noqa: E731
    rows = run_eval(items, {"cond": runner}, [0], prompt_ids={"cond": "agent/v2"})
    assert rows[0].prompt_id == "agent/v2"


def test_no_incident_predicate():
    clean = _res(answer="x", incidents=[])
    dirty = _res(answer="x", incidents=["dup_call"])
    assert check({"kind": "no_incident"}, clean)
    assert not check({"kind": "no_incident"}, dirty)
    assert check({"kind": "no_incident", "code": "not_json"}, dirty)
    assert not check({"kind": "no_incident", "code": "dup_call"}, dirty)


def test_steps_between_predicate_catches_premature_finish():
    early = _res(answer="x", steps=1)
    proper = _res(answer="x", steps=3)
    assert not check({"kind": "steps_between", "lo": 2, "hi": 4}, early)
    assert check({"kind": "steps_between", "lo": 2, "hi": 4}, proper)


def test_tool_calls_at_most_and_partial_predicates():
    two = _res(answer="x", scratch=[{"call": {"tool": "calc"}}, {"call": {"tool": "calc"}}])
    assert check({"kind": "tool_calls_at_most", "n": 2}, two)
    assert not check({"kind": "tool_calls_at_most", "n": 1}, two)
    assert check({"kind": "not_partial"}, _res(answer="x"))
    assert not check({"kind": "not_partial"}, _res(answer="x", partial=True))


def test_by_prompt_and_sensitivity_math():
    rows = [
        EvalResult(item_id="a", category="x", condition="c", prompt_id="p1", seed=1,
                   success=True, checks={}, tokens=100),
        EvalResult(item_id="b", category="x", condition="c", prompt_id="p1", seed=1,
                   success=True, checks={}, tokens=100),
        EvalResult(item_id="a", category="x", condition="c", prompt_id="p2", seed=1,
                   success=True, checks={}, tokens=200, incidents=["dup_call"]),
        EvalResult(item_id="b", category="x", condition="c", prompt_id="p2", seed=1,
                   success=False, checks={}, tokens=200),
    ]
    groups = by_prompt(rows)
    assert groups["p1"]["success_rate"] == 1.0
    assert groups["p2"]["success_rate"] == 0.5
    assert groups["p2"]["incidents"] == {"dup_call": 1}
    s = sensitivity(rows)
    assert s["spread"] == 0.5 and s["best"] == "p1" and s["worst"] == "p2"
    assert s["flip_rate"] == 0.5, "айтем b перевернувся між промптами"
    assert "spread=" in format_prompt_report(rows)


CANDIDATE_ON_PURPOSE = {"agent/v2-ua", "agent/v2-reg", "agent/v2-iter", "agent/v2-agg", "answer/full"}


def test_frozen_prompt_is_the_one_eval_uses():
    """Раніше цей тест грепав текст скрипта; тепер промпт — поле специфікації, тож перевіряємо дані."""
    from evalkit.conditions import CONDITIONS
    from ploshcha_sim.domain.spec import AppSpec

    assert AppSpec().prompt_id == "agent/v2"
    assert resolve("agent/v2").status == "frozen"

    for name, spec in CONDITIONS.items():
        for pid in (spec.prompt_id, spec.answer_prompt_id):
            variant = resolve(pid)
            assert variant.status != "rejected", f"{name} використовує відкинутий промпт {pid}"
            if variant.status != "frozen":
                assert pid in CANDIDATE_ON_PURPOSE, (
                    f"{name} тихо взяв незаморожений промпт {pid} ({variant.status})")
