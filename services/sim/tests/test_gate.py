from evalkit.harness import gated_runner, orchestrator_runner
from ploshcha_sim.adapters import FakeLlm, FakeToolbox, PresetEffort, single_model_router
from ploshcha_sim.adapters.tools_fake import DEFAULT_TOOLS, Tool
from ploshcha_sim.agents.orchestrator import Orchestrator
from ploshcha_sim.domain.gate import data_tool_names, gate_reason, needs_loop
from ploshcha_sim.domain.task import Budget

FINAL_ONLY = [t for t in DEFAULT_TOOLS if t.name == "final_answer"]
ANSWER = '{"tool": "final_answer", "text": "готово"}'


def test_gate_reads_the_toolbox_not_the_task_text():
    full, bare = FakeToolbox(), FakeToolbox(tools=FINAL_ONLY)
    assert needs_loop(full.specs()) is True
    assert needs_loop(bare.specs()) is False
    assert data_tool_names(bare.specs()) == []
    assert set(data_tool_names(full.specs())) == {"check_date", "lookup_fact", "calc"}


def test_gate_decision_is_independent_of_wording():
    bare = FakeToolbox(tools=FINAL_ONLY)
    for task in ("постав у кличний відмінок", "обчисли 2+2 інструментом calc",
                 "перевір дату Битви під Крутами"):
        assert needs_loop(bare.specs()) is False, task


def test_gate_reason_is_recorded():
    assert gate_reason(FakeToolbox(tools=FINAL_ONLY).specs()) == "no_data_tools"
    assert gate_reason(FakeToolbox().specs()) == "tools:3"


def test_empty_toolbox_is_also_gated_out():
    empty = FakeToolbox(tools=[])
    assert needs_loop(empty.specs()) is False


def test_gated_runner_answers_directly_when_no_data_tools():
    llm = FakeLlm(["Петре Кузьменку"], strict=True)
    bare = FakeToolbox(tools=FINAL_ONLY)
    run = gated_runner(llm, bare, system="ти редактор")
    r = run("звернись до Петра Кузьменка", seed=1)
    assert r.answer == "Петре Кузьменку"
    assert r.steps == 1 and r.scratch == []
    assert r.notes == ["gate:no_data_tools"]
    assert llm.calls[0]["structured"] is False, "без схеми — латинські ключі псують укр. вивід"
    assert llm.calls[0]["system"] == "ти редактор"


def test_gated_runner_uses_the_loop_when_tools_exist():
    llm = FakeLlm([ANSWER], strict=True)
    box = FakeToolbox()
    loop = orchestrator_runner(
        lambda: Orchestrator(single_model_router(llm), PresetEffort(), box, verifier=False),
        budget=Budget(max_steps=4))
    run = gated_runner(llm, box, loop_runner=loop)
    r = run("перевір дату", seed=1)
    assert r.answer == "готово"
    assert llm.calls[0]["structured"] is True, "тул-задача йде через схему, як і має"
    assert "gate:" not in " ".join(r.notes)


def test_gate_falls_back_to_direct_call_without_a_loop_runner():
    llm = FakeLlm(["пряма відповідь"], strict=True)
    run = gated_runner(llm, FakeToolbox(), loop_runner=None)
    r = run("перевір дату", seed=1)
    assert r.answer == "пряма відповідь"
    assert r.notes == ["gate:tools:3"]


def test_adding_a_tool_flips_the_gate():
    lookup = next(t for t in DEFAULT_TOOLS if t.name == "lookup_fact")
    extra = Tool("нюхач", "Нюхає повітря.", lookup.params, lambda a: {"ok": True})

    assert needs_loop(FakeToolbox(tools=FINAL_ONLY).specs()) is False
    with_one = FakeToolbox(tools=FINAL_ONLY + [extra])
    assert needs_loop(with_one.specs()) is True
    assert data_tool_names(with_one.specs()) == ["нюхач"]
