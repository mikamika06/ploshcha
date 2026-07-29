from evalkit.harness import orchestrator_runner
from ploshcha_sim.adapters import FakeLlm, PresetEffort
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.compose import build_toolbox
from ploshcha_sim.domain.coverage import (
    PENDING_LABEL,
    collection_items,
    mark_fetched,
    render_pending,
)
from ploshcha_sim.domain.spec import AppSpec
from ploshcha_sim.domain.task import Budget

LIST_CALL = '{"tool": "список_записів", "село": "Вербівка"}'
RECORD = '{"tool": "запис", "ідентифікатор": "зп-1893-01"}'
FINAL = '{"tool": "final_answer", "text": "готово"}'


def _orch(script, **kw):
    return Orchestrator(single_model_router(FakeLlm(script)), PresetEffort(),
                        build_toolbox(AppSpec().with_(toolset="registry")),
                        verifier=False, **kw)


def test_collection_items_takes_the_longest_string_list():
    value = {"відомо": True, "село": "Вербівка", "записи": ["зп-1", "зп-2", "зп-3"]}
    assert collection_items(value) == ["зп-1", "зп-2", "зп-3"]


def test_an_aggregate_payload_yields_no_pending_list():
    """Агрегат віддає список СЛОВНИКІВ — покриття там не потрібне й не має вигадуватись."""
    value = {"відомо": True, "абзаци": [{"ідентифікатор": "аб-1", "текст": "x"}]}
    assert collection_items(value) == []
    assert collection_items("не словник") == []
    assert collection_items({"порожньо": []}) == []


def test_fetched_items_leave_the_pending_list():
    pending = ["зп-1", "зп-2", "зп-3"]
    assert mark_fetched(pending, {"ідентифікатор": "зп-2"}) == ["зп-1", "зп-3"]
    assert mark_fetched(pending, {"село": "Вербівка"}) == pending
    assert mark_fetched(pending, {"рік": 1893}) == pending


def test_pending_is_rendered_only_while_something_is_left():
    assert render_pending([]) is None
    text = render_pending(["a", "b"])
    assert text is not None and text.startswith(PENDING_LABEL) and "(2)" in text
    long = render_pending([str(i) for i in range(20)], limit=3)
    assert "ще 17" in long


def test_coverage_off_by_default_changes_nothing():
    orch = _orch([LIST_CALL, FINAL])
    result = orch.run("задача", seed=1, budget=Budget(max_steps=4))
    assert PENDING_LABEL not in (result.answer or "")
    assert orch.coverage is False


def test_the_remaining_list_reaches_the_prompt():
    llm = FakeLlm([LIST_CALL, RECORD, FINAL])
    orch = Orchestrator(single_model_router(llm), PresetEffort(),
                        build_toolbox(AppSpec().with_(toolset="registry")),
                        verifier=False, coverage=True)
    orch.run("задача", seed=1, budget=Budget(max_steps=5))
    prompts = [c["prompt"] for c in llm.calls]
    assert PENDING_LABEL not in prompts[0], "до першого виклику залишку ще нема"
    assert PENDING_LABEL in prompts[1], "після переліку залишок мусить бути в промпті"
    assert "зп-1893-01" in prompts[1]
    assert "зп-1893-01" not in prompts[2].split(PENDING_LABEL)[1], "здобуте зникає із залишку"


def test_pending_shrinks_by_one_per_fetch():
    llm = FakeLlm([LIST_CALL, RECORD, '{"tool": "запис", "ідентифікатор": "зп-1893-02"}', FINAL])
    orch = Orchestrator(single_model_router(llm), PresetEffort(),
                        build_toolbox(AppSpec().with_(toolset="registry")),
                        verifier=False, coverage=True)
    orch.run("задача", seed=1, budget=Budget(max_steps=6))
    counts = [p.split(f"{PENDING_LABEL} (")[1].split(")")[0]
              for p in (c["prompt"] for c in llm.calls) if PENDING_LABEL in p]
    assert counts == ["8", "7", "6"], counts


def test_coverage_is_a_condition_axis():
    from evalkit.conditions import CONDITIONS
    assert CONDITIONS["chain-cover-schema@16"].coverage is True
    assert CONDITIONS["chain-schema@16"].coverage is False
    assert (CONDITIONS["chain-cover-schema@16"].sha256
            != CONDITIONS["chain-schema@16"].sha256)


def test_runner_keeps_pending_out_of_the_next_run():
    llm = FakeLlm([LIST_CALL, RECORD, FINAL] * 2)
    runner = orchestrator_runner(
        lambda: Orchestrator(single_model_router(llm), PresetEffort(),
                             build_toolbox(AppSpec().with_(toolset="registry")),
                             verifier=False, coverage=True),
        budget=Budget(max_steps=5))
    first, second = runner("t1", 1), runner("t2", 1)
    assert first.steps == second.steps, "залишок не має протікати між прогонами"


def test_iteration_is_not_near_duplication():
    """Дефект 19: обхід колекції класифікувався як майже-дубль, і драбина його гасила."""
    from ploshcha_sim.domain.coverage import targets_pending
    from ploshcha_sim.domain.recovery import is_near_duplicate

    history = [("запис", {"ідентифікатор": "зп-1893-01"})]
    args = {"ідентифікатор": "зп-1893-02"}
    assert is_near_duplicate("запис", args, history), "детектор справді вважає це майже-дублем"
    assert targets_pending(args, ["зп-1893-02", "зп-1895-01"]), "але це наступний елемент переліку"
    assert not targets_pending({"ідентифікатор": "зп-1893-01"}, ["зп-1893-02"]), "здобуте — не ітерація"


def test_the_ladder_lets_iteration_through():
    llm = FakeLlm([LIST_CALL, RECORD, '{"tool": "запис", "ідентифікатор": "зп-1893-02"}',
                   '{"tool": "запис", "ідентифікатор": "зп-1895-01"}', FINAL])
    orch = Orchestrator(single_model_router(llm), PresetEffort(),
                        build_toolbox(AppSpec().with_(toolset="registry")),
                        verifier=False, coverage=True, recovery=True)
    result = orch.run("задача", seed=1, budget=Budget(max_steps=8))
    calls = [x["call"]["tool"] for x in result.scratch]
    assert calls.count("запис") == 3, f"усі три обходи мусять пройти: {calls}"
    assert "near_dup_call" not in result.incidents, result.incidents
