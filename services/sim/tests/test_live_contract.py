"""Кожна подія ЖИВОГО потоку мусить пройти сам контракт, а не лише збігатися енумами.

Чому окремий файл. `test_contract_parity.py` звіряє енуми, а `test_appspec_parity.py` валідує
ПАКЕТНІ фікстури — але в пакетному режимі `emit_ticks/emit_motion/emit_voices` вимкнені для
байт-сумісності, тобто єдиний шлях валідації виключав саме ті три типи, які додав Я6. Наслідок був
не теоретичний: `tick.begin` віддавав `{"tick": N}` замість `timeOfDay`, а `to`/`place` — рядок
замість `PlaceRef`. `.strict()` на фронті відкидає такі конверти молча: механізм працює, шлях
спостереження зламаний.

Тут валідатор — сама схема (`oneOf` по всіх типах), а не мій перелік полів: якби я перелічував поля
руками, я б повторив ту саму помилку, що й у продюсері.
"""

import json
from pathlib import Path

import pytest

from ploshcha_sim.adapters import FakeLlm, FakeToolbox, PresetEffort
from ploshcha_sim.adapters.projector import StreamProjector, project_run
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.domain.state import PHASES, TimeOfDay
from ploshcha_sim.live import BusTrace, EventBus
from ploshcha_sim.ports.trace import StepRecord

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "contracts" / "ploshcha-events.schema.json"
TS = "2026-01-01T00:00:00Z"

LIVE_ONLY = {"tick.begin", "agent.moved", "utterance.spoken"}


@pytest.fixture(scope="module")
def validator():
    jsonschema = pytest.importorskip("jsonschema")
    if not SCHEMA_PATH.exists():
        pytest.skip(f"немає {SCHEMA_PATH}")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def _why(validator, event: dict) -> str:
    """Схема — `oneOf`, тож верхня помилка завжди «not valid under any»: беремо найглибшу гілку."""
    from jsonschema.exceptions import best_match

    match = best_match(validator.iter_errors(event))
    while match is not None and match.context:
        match = best_match(match.context)
    return f"{match.json_path}: {match.message}" if match else ""


def check(validator, events: list[dict]) -> set[str]:
    for event in events:
        assert validator.is_valid(event), (
            f"{event['type']} не проходить контракт — {_why(validator, event)}"
            f"\nконверт: {json.dumps(event, ensure_ascii=False)[:300]}"
        )
    return {e["type"] for e in events}


def tc(tool, **args):
    return json.dumps({"tool": tool, **args}, ensure_ascii=False)


SCENE = {"id": "ploshcha", "name": "Площа"}


def live_events(*, verifier: bool = True, scene: dict | None = SCENE) -> list[dict]:
    bus = EventBus()
    proj = StreamProjector("r", TS, scene=scene, max_ticks=6)
    trace = BusTrace(bus, proj)
    llm = FakeLlm([tc("lookup_fact", entity="мешти"), tc("final_answer", text="Мешти — черевики."),
                   json.dumps({"kind": "supported", "accepted": True,
                               "reason": "Підтверджено довідником."}, ensure_ascii=False)],
                  model="fake")
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                        verifier=verifier, trace=trace, run_id="r")
    for event in proj.start():
        bus.publish(event)
    result = orch.run("що таке мешти")
    for event in proj.close(result, done=True):
        bus.publish(event)
    return bus.since(0)[0]


# ── енум фази ─────────────────────────────────────────────────────────────────

def test_the_time_of_day_enum_matches_the_contract(validator):
    """Розходження, яке пропустив паритет-тест: у ядрі був `afternoon`, у контракті — `dusk`."""
    declared = set(validator.schema["$defs"]["TimeOfDay"]["enum"])
    assert declared == set(PHASES)
    assert declared == set(TimeOfDay.__args__)


@pytest.mark.parametrize("tick", range(len(PHASES) * 2 + 1))
def test_every_tick_produces_a_contract_valid_phase(validator, tick):
    """Раніше кожна шоста фаза дала б невалідну подію — а тест бачив би лише перші п'ять."""
    check(validator, StreamProjector("r", TS)._tick_events(tick))


# ── живий потік ───────────────────────────────────────────────────────────────

def test_every_event_of_a_live_run_passes_the_contract(validator):
    seen = check(validator, live_events())
    assert LIVE_ONLY <= seen, f"живі типи не покриті прогоном: {LIVE_ONLY - seen}"


def test_the_live_run_covers_the_cognition_events_too(validator):
    seen = check(validator, live_events())
    assert {"run.started", "route.decided", "tool.called", "tool.result",
            "verify.verdict", "task.outcome", "run.done"} <= seen


def test_a_run_without_a_verifier_is_also_valid(validator):
    check(validator, live_events(verifier=False))


def test_a_run_without_a_scene_is_also_valid(validator):
    events = live_events(scene=None)
    assert not [e for e in events if e["type"] == "run.started"]
    check(validator, events)


# ── окремі типи, які контракт описує строго ───────────────────────────────────

def test_the_place_of_an_utterance_is_a_place_ref_not_a_string(validator):
    events = live_events()
    said = [e for e in events if e["type"] == "utterance.spoken"]
    assert said, "прогін мусить дати хоч один голос, інакше тест нічого не перевіряє"
    for event in said:
        assert isinstance(event["payload"]["place"], dict)
        assert event["payload"]["place"].get("poi")
    check(validator, said)


def test_the_destination_of_a_move_is_a_place_ref_not_a_string(validator):
    moved = [e for e in live_events() if e["type"] == "agent.moved"]
    assert moved
    for event in moved:
        assert isinstance(event["payload"]["to"], dict)
        assert event["payload"]["to"].get("poi")
    check(validator, moved)


def test_a_string_place_would_have_been_rejected(validator):
    """Доказ, що валідатор справді ловить саму поламку, а не проходить усе поспіль."""
    broken = StreamProjector("r", TS).feed(
        StepRecord(run_id="r", tick=1, agent="orchestrator", stage="synthesize", model="m",
                   lane="mamay", prompt="", raw_output="Сказав."))
    said = next(e for e in broken if e["type"] == "utterance.spoken")
    said["payload"]["place"] = "square"
    assert list(validator.iter_errors(said)), "рядок замість PlaceRef мусить бути відкинутий"


def test_a_tick_payload_without_the_time_of_day_would_have_been_rejected(validator):
    tick = StreamProjector("r", TS)._tick_events(1)[0]
    tick["payload"] = {"tick": 1}
    assert list(validator.iter_errors(tick)), "стара форма `{tick: N}` мусить бути відкинута"


# ── події сторожа: їх емітить LiveRunner, а не проєктор ───────────────────────

def _runner(tmp_path, *, tokens: int = 1):
    from ploshcha_sim.adapters.queue_sqlite import SqliteQueue
    from ploshcha_sim.domain.governor import Governor
    from ploshcha_sim.live import LiveRunner

    bus = EventBus()
    queue = SqliteQueue(str(tmp_path / "q.db"))

    def make(trace, run_id, place=None):
        llm = FakeLlm([tc("final_answer", text="Готово.")], model="fake")
        return Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                            verifier=False, trace=trace, run_id=run_id)

    return bus, LiveRunner(bus, queue, make, governor=Governor(max_tokens=tokens))


def test_the_governor_stop_event_passes_the_contract(validator, tmp_path):
    """Стеля — не аварія, а очікуваний стан; якщо подія невалідна, спостерігач не дізнається чому."""
    import time

    bus, runner = _runner(tmp_path)
    runner.governor.record(tokens=10_000)
    runner.resume()
    runner.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and runner.stopped_reason is None:
        time.sleep(0.05)
    runner.stop()

    events = bus.since(0)[0]
    degraded = [e for e in events if e["type"] == "run.degraded"]
    assert degraded, "стеля мусить дати run.degraded"
    assert degraded[0]["payload"]["stage"], "контракт вимагає `stage`, а не лише `reason`"
    check(validator, events)


def test_a_crashed_loop_reports_a_contract_valid_error(validator, tmp_path):
    import time

    bus, runner = _runner(tmp_path, tokens=10_000)
    runner.queue.put("k", {"task": "тема"})

    def boom(trace, run_id, place=None):
        raise RuntimeError("двигун не піднявся")

    runner.make_orchestrator = boom
    runner.resume()
    runner.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and not [e for e in bus.since(0)[0] if e["type"] == "run.error"]:
        time.sleep(0.05)
    runner.stop()

    events = bus.since(0)[0]
    assert [e for e in events if e["type"] == "run.error"], "падіння мусить бути видно в потоці"
    check(validator, events)


# ── пакетний режим не зіпсований ──────────────────────────────────────────────

def test_batch_projection_still_passes_the_contract(validator):
    records = [StepRecord(run_id="r", tick=1, agent="orchestrator", stage="select", model="m",
                          lane="mamay", prompt="", raw_output=tc("lookup_fact", entity="X"),
                          parsed={"tool": "lookup_fact", "entity": "X"})]

    class Result:
        outcome = "answer"
        evidence = True
        scratch = [{"call": {"tool": "lookup_fact"}, "result": {"відомо": True}, "found": True}]
        notes: list[str] = []
        incidents: list[str] = []

    seen = check(validator, project_run(records, Result(), run_id="r", ts=TS, scene=SCENE))
    assert not (LIVE_ONLY & seen), "пакетний режим мусить лишатись байт-сумісним"
