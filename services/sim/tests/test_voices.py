import json

import pytest

from ploshcha_sim.adapters import FakeLlm, FakeToolbox, PresetEffort
from ploshcha_sim.adapters.projector import (
    VILLAGERS,
    VOICE_OF_LANE,
    VOICE_VERIFIER,
    StreamProjector,
    is_prose,
    project_run,
    villager_of_span,
)
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.live import BusTrace, EventBus
from ploshcha_sim.ports.trace import StepRecord

TS = "2026-01-01T00:00:00Z"


def rec(**kw):
    base = dict(run_id="r", tick=1, agent="orchestrator", stage="synthesize", model="m",
                lane="mamay", prompt="", raw_output="")
    base.update(kw)
    return StepRecord(**base)


def _proj(**kw):
    return StreamProjector("r", TS, **kw)


# ── що є прозою ───────────────────────────────────────────────────────────────

def test_json_tool_call_is_not_speech():
    r = rec(stage="select", lane="lapa", raw_output='{"tool": "lookup_fact", "entity": "X"}',
            parsed={"tool": "lookup_fact", "entity": "X"})
    assert is_prose(r) is False


def test_bare_json_without_parsed_is_still_not_speech():
    assert is_prose(rec(raw_output='{"accepted": true}')) is False
    assert is_prose(rec(raw_output='[1, 2, 3]')) is False


def test_prose_answer_is_speech():
    assert is_prose(rec(raw_output="Бараболя — це картопля.")) is True


def test_empty_output_is_not_speech():
    assert is_prose(rec(raw_output="   ")) is False


def test_subagent_output_is_always_speech():
    assert is_prose(rec(agent="subagent", raw_output="Кажу як є.")) is True


# ── голоси ────────────────────────────────────────────────────────────────────

def test_answer_step_speaks_with_the_lane_voice():
    events = _proj().feed(rec(raw_output="Громада вирішила: копати."))
    said = [e for e in events if e["type"] == "utterance.spoken"]
    assert len(said) == 1
    assert said[0]["payload"]["agentId"] == VOICE_OF_LANE["mamay"]
    assert said[0]["payload"]["text"] == "Громада вирішила: копати."


def test_executor_lane_has_its_own_voice():
    events = _proj().feed(rec(lane="lapa", stage="synthesize", raw_output="Я дістав із довідника."))
    said = next(e for e in events if e["type"] == "utterance.spoken")
    assert said["payload"]["agentId"] == VOICE_OF_LANE["lapa"]
    assert said["payload"]["agentId"] != VOICE_OF_LANE["mamay"]


def test_tool_step_produces_no_utterance():
    events = _proj().feed(rec(stage="select", lane="lapa",
                              raw_output='{"tool": "lookup_fact", "entity": "X"}',
                              parsed={"tool": "lookup_fact", "entity": "X"}))
    assert not [e for e in events if e["type"] == "utterance.spoken"]
    assert [e["type"] for e in events if e["type"] == "tool.called"]


def test_subagent_speaks_as_a_villager_at_the_square():
    events = _proj().feed(rec(agent="subagent", span="graph/2", raw_output="Я б не спішив."))
    said = next(e for e in events if e["type"] == "utterance.spoken")
    assert said["payload"]["agentId"] in VILLAGERS
    assert said["payload"]["place"] == {"poi": "square"}


def test_the_same_span_is_always_the_same_villager():
    a = villager_of_span("graph/3")
    b = villager_of_span("graph/3")
    assert a == b
    assert villager_of_span("graph/4") != a or len(VILLAGERS) == 1


def test_verifier_reason_speaks_from_the_church():
    events = _proj().feed(rec(agent="verifier", stage="judge",
                              parsed={"kind": "supported", "accepted": True,
                                      "reason": "Доказ у результаті інструмента."}))
    said = next(e for e in events if e["type"] == "utterance.spoken")
    assert said["payload"]["agentId"] == VOICE_VERIFIER
    assert said["payload"]["place"] == {"poi": "church"}
    assert [e for e in events if e["type"] == "verify.verdict"]


def test_verifier_without_reason_stays_silent():
    events = _proj().feed(rec(agent="verifier", stage="judge",
                              parsed={"kind": "supported", "accepted": True}))
    assert not [e for e in events if e["type"] == "utterance.spoken"]


def test_whitespace_is_collapsed_and_text_is_capped():
    events = _proj().feed(rec(raw_output="а" * 900))
    said = next(e for e in events if e["type"] == "utterance.spoken")
    assert len(said["payload"]["text"]) == 600


def test_multiline_output_becomes_one_line():
    events = _proj().feed(rec(raw_output="Перший рядок.\n\n  Другий рядок."))
    said = next(e for e in events if e["type"] == "utterance.spoken")
    assert said["payload"]["text"] == "Перший рядок. Другий рядок."


# ── сумісність пакетного режиму ───────────────────────────────────────────────

def test_batch_projection_stays_silent():
    events = project_run([rec(raw_output="Проза тут є.")], None, run_id="r", ts=TS)
    assert not [e for e in events if e["type"] == "utterance.spoken"]


# ── живий прогін ──────────────────────────────────────────────────────────────

def tc(tool, **args):
    return json.dumps({"tool": tool, **args}, ensure_ascii=False)


def test_live_run_speaks_once_per_prose_output():
    bus = EventBus()
    proj = StreamProjector("r", TS)
    trace = BusTrace(bus, proj)
    llm = FakeLlm([tc("lookup_fact", entity="X"), tc("final_answer", text="Це картопля.")],
                  model="fake")
    Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(), verifier=False,
                 trace=trace, run_id="r").run("q")
    said = [e for e in bus.since(0)[0] if e["type"] == "utterance.spoken"]
    assert said == [] or all(not s["payload"]["text"].startswith("{") for s in said)


@pytest.mark.parametrize("lane", ["mamay", "lapa", "unknown"])
def test_every_lane_has_a_voice(lane):
    events = _proj().feed(rec(lane=lane, raw_output="Щось сказав."))
    said = next(e for e in events if e["type"] == "utterance.spoken")
    assert said["payload"]["agentId"]
