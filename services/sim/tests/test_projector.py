import json
from pathlib import Path

import pytest

from ploshcha_sim.adapters import FakeLlm, FakeToolbox, InMemoryTrace, PresetEffort
from ploshcha_sim.adapters.projector import PROTOCOL, project_run
from ploshcha_sim.adapters.router_profile import profile_router
from ploshcha_sim.adapters.tools_lexis import LEXIS_TOOLS
from ploshcha_sim.agents import Orchestrator

FIXTURE = Path(__file__).resolve().parents[3] / "packages" / "fixtures" / "runs" / "projected-run.jsonl"
TS = "2026-07-31T09:00:00Z"


def _call(tool, **args):
    return json.dumps({"tool": tool, **args}, ensure_ascii=False)


def _run(script, **kw):
    llm = FakeLlm(script, model="fixture")
    trace = InMemoryTrace()
    orch = Orchestrator(profile_router(llm, llm), PresetEffort(), FakeToolbox(tools=LEXIS_TOOLS),
                        trace=trace, run_id="t", answer_channel="text",
                        verify_mode="grounded", **kw)
    result = orch.run("питання", seed=1)
    return list(trace.records), result


def _events(script, **kw):
    records, result = _run(script, **kw)
    return project_run(records, result, run_id="t", ts=TS)


FOUND = [_call("словник", слово="ботей"), _call("final_answer", text="отара"), "отара",
         json.dumps({"kind": "supported", "reason": "ок"}, ensure_ascii=False)]
MISSING = [_call("словник", слово="абахта"), _call("final_answer", text="немає в довіднику"),
           "немає в довіднику", json.dumps({"kind": "abstain", "reason": "нема"}, ensure_ascii=False)]


def test_the_sequence_is_monotone_from_zero():
    events = _events(FOUND)
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert {e["protocol"] for e in events} == {PROTOCOL}
    assert {e["runId"] for e in events} == {"t"}


def test_the_final_answer_is_not_a_tool_call():
    """Термінатор циклу — не інструмент даних: подія про нього ламала б парність called/result."""
    events = _events(FOUND)
    called = [e for e in events if e["type"] == "tool.called"]
    results = [e for e in events if e["type"] == "tool.result"]
    assert len(called) == len(results) == 1
    assert called[0]["payload"]["tool"] == "словник"


def test_the_tool_result_carries_typed_absence():
    assert [e["payload"]["found"] for e in _events(FOUND) if e["type"] == "tool.result"] == [True]
    assert [e["payload"]["found"] for e in _events(MISSING) if e["type"] == "tool.result"] == [False]


def test_abstain_arrives_as_a_state_not_an_error():
    """Головний інваріант контракту: відмова доїжджає окремим станом, а не як помилка виконання."""
    events = _events(MISSING, absent_answer=True)
    outcome = next(e for e in events if e["type"] == "task.outcome")
    assert outcome["payload"]["outcome"] == "abstain"
    assert outcome["payload"]["evidence"] is False
    assert not [e for e in events if e["type"] == "run.error"]


def test_the_verdict_travels_as_a_kind():
    verdicts = [e["payload"] for e in _events(MISSING) if e["type"] == "verify.verdict"]
    assert verdicts and verdicts[0]["kind"] == "abstain" and verdicts[0]["accepted"] is True


def test_the_route_event_reports_the_lane_not_just_the_model():
    """`StepRecord.lane` з'явився саме тут: проєкція показала, що яруса в трасі бракує."""
    routes = [e["payload"] for e in _events(FOUND) if e["type"] == "route.decided"]
    assert routes and {r["lane"] for r in routes} <= {"lapa", "mamay", "unknown"}
    assert any(r["lane"] in ("lapa", "mamay") for r in routes)


def test_an_unknown_lane_is_normalised_not_leaked():
    from ploshcha_sim.ports.trace import StepRecord

    record = StepRecord(run_id="t", tick=0, agent="orchestrator", stage="select", model="m",
                        lane="щось-нове", prompt="", raw_output="")
    events = project_run([record], None, run_id="t", ts=TS)
    assert events[0]["payload"]["lane"] == "unknown"


def test_projection_without_a_scene_starts_straight_at_the_work():
    assert _events(FOUND)[0]["type"] != "run.started"


def test_the_committed_fixture_is_reproducible():
    """Фікстуру перевіряє TS-валідатор контракту, тож вона мусить збігатися з тим, що ядро віддає
    зараз: розходження означає, що доказ сумісності протух."""
    if not FIXTURE.exists():
        pytest.skip("фікстуру ще не згенеровано")
    import subprocess

    root = Path(__file__).resolve().parents[1]
    before = FIXTURE.read_text(encoding="utf-8")
    subprocess.run(["uv", "run", "python", "scripts/project_fixture.py", "--ts", TS],
                   cwd=root, check=True, capture_output=True)
    assert FIXTURE.read_text(encoding="utf-8") == before, "перегенеруй фікстуру: ядро змінило вивід"
