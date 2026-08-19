import json
import threading
import time

import pytest

from ploshcha_sim.adapters import FakeLlm, FakeToolbox, PresetEffort
from ploshcha_sim.adapters.projector import POI_OF_STAGE, StreamProjector, poi_of_stage
from ploshcha_sim.adapters.queue_sqlite import SqliteQueue
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.domain.governor import Governor
from ploshcha_sim.live import BusTrace, EventBus, LiveRunner, handle_command
from ploshcha_sim.ports.router import STEP_KINDS


def tc(tool, **args):
    return json.dumps({"tool": tool, **args}, ensure_ascii=False)


# ── шина ──────────────────────────────────────────────────────────────────────

def test_bus_keeps_order_and_cursor():
    bus = EventBus()
    bus.publish([{"n": 1}, {"n": 2}])
    events, cursor = bus.since(0)
    assert [e["n"] for e in events] == [1, 2]
    assert cursor == 2


def test_late_viewer_gets_the_tail_not_the_history():
    bus = EventBus()
    bus.publish([{"n": i} for i in range(5)])
    cursor = bus.tail_cursor()
    bus.publish({"n": 99})
    events, _ = bus.since(cursor)
    assert [e["n"] for e in events] == [99]


def test_overflow_is_visible_not_hidden():
    bus = EventBus(capacity=3)
    bus.publish([{"n": i} for i in range(5)])
    events, _ = bus.since(0)
    assert len(events) == 3
    assert bus.dropped == 2


def test_reconnect_from_cursor_reads_the_missed_tail():
    bus = EventBus()
    bus.publish([{"n": 0}, {"n": 1}])
    events, cursor = bus.since(0)
    bus.publish([{"n": 2}, {"n": 3}])
    missed, _ = bus.since(cursor)
    assert [e["n"] for e in missed] == [2, 3]


def test_wait_returns_on_publish():
    bus = EventBus()
    got: list = []

    def waiter():
        events, _ = bus.wait(0, timeout=3.0)
        got.extend(events)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    bus.publish({"n": 7})
    t.join(3.0)
    assert [e["n"] for e in got] == [7]


def test_wait_times_out_without_events():
    bus = EventBus()
    events, cursor = bus.wait(0, timeout=0.05)
    assert events == [] and cursor == 0


# ── POI ───────────────────────────────────────────────────────────────────────

def test_every_step_kind_maps_to_a_poi():
    for kind in STEP_KINDS:
        assert kind in POI_OF_STAGE, f"kind {kind} без POI"


def test_poi_mapping_is_diegetic():
    assert poi_of_stage("recall") == "well"
    assert poi_of_stage("judge") == "church"
    assert poi_of_stage("select") == "forge"
    assert poi_of_stage("synthesize") == "square"
    assert poi_of_stage("невідома") == "square"


# ── потоковий проєктор ────────────────────────────────────────────────────────

def _orch(replies, trace=None):
    llm = FakeLlm(replies, model="fake")
    return Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(),
                        verifier=False, trace=trace, run_id="r")


def test_stream_projector_emits_while_the_run_is_going():
    bus = EventBus()
    proj = StreamProjector("r", "2026-01-01T00:00:00Z")
    trace = BusTrace(bus, proj)
    _orch([tc("lookup_fact", entity="X"), tc("final_answer", text="R")], trace=trace).run("q")
    events, _ = bus.since(0)
    types = [e["type"] for e in events]
    assert "route.decided" in types
    assert "tool.called" in types
    assert "tool.result" in types, "результат інструмента мусить доїхати В ПОТОЦІ"
    assert "agent.moved" in types


def test_seq_is_monotonic_without_gaps():
    bus = EventBus()
    proj = StreamProjector("r", "2026-01-01T00:00:00Z")
    trace = BusTrace(bus, proj)
    _orch([tc("lookup_fact", entity="X"), tc("final_answer", text="R")], trace=trace).run("q")
    events, _ = bus.since(0)
    assert [e["seq"] for e in events] == list(range(len(events)))


def test_tool_result_carries_ok_and_found():
    bus = EventBus()
    proj = StreamProjector("r", "2026-01-01T00:00:00Z")
    trace = BusTrace(bus, proj)
    _orch([tc("lookup_fact", entity="X"), tc("final_answer", text="R")], trace=trace).run("q")
    events, _ = bus.since(0)
    res = next(e for e in events if e["type"] == "tool.result")
    assert res["payload"]["tool"] == "lookup_fact"
    assert res["payload"]["ok"] is True
    assert "found" in res["payload"]


# ── губернатор і команди ──────────────────────────────────────────────────────

class _Runner(LiveRunner):
    pass


@pytest.fixture
def runner(tmp_path):
    bus = EventBus()
    queue = SqliteQueue(str(tmp_path / "q.db"))

    def make(trace, run_id):
        return _orch([tc("final_answer", text="R")], trace=trace)

    return bus, queue, _Runner(bus, queue, make, governor=Governor(max_tokens=1))


def test_runner_starts_paused(runner):
    _, _, r = runner
    assert r.state == "paused"
    assert r.health()["state"] == "paused"


def test_cap_reached_degrades_and_pauses(runner):
    bus, _, r = runner
    r.governor.record(tokens=10_000)
    r.resume()
    r.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and r.stopped_reason is None:
        time.sleep(0.05)
    r.stop()
    assert r.stopped_reason is not None, "стеля мусить зупинити цикл"
    assert r.state == "stopped"
    types = [e["type"] for e in bus.since(0)[0]]
    assert "run.degraded" in types


def test_resume_after_a_cap_is_refused(runner):
    _, _, r = runner
    r.stopped_reason = "межа токенів"
    r.resume()
    assert r.state == "paused", "після стелі resume не має тихо продовжувати"


def test_topic_command_enqueues_work(runner):
    _, queue, r = runner
    code, body = handle_command({"kind": "topic", "text": "Чому криниця пересохла?"}, r)
    assert code == 200 and body["ok"] is True
    assert queue.stats().get("pending", 0) == 1


def test_empty_topic_is_rejected(runner):
    _, _, r = runner
    code, body = handle_command({"kind": "topic", "text": "   "}, r)
    assert code == 400 and "error" in body


def test_unknown_command_is_rejected(runner):
    _, _, r = runner
    code, _ = handle_command({"kind": "дивна"}, r)
    assert code == 400


def test_pause_resume_roundtrip(runner):
    _, _, r = runner
    handle_command({"kind": "resume"}, r)
    assert r.state == "running"
    handle_command({"kind": "pause"}, r)
    assert r.state == "paused"


def test_health_reports_spend_and_caps(runner):
    _, _, r = runner
    r.governor.record(tokens=123, usd=0.5)
    h = r.health()
    assert h["spend"]["tokens"] == 123
    assert h["caps"]["maxTokens"] == 1
    assert "queue" in h and "events" in h


# ── мертві айтеми ─────────────────────────────────────────────────────────────

def test_a_dead_item_can_be_brought_back(runner):
    """У базі лежав `dead: 1` від уже виправленого `TypeError`, і оживити його можна було лише
    руками в SQLite."""
    _, queue, r = runner
    queue.put("k", {"task": "тема"})
    for _ in range(9):
        queue.lease("w")
        queue.fail("k", "зламалось")
    assert queue.stats().get("dead") == 1

    code, body = handle_command({"kind": "requeue"}, r)
    assert code == 200 and body["requeued"] == 1
    assert queue.stats().get("pending") == 1
    assert not queue.stats().get("dead")


def test_requeue_can_target_one_key(runner):
    _, queue, r = runner
    for key in ("a", "b"):
        queue.put(key, {"task": key})
        for _ in range(9):
            queue.lease("w")
            queue.fail(key, "зламалось")
    code, body = handle_command({"kind": "requeue", "key": "a"}, r)
    assert code == 200 and body["requeued"] == 1
    assert queue.stats().get("dead") == 1
