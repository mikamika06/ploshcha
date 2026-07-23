"""Крок act: валідна дія, невалідна схема, невалідний світ, fallback, траса."""

import json

from ploshcha_sim.adapters import FakeLlm, InMemoryTrace
from ploshcha_sim.agents import act, build_act_prompt
from ploshcha_sim.domain import ACTION_TYPES
from ploshcha_sim.domain.reducer import RECIPIENT_NOT_HERE, UNKNOWN_POI


def test_valid_action_parsed(world):
    llm = FakeLlm([json.dumps({"type": "move_to", "poi": "ploshcha"})])
    r = act(world, "koval", llm)
    assert r.action.type == "move_to" and r.action.poi == "ploshcha"
    assert r.schema_valid and r.world_valid and not r.fallback


def test_broken_json_falls_back_to_wait(world):
    r = act(world, "koval", FakeLlm(["не JSON зовсім"]))
    assert r.action.type == "wait"
    assert r.schema_valid is False and r.fallback
    assert r.reject_reason == "schema_invalid"


def test_schema_ok_but_world_invalid_falls_back(world):
    llm = FakeLlm([json.dumps({"type": "move_to", "poi": "mars"})])
    r = act(world, "koval", llm)
    assert r.schema_valid and not r.world_valid
    assert r.reject_reason == UNKNOWN_POI
    assert r.action.type == "wait" and r.action.reason == UNKNOWN_POI


def test_world_validation_catches_distant_recipient(world):
    llm = FakeLlm([json.dumps({"type": "speak", "to": ["koval"], "text": "гей"})])
    r = act(world, "mati", llm)
    assert r.reject_reason == RECIPIENT_NOT_HERE and r.fallback


def test_usage_and_latency_propagated(world):
    llm = FakeLlm([json.dumps({"type": "wait"})])
    r = act(world, "koval", llm)
    assert r.usage.prompt_tokens > 0
    assert r.usage.total >= r.usage.prompt_tokens


def test_structured_call_passes_action_schema(world):
    llm = FakeLlm([json.dumps({"type": "wait"})])
    act(world, "koval", llm)
    call = llm.calls[0]
    assert call["structured"] is True
    blob = str(call["schema"])
    for t in ACTION_TYPES:
        assert t in blob


def test_trace_record_written(world):
    trace = InMemoryTrace()
    llm = FakeLlm([json.dumps({"type": "use_object", "poi": "kuznya"})])
    act(world, "koval", llm, trace=trace, run_id="r1", ablation={"reflect": False})
    rec = trace.records[0]
    assert (rec.run_id, rec.agent, rec.stage, rec.tick) == ("r1", "koval", "act", 0)
    assert rec.schema_valid and rec.world_valid
    assert rec.parsed["poi"] == "kuznya"
    assert rec.ablation == {"reflect": False}
    assert rec.prompt and rec.raw_output


def test_trace_seq_increments(world):
    trace = InMemoryTrace()
    llm = FakeLlm([json.dumps({"type": "wait"}), json.dumps({"type": "wait"})])
    act(world, "koval", llm, trace=trace)
    act(world, "mati", llm, trace=trace)
    assert [r.seq for r in trace.records] == [0, 1]


# ── промпт ───────────────────────────────────────────────────────────────────


def test_prompt_lists_only_real_pois_and_neighbours(world):
    p = build_act_prompt(world, "mati")
    assert "ploshcha" in p and "kuznya" in p
    assert "did" in p            # той, хто поруч
    assert "koval" not in p.split("Поруч із тобою:")[1].split("\n")[0]


def test_prompt_marks_non_usable_poi(world):
    p = build_act_prompt(world, "koval")
    usable_line = [l for l in p.split("\n") if "use_object" in l][0]
    assert "richka" not in usable_line


def test_prompt_includes_board_observations_memories(world, memories):
    world.board.append("Завтра толока")
    p = build_act_prompt(world, "koval", observations=["дзвін ударив"], memories=memories[:1])
    assert "Завтра толока" in p and "дзвін ударив" in p and memories[0].text in p


def test_truncation_is_not_counted_as_schema_error(world):
    """Обрізання по max_tokens — провал БЮДЖЕТУ, окремий код."""
    from ploshcha_sim.agents.act import TRUNCATED

    llm = FakeLlm(['{"type":"speak","to":["mati"],"text":"дуже довга каз'], finish_reason="length")
    r = act(world, "koval", llm)
    assert r.schema_valid is False and r.reject_reason == TRUNCATED


def test_prompt_labels_neighbours_with_id_and_name(world):
    p = build_act_prompt(world, "mati")
    assert "did (Свирид)" in p
    assert "САМЕ id" in p
