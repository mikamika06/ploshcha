import json

from ploshcha_sim.adapters import FakeLlm, InMemoryTrace, PresetEffort, single_model_router
from ploshcha_sim.agents import verify


def router_with(script):
    return single_model_router(FakeLlm(script, model="judge"))


def test_accept_when_model_says_accepted():
    v = verify("задача", "відповідь", router_with([json.dumps({"accepted": True, "reason": "ok"})]), PresetEffort())
    assert v.accepted is True and v.reason == "ok"


def test_reject_when_model_refutes():
    v = verify("задача", "відповідь", router_with([json.dumps({"accepted": False, "reason": "хибно"})]), PresetEffort())
    assert v.accepted is False and v.reason == "хибно"


def test_parse_fail_defaults_to_reject():
    v = verify("задача", "відповідь", router_with(["не JSON"]), PresetEffort())
    assert v.accepted is False and v.reason == "verify_parse_fail"


def test_verify_routes_to_judge_model():
    llm = FakeLlm([json.dumps({"accepted": True, "reason": "ok"})], model="judge")
    verify("q", "a", single_model_router(llm), PresetEffort(), seed=3)
    assert llm.calls[0]["seed"] == 3


def test_verify_traces():
    trace = InMemoryTrace()
    verify("q", "a", router_with([json.dumps({"accepted": True, "reason": "ok"})]), PresetEffort(), trace=trace)
    assert trace.records[0].stage == "judge" and trace.records[0].agent == "verifier"


def test_verify_includes_evidence_in_prompt():
    llm = FakeLlm([json.dumps({"accepted": True, "reason": "ok"})], model="judge")
    verify("q", "a", single_model_router(llm), PresetEffort(),
           evidence=[{"call": {"tool": "check_date"}, "result": {"matches": True}}])
    assert "matches" in llm.calls[0]["prompt"]
