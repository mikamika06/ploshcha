from ploshcha_sim.adapters import FakeLlm, FakeToolbox, PresetEffort, single_model_router
from ploshcha_sim.adapters.router_profile import sampling_effort
from ploshcha_sim.agents.orchestrator import Orchestrator
from ploshcha_sim.ports import StepRecord
from ploshcha_sim.ports.router import EffortConfig


def test_fake_llm_records_seed():
    llm = FakeLlm(["ok"])
    llm.generate_structured("p", {"type": "object"}, seed=42)
    assert llm.calls[0]["seed"] == 42


def test_seed_defaults_to_none():
    llm = FakeLlm(["ok"])
    llm.generate("p")
    assert llm.calls[0]["seed"] is None


def test_step_record_carries_seed():
    rec = StepRecord(run_id="r", tick=0, agent="a", stage="act", model="m", prompt="p", raw_output="o", seed=7)
    assert rec.seed == 7


def test_effort_temperature_defaults_greedy():
    assert EffortConfig().temperature == 0.0
    assert PresetEffort().effort("select").temperature == 0.0
    assert PresetEffort().effort("judge").temperature == 0.0


def test_sampling_effort_sets_temperature_on_both_tiers():
    eff = sampling_effort(0.7)
    assert eff.effort("select").temperature == 0.7
    assert eff.effort("judge").temperature == 0.7
    assert eff.effort("select").tier == "strict"
    assert eff.effort("judge").think_tokens == 768


def test_orchestrator_passes_temperature_to_port():
    llm = FakeLlm(['{"tool": "final_answer", "text": "ok"}'])
    orch = Orchestrator(single_model_router(llm), sampling_effort(0.7), FakeToolbox(), verifier=False)
    orch.run("задача", seed=3)
    assert llm.calls[0]["temperature"] == 0.7
    assert llm.calls[0]["seed"] == 3


def test_orchestrator_greedy_by_default():
    llm = FakeLlm(['{"tool": "final_answer", "text": "ok"}'])
    orch = Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(), verifier=False)
    orch.run("задача", seed=0)
    assert llm.calls[0]["temperature"] == 0.0
