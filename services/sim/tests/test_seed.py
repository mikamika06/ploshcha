from ploshcha_sim.adapters import FakeLlm
from ploshcha_sim.ports import StepRecord


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
