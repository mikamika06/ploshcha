import pytest

from ploshcha_sim.adapters import FakeLlm, PresetEffort, profile_router, single_model_router
from ploshcha_sim.adapters.router_profile import HIGH_KINDS, LAPA_KINDS, MAMAY_KINDS, ProfileRouter
from ploshcha_sim.ports.router import STEP_KINDS


@pytest.fixture
def lapa():
    return FakeLlm([], model="lapa")


@pytest.fixture
def mamay():
    return FakeLlm([], model="mamay")


def test_profile_router_sends_closed_kinds_to_lapa(lapa, mamay):
    r = profile_router(lapa, mamay)
    for k in ("parse", "classify", "select", "ground", "gate"):
        assert r.route(k).model == "lapa"


def test_profile_router_sends_open_kinds_to_mamay(lapa, mamay):
    r = profile_router(lapa, mamay)
    for k in ("decide", "generate", "synthesize", "judge"):
        assert r.route(k).model == "mamay"


def test_gate_goes_to_lapa_not_mamay(lapa, mamay):
    assert profile_router(lapa, mamay).route("gate").model == "lapa"


def test_router_covers_every_step_kind(lapa, mamay):
    r = profile_router(lapa, mamay)
    for k in STEP_KINDS:
        assert r.route(k).model in ("lapa", "mamay")


def test_router_is_deterministic(lapa, mamay):
    r = profile_router(lapa, mamay)
    assert r.route("select").model == r.route("select").model == "lapa"


def test_lapa_and_mamay_kinds_partition_all(lapa, mamay):
    assert set(LAPA_KINDS) | set(MAMAY_KINDS) == set(STEP_KINDS)
    assert set(LAPA_KINDS) & set(MAMAY_KINDS) == set()


def test_single_model_router_sends_all_to_one(lapa):
    r = single_model_router(lapa)
    for k in STEP_KINDS:
        assert r.route(k).model == "lapa"


def test_explicit_mapping_and_default():
    a, b = FakeLlm([], model="a"), FakeLlm([], model="b")
    r = ProfileRouter({"generate": b}, default=a)
    assert r.route("generate").model == "b"
    assert r.route("parse").model == "a"


def test_missing_route_without_default_raises():
    a = FakeLlm([], model="a")
    r = ProfileRouter({"generate": a})
    with pytest.raises(KeyError):
        r.route("parse")


def test_preset_effort_low_for_closed_kinds():
    e = PresetEffort()
    cfg = e.effort("parse")
    assert cfg.think_tokens == 0 and cfg.tier == "strict" and cfg.samples == 1 and cfg.verify is False


def test_preset_effort_high_for_open_kinds():
    e = PresetEffort()
    cfg = e.effort("synthesize")
    assert cfg.think_tokens > 0 and cfg.tier == "wire" and cfg.verify is True


def test_preset_effort_covers_every_kind():
    e = PresetEffort()
    for k in STEP_KINDS:
        assert e.effort(k).max_tokens > 0


def test_high_kinds_get_more_thinking_budget():
    e = PresetEffort()
    assert e.effort("generate").think_tokens > e.effort("parse").think_tokens


def test_preset_effort_high_matches_high_kinds():
    e = PresetEffort()
    for k in HIGH_KINDS:
        assert e.effort(k).think_tokens > 0


def test_effort_config_defaults_to_no_thinking():
    from ploshcha_sim.ports.router import EffortConfig
    assert EffortConfig().think_tokens == 0 and EffortConfig().force_thinking is False
