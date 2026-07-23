"""Редьюсер: чистота, детермінізм, дії, таксономія відмов."""

from ploshcha_sim.domain import (
    MoveTo,
    PostToBoard,
    Reflect,
    Speak,
    UseObject,
    Wait,
    phase_of,
    tick,
    validate_action,
)
from ploshcha_sim.domain.reducer import (
    NOT_AT_POI,
    POI_NOT_USABLE,
    RECIPIENT_NOT_HERE,
    SPEAKING_TO_SELF,
    UNKNOWN_AGENT,
    UNKNOWN_POI,
    UNKNOWN_RECIPIENT,
)

# ── чистота й детермінізм ────────────────────────────────────────────────────


def test_reducer_does_not_mutate_input(world):
    before = world.model_dump_json()
    tick(world, {"koval": MoveTo(poi="ploshcha")})
    assert world.model_dump_json() == before, "редьюсер мутував вхідний стан"


def test_reducer_is_deterministic(world):
    d = {"koval": MoveTo(poi="ploshcha"), "mati": Wait()}
    a = tick(world, d).state.model_dump_json()
    b = tick(world, d).state.model_dump_json()
    assert a == b


def test_agent_order_does_not_affect_result(world):
    d1 = {"koval": MoveTo(poi="ploshcha"), "mati": MoveTo(poi="kuznya")}
    d2 = {"mati": MoveTo(poi="kuznya"), "koval": MoveTo(poi="ploshcha")}
    assert tick(world, d1).state.model_dump_json() == tick(world, d2).state.model_dump_json()


# ── просування часу ──────────────────────────────────────────────────────────


def test_tick_increments_and_advances_phase(world):
    r = tick(world, {})
    assert r.state.tick == 1
    assert r.state.time_of_day == phase_of(1) == "morning"


def test_phase_cycles_over_day():
    assert phase_of(0) == "dawn"
    assert phase_of(5) == "night"
    assert phase_of(6) == "dawn"


# ── застосування кожної дії ──────────────────────────────────────────────────


def test_move_to_changes_location(world):
    r = tick(world, {"koval": MoveTo(poi="ploshcha")})
    assert r.state.agents["koval"].location == "ploshcha"
    assert r.outcomes[0].ok


def test_use_object_requires_presence_and_marks_activity(world):
    r = tick(world, {"koval": UseObject(poi="kuznya")})
    assert r.state.agents["koval"].activity == "using:kuznya"


def test_speak_logs_utterance_at_agent_location(world):
    r = tick(world, {"mati": Speak(to=["did"], text="Добридень")})
    u = r.state.utterances[0]
    assert (u.speaker, u.to, u.poi, u.tick) == ("mati", ["did"], "ploshcha", 0)


def test_speak_broadcast_allowed_without_recipients(world):
    assert tick(world, {"koval": Speak(text="гей")}).outcomes[0].ok


def test_post_to_board_appends_topic(world):
    r = tick(world, {"did": PostToBoard(topic="Завтра толока")})
    assert r.state.board == ["Завтра толока"]


def test_reflect_and_wait_do_not_change_world(world):
    r = tick(world, {"koval": Reflect(), "mati": Wait(reason="нема кого")})
    assert r.state.agents["koval"].activity == "reflecting"
    assert r.state.agents["mati"].activity == "waiting"
    assert r.state.board == [] and r.state.utterances == []


# ── таксономія відмов (джерело метрики irrelevance, Вісь A) ──────────────────


def test_reject_unknown_poi(world):
    assert validate_action(world, "koval", MoveTo(poi="mars")) == UNKNOWN_POI


def test_reject_unknown_agent(world):
    assert validate_action(world, "primara", Wait()) == UNKNOWN_AGENT


def test_reject_use_object_when_not_there(world):
    assert validate_action(world, "mati", UseObject(poi="kuznya")) == NOT_AT_POI


def test_reject_use_of_non_usable_poi(world):
    world.agents["koval"].location = "richka"
    assert validate_action(world, "koval", UseObject(poi="richka")) == POI_NOT_USABLE


def test_reject_speak_to_unknown_recipient(world):
    assert validate_action(world, "mati", Speak(to=["nikoho"], text="?")) == UNKNOWN_RECIPIENT


def test_reject_speak_to_distant_recipient(world):
    assert validate_action(world, "mati", Speak(to=["koval"], text="?")) == RECIPIENT_NOT_HERE


def test_reject_speaking_to_self(world):
    assert validate_action(world, "mati", Speak(to=["mati"], text="?")) == SPEAKING_TO_SELF


def test_rejected_action_leaves_world_untouched(world):
    r = tick(world, {"koval": MoveTo(poi="mars")})
    assert r.state.agents["koval"].location == "kuznya"
    assert r.rejected[0].reason == UNKNOWN_POI
    assert r.state.tick == 1, "тік іде далі навіть якщо дія відхилена"


def test_outcomes_cover_every_decision(world):
    r = tick(world, {"koval": Wait(), "mati": MoveTo(poi="mars"), "did": Reflect()})
    assert {o.agent for o in r.outcomes} == {"koval", "mati", "did"}
    assert sum(o.ok for o in r.outcomes) == 2
