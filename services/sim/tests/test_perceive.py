"""Perceive: діф двох станів світу -> спостереження. Без LLM."""

import pytest

from ploshcha_sim.agents import perceive, to_memories
from ploshcha_sim.domain import Utterance


def said(world, speaker: str, text: str, to: list[str] | None = None, poi: str = "ploshcha"):
    world.utterances.append(
        Utterance(tick=world.tick, speaker=speaker, to=to or [], text=text, poi=poi)
    )


def test_addressed_speech_is_marked_as_to_me(world):
    world.agents["koval"].location = "ploshcha"
    world.agents["mati"].location = "ploshcha"
    said(world, "mati", "ходи їсти", to=["koval"])
    assert perceive(world, "koval") == ["mati (Оксана) сказав тобі: «ходи їсти»"]


def test_broadcast_is_marked_as_to_all(world):
    world.agents["koval"].location = "ploshcha"
    world.agents["mati"].location = "ploshcha"
    said(world, "mati", "толока в неділю")
    assert "сказав усім" in perceive(world, "koval")[0]


def test_speech_to_third_party_is_overheard(world):
    for a in ("koval", "mati", "did"):
        world.agents[a].location = "ploshcha"
    said(world, "mati", "чув про весілля?", to=["did"])
    assert perceive(world, "koval")[0].startswith("Ти почув")


def test_own_speech_is_not_perceived(world):
    world.agents["koval"].location = "ploshcha"
    said(world, "koval", "гей")
    assert perceive(world, "koval") == []


def test_speech_elsewhere_is_not_heard(world):
    world.agents["koval"].location = "kuznya"
    world.agents["mati"].location = "ploshcha"
    said(world, "mati", "гей", poi="ploshcha")
    assert perceive(world, "koval") == []


def test_stale_speech_is_not_heard(world):
    world.agents["koval"].location = "ploshcha"
    world.agents["mati"].location = "ploshcha"
    said(world, "mati", "старе")
    world.tick = 5
    assert perceive(world, "koval") == []


# ── діф проти попереднього стану ─────────────────────────────────────────────


def test_arrival_and_departure_detected(world):
    prev = world.model_copy(deep=True)
    world.agents["koval"].location = "ploshcha"
    world.agents["mati"].location = "ploshcha"
    prev.agents["koval"].location = "ploshcha"
    prev.agents["mati"].location = "kuznya"
    prev.agents["did"].location = "ploshcha"
    world.agents["did"].location = "richka"
    got = perceive(world, "koval", previous=prev)
    assert "mati (Оксана) підійшов." in got
    assert "did (Свирид) пішов." in got


def test_own_move_is_perceived(world):
    prev = world.model_copy(deep=True)
    prev.agents["koval"].location = "kuznya"
    world.agents["koval"].location = "ploshcha"
    assert perceive(world, "koval", previous=prev)[0] == "Ти прийшов у ploshcha (Площа)."


def test_board_diff_reports_only_new_topics(world):
    prev = world.model_copy(deep=True)
    prev.board = ["стара тема"]
    world.board = ["стара тема", "нова тема"]
    got = perceive(world, "koval", previous=prev)
    assert got == ["На Дошці зʼявилось: «нова тема»"]


def test_without_previous_whole_board_is_new(world):
    """previous=None означає «перше сприйняття», не «нічого не змінилось»."""
    world.board = ["тема"]
    assert perceive(world, "koval") == ["На Дошці зʼявилось: «тема»"]


def test_labels_carry_id_and_name(world):
    world.agents["koval"].location = "ploshcha"
    world.agents["mati"].location = "ploshcha"
    said(world, "mati", "гей", to=["koval"])
    assert "mati (Оксана)" in perceive(world, "koval")[0]


# ── перетворення в памʼять ───────────────────────────────────────────────────


def test_to_memories_ids_are_unique_and_traceable():
    items = to_memories(["a", "b"], [3, 7], tick=4, agent_id="koval")
    assert [m.id for m in items] == ["koval:t4:0", "koval:t4:1"]
    assert [m.importance for m in items] == [3, 7]
    assert all(m.kind == "observation" and m.tick == 4 for m in items)


def test_to_memories_rejects_length_mismatch():
    with pytest.raises(ValueError):
        to_memories(["a", "b"], [3], tick=0, agent_id="koval")


def test_to_memories_seq_start_avoids_collisions():
    first = to_memories(["a"], [3], tick=1, agent_id="koval")
    second = to_memories(["b"], [3], tick=1, agent_id="koval", seq_start=len(first))
    assert first[0].id != second[0].id
