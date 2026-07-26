"""Чи стикуються шви: perceive -> importance -> памʼять -> recall -> act -> reflect."""

import json

from ploshcha_sim.adapters import FakeLlm, InMemoryTrace
from ploshcha_sim.agents import (
    act,
    perceive,
    rate_importance,
    recall,
    reflect,
    should_reflect,
    to_memories,
)
from ploshcha_sim.domain import Utterance


def test_full_memory_cycle_feeds_the_action(world):
    trace = InMemoryTrace()
    world.agents["koval"].location = "ploshcha"
    world.agents["mati"].location = "ploshcha"
    world.utterances.append(
        Utterance(tick=0, speaker="mati", to=["koval"], text="весілля в неділю", poi="ploshcha")
    )

    observed = perceive(world, "koval")
    ratings = rate_importance(observed).ratings
    world.agents["koval"].memory += to_memories(observed, ratings, world.tick, "koval")

    hits = recall(world, "koval", observations=observed, k=3, trace=trace)
    assert hits and "весілля" in hits[0].item.text

    llm = FakeLlm([json.dumps({"type": "speak", "to": ["mati"], "text": "чув про весілля"})])
    r = act(world, "koval", llm, observations=observed, memories=[h.item for h in hits], trace=trace)

    assert r.schema_valid and r.world_valid
    assert "весілля" in llm.calls[0]["prompt"]  # памʼять реально дійшла до промпту
    assert [rec.stage for rec in trace.records] == ["recall", "act"]


def test_accumulated_observations_eventually_trigger_reflection(world):
    world.agents["koval"].location = "ploshcha"
    world.agents["mati"].location = "ploshcha"
    agent = world.agents["koval"]

    for tick in range(6):
        world.tick = tick
        world.utterances.append(
            Utterance(tick=tick, speaker="mati", to=["koval"], text="весілля в неділю", poi="ploshcha")
        )
        observed = perceive(world, "koval")
        ratings = rate_importance(observed).ratings
        agent.memory += to_memories(observed, ratings, tick, "koval", seq_start=len(agent.memory))

    assert should_reflect(agent)

    llm = FakeLlm([
        json.dumps({"questions": ["Що там за весілля?"]}),
        json.dumps({"insights": [{"text": "У неділю буде весілля", "evidence": [1]}]}),
    ])
    r = reflect(world, "koval", llm)
    agent.memory += r.items

    assert r.items and r.items[0].evidence
    assert not should_reflect(agent)  # рефлексія скинула накопичення
