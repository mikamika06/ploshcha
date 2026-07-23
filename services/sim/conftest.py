"""Фікстури + шлях до пакета."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from ploshcha_sim.domain import POI, AgentState, MemoryItem, Persona, WorldState  # noqa: E402


@pytest.fixture
def world() -> WorldState:
    """Мінімальне село: 4 POI (один не-usable), 3 агенти."""
    return WorldState(
        tick=0,
        time_of_day="dawn",
        pois={
            "ploshcha": POI(id="ploshcha", name="Площа", kind="square"),
            "kuznya": POI(id="kuznya", name="Кузня", kind="forge"),
            "krynytsia": POI(id="krynytsia", name="Криниця", kind="well"),
            "richka": POI(id="richka", name="Річка", kind="river", usable=False),
        },
        agents={
            "koval": AgentState(
                id="koval",
                persona=Persona(role="koval", name="Остап", bio="Коваль."),
                location="kuznya",
            ),
            "mati": AgentState(
                id="mati",
                persona=Persona(role="mati", name="Оксана", bio="Мати."),
                location="ploshcha",
            ),
            "did": AgentState(
                id="did",
                persona=Persona(role="did", name="Свирид", bio="Дід."),
                location="ploshcha",
            ),
        },
    )


@pytest.fixture
def memories() -> list[MemoryItem]:
    return [
        MemoryItem(id="m_old_imp", tick=0, kind="observation", text="давнє важливе", importance=9),
        MemoryItem(id="m_new_low", tick=10, kind="observation", text="свіже дрібне", importance=2),
        MemoryItem(id="m_mid", tick=5, kind="reflection", text="середнє", importance=5),
    ]
