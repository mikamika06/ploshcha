"""Стан світу ПЛОЩІ. Деталі — docs/sprints/S0-core.md."""

from typing import Literal

from pydantic import BaseModel, Field

from .memory import MemoryItem

TimeOfDay = Literal["dawn", "morning", "noon", "afternoon", "evening", "night"]

PHASES: tuple[TimeOfDay, ...] = ("dawn", "morning", "noon", "afternoon", "evening", "night")
TICKS_PER_PHASE = 1


def phase_of(tick: int) -> TimeOfDay:
    return PHASES[(tick // TICKS_PER_PHASE) % len(PHASES)]


class Persona(BaseModel):
    role: str = Field(description="koval | mirosh | pip | mati | ...")
    name: str
    bio: str = ""
    traits: list[str] = Field(default_factory=list)


class POI(BaseModel):
    id: str
    name: str
    kind: str = Field(description="forge | well | square | church | mill | board | ...")
    usable: bool = True


class PlanStep(BaseModel):
    from_tick: int = Field(ge=0)
    intent: str = Field(min_length=1)


class AgentState(BaseModel):
    id: str
    persona: Persona
    location: str
    activity: str = "idle"
    plan: list[PlanStep] = Field(default_factory=list)
    relationships: dict[str, float] = Field(default_factory=dict)
    memory: list[MemoryItem] = Field(default_factory=list)


class Utterance(BaseModel):
    tick: int
    speaker: str
    to: list[str] = Field(default_factory=list)
    text: str
    poi: str


class WorldState(BaseModel):
    tick: int = Field(default=0, ge=0)
    time_of_day: TimeOfDay = "dawn"
    pois: dict[str, POI] = Field(default_factory=dict)
    agents: dict[str, AgentState] = Field(default_factory=dict)
    board: list[str] = Field(default_factory=list)
    utterances: list[Utterance] = Field(default_factory=list)

    def agents_at(self, poi_id: str) -> list[str]:
        return sorted(a.id for a in self.agents.values() if a.location == poi_id)

    def utterances_at(self, poi_id: str, tick: int) -> list[Utterance]:
        return [u for u in self.utterances if u.poi == poi_id and u.tick == tick]
