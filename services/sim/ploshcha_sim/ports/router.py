from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from .llm import LlmPort

StepKind = Literal[
    "parse", "classify", "select", "ground", "decide",
    "generate", "synthesize", "judge", "gate", "speak",
]

STEP_KINDS: tuple[StepKind, ...] = (
    "parse", "classify", "select", "ground", "decide",
    "generate", "synthesize", "judge", "gate", "speak",
)

Tier = Literal["none", "wire", "strict"]

# Ярус моделі — НЕ те саме, що Tier (той про обмеження схеми).
ModelLane = Literal["lapa", "mamay", "unknown"]

ESCALATE_KIND: dict[StepKind, StepKind] = {
    "parse": "decide",
    "classify": "decide",
    "select": "decide",
    "ground": "decide",
    "gate": "decide",
    "decide": "synthesize",
}


def escalated(kind: StepKind) -> StepKind | None:
    target = ESCALATE_KIND.get(kind)
    return target if target != kind else None


class EffortConfig(BaseModel):
    think_tokens: int = 0
    force_thinking: bool = False
    max_tokens: int = 256
    temperature: float = 0.0
    tier: Tier = "strict"
    samples: int = 1
    verify: bool = False


class ModelRouter(ABC):
    @abstractmethod
    def route(self, kind: StepKind) -> LlmPort: ...

    def lane(self, kind: StepKind) -> ModelLane:
        return "unknown"


class EffortPolicy(ABC):
    @abstractmethod
    def effort(self, kind: StepKind) -> EffortConfig: ...
