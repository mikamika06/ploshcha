from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from .llm import LlmPort

StepKind = Literal[
    "parse", "classify", "select", "ground", "decide",
    "generate", "synthesize", "judge", "gate",
]

STEP_KINDS: tuple[StepKind, ...] = (
    "parse", "classify", "select", "ground", "decide",
    "generate", "synthesize", "judge", "gate",
)

Tier = Literal["none", "wire", "strict"]


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


class EffortPolicy(ABC):
    @abstractmethod
    def effort(self, kind: StepKind) -> EffortConfig: ...
