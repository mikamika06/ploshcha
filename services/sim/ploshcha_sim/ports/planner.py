from abc import ABC, abstractmethod

from .router import StepKind


class Planner(ABC):
    @abstractmethod
    def next_kind(self, state) -> StepKind: ...
