from abc import ABC, abstractmethod

from ..domain.task import TaskPlan
from .router import StepKind


class Planner(ABC):
    @abstractmethod
    def next_kind(self, state) -> StepKind: ...

    def plan(self, task: str, tools) -> TaskPlan | None:
        return None

    def replan(self, task: str, state) -> TaskPlan | None:
        return None
