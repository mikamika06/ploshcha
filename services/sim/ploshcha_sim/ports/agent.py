"""Порт агента: усе, що вміє виконати задачу під бюджетом.

Існує, щоб супервізор не залежав від того, ЧИМ саме є дитина — оркестратором, іншим графом чи
заглушкою в тесті. Без цього `AgentGraph` довелось би тестувати лише через живу модель.
"""

from abc import ABC, abstractmethod

from ..domain.task import Budget, TaskResult


class AgentPort(ABC):
    @abstractmethod
    def run(self, task: str, seed: int = 0, budget: Budget | None = None) -> TaskResult: ...
