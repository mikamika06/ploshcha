"""Порт охорони: політика довіри до входу й рішення, що з ним робити.

Окремий порт, а не функція в оркестраторі, бо політика — це те, що застосунок налаштовує: одна річ
читати власну KB, інша — чужий документ із інтернету.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..domain.injection import Screening


class Policy(BaseModel):
    """`block` — не єдиний варіант: на дослідницькому прогоні цінніше ПОМІТИТИ, ніж зупинити."""

    wrap_untrusted: bool = True
    on_threat: str = "note"
    scope_tools: bool = True


class GuardPort(ABC):
    @abstractmethod
    def screen(self, text: str, *, tag: str | None = None) -> Screening: ...

    @abstractmethod
    def prepare(self, text: str, *, tag: str, trust: str = "untrusted") -> str:
        """Підготувати чужий текст до вкладення в промпт."""
