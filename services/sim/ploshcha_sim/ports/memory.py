from abc import ABC, abstractmethod

from ..domain.memory import MemoryItem


class MemoryStore(ABC):
    @abstractmethod
    def add(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[MemoryItem]: ...


class Notebook(ABC):
    """Памʼять на одну задачу: пише кроки й піднімає їх зі скором по складових."""

    @abstractmethod
    def note(self, text: str, tick: int, *, sort: str = "fact",
             evidence: list[str] | None = None) -> MemoryItem | None: ...

    @abstractmethod
    def recall(self, query: str, now_tick: int, k: int | None = None): ...
