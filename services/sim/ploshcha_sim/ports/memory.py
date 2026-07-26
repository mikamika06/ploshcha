from abc import ABC, abstractmethod

from ..domain.memory import MemoryItem


class MemoryStore(ABC):
    @abstractmethod
    def add(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[MemoryItem]: ...
