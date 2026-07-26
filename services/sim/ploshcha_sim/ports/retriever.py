from abc import ABC, abstractmethod

from ..domain.memory import MemoryItem
from ..domain.retrieval import Retrieved


class Retriever(ABC):
    @abstractmethod
    def rank(self, query: str, items: list[MemoryItem], k: int) -> list[Retrieved]: ...
