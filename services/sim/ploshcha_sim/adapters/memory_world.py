from ..domain.memory import MemoryItem
from ..ports.memory import MemoryStore
from ..ports.retriever import Retriever
from .retriever_basic import default_hybrid


class WorldStateMemory(MemoryStore):
    def __init__(self, items: list[MemoryItem] | None = None, retriever: Retriever | None = None):
        self.items = items if items is not None else []
        self.retriever = retriever or default_hybrid()

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)

    def retrieve(self, query: str, k: int) -> list[MemoryItem]:
        return [r.item for r in self.retriever.rank(query, self.items, k)]
