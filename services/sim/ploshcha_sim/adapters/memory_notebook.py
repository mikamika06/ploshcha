from ..domain.memory import MemoryItem
from ..domain.retrieval import RelevanceMode, Retrieved, retrieve
from ..ports.memory import MemoryStore, Notebook

NOTEBOOK_K = 6
DROPPED_SHOWN = 3
W_RECENCY = 0.3
W_IMPORTANCE = 0.3
W_RELEVANCE = 1.0

IMPORTANCE = {
    "fact": 8,
    "empty": 5,
    "error": 4,
    "incident": 5,
    "plan": 6,
}


class Recall:
    def __init__(self, hits: list[Retrieved], candidates: int, dropped: list[Retrieved]):
        self.hits = hits
        self.candidates = candidates
        self.dropped = dropped

    def anchors(self) -> list[str]:
        return [hit.item.id for hit in self.hits]

    def as_trace(self) -> dict:
        return {
            "candidates": self.candidates,
            "retrieved": [
                {"id": h.item.id, "total": round(h.total, 4), "relevance": round(h.relevance, 4),
                 "recency": round(h.recency, 4), "importance": round(h.importance, 4)}
                for h in self.hits
            ],
            "dropped": [
                {"id": d.item.id, "total": round(d.total, 4), "relevance": round(d.relevance, 4)}
                for d in self.dropped
            ],
        }


class NotebookMemory(MemoryStore, Notebook):
    def __init__(self, mode: RelevanceMode = "chargram", k: int = NOTEBOOK_K,
                 embedder=None, w_recency: float = W_RECENCY,
                 w_importance: float = W_IMPORTANCE, w_relevance: float = W_RELEVANCE):
        self.mode = mode
        self.k = k
        self.embedder = embedder
        self.weights = {"w_recency": w_recency, "w_importance": w_importance,
                        "w_relevance": w_relevance}
        self._items: list[MemoryItem] = []
        self._texts: set[str] = set()
        self._seq = 0

    @property
    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def add(self, item: MemoryItem) -> None:
        if item.text in self._texts:
            return
        self._texts.add(item.text)
        self._items.append(item)

    def note(self, text: str, tick: int, *, sort: str = "fact",
             evidence: list[str] | None = None) -> MemoryItem | None:
        if not text.strip() or text in self._texts:
            return None
        item = MemoryItem(
            id=f"n{self._seq}", tick=tick, kind="observation", text=text,
            importance=IMPORTANCE.get(sort, 5), provenance="notebook",
            evidence=evidence or [],
        )
        self._seq += 1
        self.add(item)
        return item

    def retrieve(self, query: str, k: int) -> list[MemoryItem]:
        return [hit.item for hit in self.recall(query, self._now(), k).hits]

    def recall(self, query: str, now_tick: int, k: int | None = None) -> Recall:
        limit = self.k if k is None else k
        if not self._items:
            return Recall([], 0, [])
        query_embedding = None
        if self.mode == "cosine" and self.embedder is not None:
            query_embedding = self.embedder.embed([query])[0]
        ranked = retrieve(self._items, query, now_tick, len(self._items),
                          mode=self.mode, query_embedding=query_embedding,
                          normalize=False, **self.weights)
        return Recall(ranked[:limit], len(self._items), ranked[limit:limit + DROPPED_SHOWN])

    def _now(self) -> int:
        return max((item.tick for item in self._items), default=0)
