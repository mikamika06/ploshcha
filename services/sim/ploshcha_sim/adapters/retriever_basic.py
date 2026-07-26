from ..domain.memory import MemoryItem
from ..domain.retrieval import (
    RelevanceMode,
    Retrieved,
    relevance_chargram,
    relevance_cosine,
    relevance_jaccard,
)
from ..ports.embedding import EmbeddingPort
from ..ports.retriever import Retriever

RRF_K = 60


class ModeRetriever(Retriever):
    def __init__(self, mode: RelevanceMode = "chargram", embedder: EmbeddingPort | None = None):
        self.mode = mode
        self.embedder = embedder

    def rank(self, query: str, items: list[MemoryItem], k: int) -> list[Retrieved]:
        if not items or k <= 0:
            return []
        query_vec = None
        if self.mode == "cosine" and self.embedder is not None:
            query_vec = self.embedder.embed([query])[0]
            missing = [m for m in items if m.embedding is None]
            for item, vec in zip(missing, self.embedder.embed([m.text for m in missing])):
                item.embedding = vec
        scored = [
            Retrieved(item=m, recency=0.0, importance=0.0,
                      relevance=self._relevance(query, m, query_vec),
                      total=self._relevance(query, m, query_vec))
            for m in items
        ]
        scored.sort(key=lambda r: (-r.total, -r.item.tick, r.item.id))
        return scored[:k]

    def _relevance(self, query: str, item: MemoryItem, query_vec) -> float:
        if self.mode == "cosine":
            return relevance_cosine(query_vec, item.embedding)
        if self.mode == "jaccard":
            return relevance_jaccard(query, item.text)
        if self.mode == "none":
            return 0.0
        return relevance_chargram(query, item.text)


def reciprocal_rank_fusion(ranked_lists: list[list[MemoryItem]], k: int, rrf_k: int = RRF_K) -> list[MemoryItem]:
    scores: dict[str, float] = {}
    by_id: dict[str, MemoryItem] = {}
    for ranked in ranked_lists:
        for pos, item in enumerate(ranked):
            scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (rrf_k + pos + 1)
            by_id[item.id] = item
    order = sorted(by_id.values(), key=lambda m: (-scores[m.id], -m.tick, m.id))
    return order[:k]


class HybridRetriever(Retriever):
    def __init__(self, retrievers: list[Retriever], rrf_k: int = RRF_K, pool: int = 20):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.pool = pool

    def rank(self, query: str, items: list[MemoryItem], k: int) -> list[Retrieved]:
        if not items or k <= 0:
            return []
        lists = [[r.item for r in ret.rank(query, items, self.pool)] for ret in self.retrievers]
        fused = reciprocal_rank_fusion(lists, k, self.rrf_k)
        n = len(fused)
        return [
            Retrieved(item=m, recency=0.0, importance=0.0,
                      relevance=(n - i) / n, total=(n - i) / n)
            for i, m in enumerate(fused)
        ]


def default_hybrid(embedder: EmbeddingPort | None = None) -> HybridRetriever:
    lexical = ModeRetriever(mode="chargram")
    dense = ModeRetriever(mode="cosine", embedder=embedder) if embedder else ModeRetriever(mode="jaccard")
    return HybridRetriever([lexical, dense])
