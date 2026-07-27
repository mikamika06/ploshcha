from ..domain.memory import MemoryItem
from ..domain.retrieval import Retrieved, tokens
from ..ports.retriever import Retriever

_MORPH = None


def _morph():
    global _MORPH
    if _MORPH is None:
        import pymorphy3
        _MORPH = pymorphy3.MorphAnalyzer(lang="uk")
    return _MORPH


def lemmas(text: str) -> list[str]:
    m = _morph()
    return [m.parse(t)[0].normal_form for t in tokens(text)]


class BM25Retriever(Retriever):
    def rank(self, query: str, items: list[MemoryItem], k: int) -> list[Retrieved]:
        if not items or k <= 0:
            return []
        from rank_bm25 import BM25Okapi
        corpus = [lemmas(m.text) for m in items]
        bm = BM25Okapi(corpus)
        scores = bm.get_scores(lemmas(query))
        ranked = sorted(zip(items, scores), key=lambda x: (-float(x[1]), -x[0].tick, x[0].id))
        return [
            Retrieved(item=m, recency=0.0, importance=0.0, relevance=float(s), total=float(s))
            for m, s in ranked[:k]
        ]
