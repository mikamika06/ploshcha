"""Retrieval памʼяті. Чисто, без I/O."""

import math
import re
from typing import Literal

from pydantic import BaseModel

from .memory import DECAY, MemoryItem, W_IMPORTANCE, W_RECENCY, W_RELEVANCE, normalized_importance, recency

RelevanceMode = Literal["jaccard", "chargram", "cosine", "none"]

GRAM_N = 3
_APOSTROPHES = "'’ʼʻ`"
_WORD_RE = re.compile(r"[0-9a-zЀ-ӿ" + _APOSTROPHES + "]+")


def normalize_text(text: str) -> str:
    lowered = text.lower()
    for ch in _APOSTROPHES:
        lowered = lowered.replace(ch, "'")
    return lowered


def tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(normalize_text(text)))


def char_grams(text: str, n: int = GRAM_N) -> set[str]:
    """N-грами по словах із межовими пробілами — стійкі до відмінків."""
    out: set[str] = set()
    for tok in _WORD_RE.findall(normalize_text(text)):
        padded = f" {tok} "
        if len(padded) <= n:
            out.add(padded)
            continue
        out.update(padded[i : i + n] for i in range(len(padded) - n + 1))
    return out


def _dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def relevance_jaccard(query: str, text: str) -> float:
    a, b = tokens(query), tokens(text)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def relevance_chargram(query: str, text: str, n: int = GRAM_N) -> float:
    return _dice(char_grams(query, n), char_grams(text, n))


def relevance_cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


class Retrieved(BaseModel):
    """Компоненти лишаються нарізно — інакше не видно, ЧОМУ пригадалось."""

    item: MemoryItem
    recency: float
    importance: float
    relevance: float
    total: float


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo == 0.0:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def raw_relevance(
    item: MemoryItem,
    query: str,
    mode: RelevanceMode,
    query_embedding: list[float] | None = None,
) -> float:
    if mode == "none":
        return 0.0
    if mode == "cosine":
        return relevance_cosine(query_embedding, item.embedding)
    if mode == "jaccard":
        return relevance_jaccard(query, item.text)
    return relevance_chargram(query, item.text)


def retrieve(
    items: list[MemoryItem],
    query: str,
    now_tick: int,
    k: int,
    *,
    mode: RelevanceMode = "chargram",
    query_embedding: list[float] | None = None,
    normalize: bool = True,
    w_recency: float = W_RECENCY,
    w_importance: float = W_IMPORTANCE,
    w_relevance: float = W_RELEVANCE,
    decay: float = DECAY,
) -> list[Retrieved]:
    if not items or k <= 0:
        return []

    rec = [recency(now_tick, m.tick, decay) for m in items]
    imp = [normalized_importance(m) for m in items]
    rel = [raw_relevance(m, query, mode, query_embedding) for m in items]
    if normalize:
        rec, imp, rel = _minmax(rec), _minmax(imp), _minmax(rel)

    scored = [
        Retrieved(
            item=m,
            recency=rec[i],
            importance=imp[i],
            relevance=rel[i],
            total=w_recency * rec[i] + w_importance * imp[i] + w_relevance * rel[i],
        )
        for i, m in enumerate(items)
    ]
    scored.sort(key=lambda r: (-r.total, -r.item.tick, r.item.id))
    return scored[:k]
