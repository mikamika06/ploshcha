"""Памʼять агента + чистий скор retrieval."""

from typing import Literal

from pydantic import BaseModel, Field

MemoryKind = Literal["observation", "reflection", "plan"]


class MemoryItem(BaseModel):
    id: str
    tick: int = Field(ge=0)
    kind: MemoryKind
    text: str = Field(min_length=1)
    importance: int = Field(ge=1, le=10)
    provenance: str = Field(default="sim", description="sim | agent:<id> | board | external")
    embedding: list[float] | None = None
    evidence: list[str] = Field(default_factory=list, description="id памʼятей-підстав")


def decay_from_half_life(half_life_ticks: float) -> float:
    """Скільки лишається за тік, щоб за half_life_ticks лишилась половина."""
    return 0.5 ** (1.0 / half_life_ticks)


# 6 тіків = доба (PHASES у state.py), тож 18 тіків = три дні.
HALF_LIFE_TICKS = 18.0
DECAY = decay_from_half_life(HALF_LIFE_TICKS)
W_RECENCY = 1.0
W_IMPORTANCE = 1.0
W_RELEVANCE = 1.0


def recency(now_tick: int, item_tick: int, decay: float = DECAY) -> float:
    return decay ** max(0, now_tick - item_tick)


def normalized_importance(item: MemoryItem) -> float:
    return item.importance / 10.0


def score(
    item: MemoryItem,
    now_tick: int,
    relevance: float,
    *,
    w_recency: float = W_RECENCY,
    w_importance: float = W_IMPORTANCE,
    w_relevance: float = W_RELEVANCE,
    decay: float = DECAY,
) -> float:
    return (
        w_recency * recency(now_tick, item.tick, decay)
        + w_importance * normalized_importance(item)
        + w_relevance * relevance
    )


def top_k(
    items: list[MemoryItem],
    now_tick: int,
    relevances: dict[str, float],
    k: int,
    **weights: float,
) -> list[MemoryItem]:
    ranked = sorted(
        items,
        key=lambda m: (-score(m, now_tick, relevances.get(m.id, 0.0), **weights), -m.tick, m.id),
    )
    return ranked[: max(0, k)]
