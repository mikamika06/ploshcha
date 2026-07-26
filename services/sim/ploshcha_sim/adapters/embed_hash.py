"""Детермінований хеш-векторизатор (НЕ семантичний)."""

import hashlib

from ..domain.retrieval import char_grams
from ..ports.embedding import EmbeddingPort


class HashEmbedder(EmbeddingPort):
    """НЕ семантичний: хешує char-грами у фіксовані відра.

    Потрібен для двох речей: детермінованих тестів і нижньої межі в абляції
    («наскільки семантична модель краща за поверхневий збіг»).
    """

    def __init__(self, dim: int = 64, model: str = "hash-3gram"):
        self.dim = dim
        self.model = model
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for gram in char_grams(text):
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            vec[int.from_bytes(h, "big") % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec
