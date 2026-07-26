"""Порт векторизації."""

from abc import ABC, abstractmethod


class EmbeddingPort(ABC):
    """Один виклик на пакет текстів — щоб адаптер міг батчити."""

    model: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...
