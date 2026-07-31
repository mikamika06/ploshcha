"""Порт черги: одиниця роботи, яка ПЕРЕЖИВАЄ краш.

Ключове не «черга» як структура, а два інваріанти, без яких масштаб — фікція:
  • **ідемпотентність постановки**: той самий документ, поставлений двічі, не подвоює роботу;
  • **аренда, а не видача**: якщо процес помер, робота повертається в чергу, а не зникає з нею.

Друге саме тому, що падіння на тисячі документів — норма, а не виняток.
"""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

TaskStatus = Literal["pending", "leased", "done", "failed", "dead"]


class WorkItem(BaseModel):
    key: str = Field(description="ідентифікатор одиниці роботи; повтор із тим самим key = той самий айтем")
    payload: dict = Field(default_factory=dict)
    attempts: int = 0
    status: TaskStatus = "pending"
    result: dict | None = None
    error: str | None = None


class QueuePort(ABC):
    @abstractmethod
    def put(self, key: str, payload: dict) -> bool:
        """`False` означає «вже було» — це нормальний випадок, не помилка."""

    @abstractmethod
    def lease(self, worker: str) -> WorkItem | None: ...

    @abstractmethod
    def ack(self, key: str, result: dict) -> None: ...

    @abstractmethod
    def fail(self, key: str, error: str) -> None: ...

    @abstractmethod
    def recover_stale(self, older_than_s: float) -> int:
        """Повернути в чергу арендоване, але покинуте. Це і є переживання краша."""

    @abstractmethod
    def stats(self) -> dict[str, int]: ...
