"""Шина подій живого прогону: ring-буфер із монотонним seq і очікуванням без опитування.

Два інваріанти, без яких «онлайн» ламається тихо:
  • глядач, що приєднався пізніше, дістає ХВІСТ, а не все з початку — інакше пізній клієнт
    відтворює історію як «зараз»;
  • обрив зʼєднання не губить події: `since` доливає з буфера, тому реконект безшовний.

Буфер обмежений навмисно: живий прогін може йти годинами, і памʼять не має рости без межі.
Переповнення видиме через `dropped`, а не приховане.
"""

import threading

DEFAULT_CAPACITY = 2000


class EventBus:
    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        self.capacity = capacity
        self._events: list[dict] = []
        self._first_seq = 0
        self.dropped = 0
        self._cond = threading.Condition()
        self._closed = False

    @property
    def next_seq(self) -> int:
        with self._cond:
            return self._first_seq + len(self._events)

    def publish(self, events: list[dict] | dict) -> None:
        batch = [events] if isinstance(events, dict) else list(events)
        if not batch:
            return
        with self._cond:
            self._events.extend(batch)
            overflow = len(self._events) - self.capacity
            if overflow > 0:
                del self._events[:overflow]
                self._first_seq += overflow
                self.dropped += overflow
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def since(self, cursor: int) -> tuple[list[dict], int]:
        """Події з позиції `cursor` і новий курсор. Курсор — абсолютний індекс, не seq події."""
        with self._cond:
            start = max(cursor, self._first_seq)
            out = self._events[start - self._first_seq:]
            return list(out), start + len(out)

    def wait(self, cursor: int, timeout: float = 15.0) -> tuple[list[dict], int]:
        """Блокує до появи нових подій або таймауту. Таймаут — це нормальний вихід (heartbeat)."""
        with self._cond:
            if cursor >= self._first_seq + len(self._events) and not self._closed:
                self._cond.wait(timeout)
        return self.since(cursor)

    def tail_cursor(self) -> int:
        """Курсор «з цього місця», щоб новий глядач не отримав усю історію."""
        with self._cond:
            return self._first_seq + len(self._events)
