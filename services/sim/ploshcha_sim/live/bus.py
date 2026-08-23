"""Шина подій живого прогону: ring-буфер із монотонним seq і очікуванням без опитування.

Три інваріанти, без яких «онлайн» ламається тихо:
  • глядач, що приєднався пізніше, дістає ХВІСТ, а не все з початку — інакше пізній клієнт
    відтворює історію як «зараз»;
  • обрив зʼєднання не губить події: `since` доливає з буфера, тому реконект безшовний;
  • подія, породжена прогоном ОДНОГО гостя, не доїжджає в потік іншого.

Про третій окремо. Мітка сесії лежить ПОРУЧ із подією, а не всередині неї: контракт
`ploshcha-events.schema.json` валідується строго, і зайве поле в конверті фронт відкидав би цілком.
Тому буфер тримає два паралельні списки, а назовні йдуть ті самі конверти, що й раніше.

Мітка `None` у ПОДІЇ означає «спільне»: стан самого ядра (стеля витрат, смерть робітника, тік
світу) стосується всіх, хто дивиться, бо це не чиясь розмова. Мітка `None` у ГЛЯДАЧА означає
«інспектор»: він бачить усе, і саме на цьому тримаються старі тести й `soak`.

Буфер обмежений навмисно: живий прогін може йти годинами, і памʼять не має рости без межі.
Переповнення видиме через `dropped`, а не приховане.
"""

import threading

DEFAULT_CAPACITY = 2000


class EventBus:
    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        self.capacity = capacity
        self._events: list[dict] = []
        # Паралельний список міток: тримати їх окремо дешевше, ніж загортати кожен конверт у
        # пару, і головне — конверт лишається тим самим обʼєктом, що йде в SSE без копії.
        self._sids: list[str | None] = []
        self._first_seq = 0
        self.dropped = 0
        self._cond = threading.Condition()
        self._closed = False

    @property
    def next_seq(self) -> int:
        with self._cond:
            return self._first_seq + len(self._events)

    def publish(self, events: list[dict] | dict, sid: str | None = None) -> None:
        batch = [events] if isinstance(events, dict) else list(events)
        if not batch:
            return
        with self._cond:
            self._events.extend(batch)
            self._sids.extend([sid] * len(batch))
            overflow = len(self._events) - self.capacity
            if overflow > 0:
                del self._events[:overflow]
                del self._sids[:overflow]
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

    def _visible(self, cursor: int, sid: str | None,
                 shared_from: int | None = None) -> tuple[list[tuple[int, dict]], int]:
        """`shared_from` — позиція, з якої глядач слухає НАЖИВО.

        Усе, що раніше, він домальовує з історії, і там йому належать лише ВЛАСНІ події. Спільні
        (мітка `None`) — це стан ядра й прогони, запущені без сесії: доки вони йдуть, їх видно
        всім, але відтворювати їх новому глядачеві не можна. Доти фронт просив історію з нуля, і
        кожен, хто заходив із будь-якого пристрою, діставав чужі прогони як свої — на Дошці це
        виглядало купою сміттєвих тем, яких він ніколи не кидав.
        """
        start = max(cursor, self._first_seq)
        end = self._first_seq + len(self._events)
        out: list[tuple[int, dict]] = []
        for i in range(start - self._first_seq, len(self._events)):
            own = self._sids[i]
            seq = self._first_seq + i
            if sid is not None and shared_from is not None and seq < shared_from and own != sid:
                continue
            if sid is None or own is None or own == sid:
                out.append((seq, self._events[i]))
        return out, end

    def since_ids(self, cursor: int, sid: str | None = None,
                  shared_from: int | None = None) -> tuple[list[tuple[int, dict]], int]:
        """Те саме, що `since`, але з АБСОЛЮТНОЮ позицією кожної події.

        Позиція потрібна SSE як `id:`: після фільтрації подій менше, ніж просунувся курсор, тож
        рахувати `id` арифметикою від довжини пачки вже не можна — реконект зі стрибком курсора
        мовчки губив би чужі-й-свої події між ними.
        """
        with self._cond:
            return self._visible(cursor, sid, shared_from)

    def since(self, cursor: int, sid: str | None = None,
              shared_from: int | None = None) -> tuple[list[dict], int]:
        """Події з позиції `cursor` і новий курсор. Курсор — абсолютний індекс, не seq події."""
        with self._cond:
            pairs, end = self._visible(cursor, sid, shared_from)
        return [ev for _, ev in pairs], end

    def wait(self, cursor: int, timeout: float = 15.0,
             sid: str | None = None) -> tuple[list[dict], int]:
        """Блокує до появи нових подій або таймауту. Таймаут — це нормальний вихід (heartbeat)."""
        pairs, end = self.wait_ids(cursor, timeout, sid)
        return [ev for _, ev in pairs], end

    def wait_ids(self, cursor: int, timeout: float = 15.0, sid: str | None = None,
                 shared_from: int | None = None) -> tuple[list[tuple[int, dict]], int]:
        with self._cond:
            if cursor >= self._first_seq + len(self._events) and not self._closed:
                self._cond.wait(timeout)
            return self._visible(cursor, sid, shared_from)

    def tail_cursor(self) -> int:
        """Курсор «з цього місця», щоб новий глядач не отримав усю історію."""
        with self._cond:
            return self._first_seq + len(self._events)
