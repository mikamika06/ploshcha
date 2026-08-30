"""Чутки села й репутація тих, хто їх пускає.

Чутка — твердження без підстави, сказане вголос. Вона мандрує селом, і її можна підтвердити або
спростувати. Це не декорація: у ядрі ми вже розрізняємо твердження з доказом і без, тризначним
`found` та вироком верифікатора, — чутка лише дає цьому строк життя й наслідок.

★ Наслідок і є суть: кому чутку спростували, того наступного разу слухають менше — БУКВАЛЬНО менше
тактів у партитурі. Рахує код, бо це визначено даними, а не судженням.
"""

import sqlite3
from contextlib import contextmanager
import time
from pathlib import Path

# Скільки тактів втрачає той, кому спростували. Не нуль: людину не викидають із села за помилку.
PENALTY = 0.22
FLOOR = 0.4


class SqliteRumours:
    def __init__(self, path: str | Path, *, clock=time.time):
        self.path = str(path)
        self._clock = clock
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS rumours ("
                       "id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, "
                       "who TEXT NOT NULL, claim TEXT NOT NULL, "
                       "status TEXT NOT NULL DEFAULT 'нова', created_at REAL NOT NULL)")

    @contextmanager
    def _db(self):
        """★ Зʼєднання ЗАКРИВАЄТЬСЯ явно.

        `with sqlite3.connect(...)` фіксує транзакцію, але не закриває зʼєднання: доки обʼєкт
        живий, живий і його дескриптор. У черзі цей самий взірець поклав ядро на проді — 964
        відкритих дескриптори при стелі 1024, `[Errno 24] Too many open files`, а слідом
        «OperationalError: unable to open database file» і цикл на паузі.
        """
        con = sqlite3.connect(self.path, timeout=10)
        try:
            con.row_factory = sqlite3.Row
            yield con
            # `with sqlite3.Connection` фіксував транзакцію сам; закриваємо руками — фіксуємо теж
            # руками, інакше запис відкочується на закритті (заміряно: хроніка переставала
            # накопичуватись, а звʼязки не росли).
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def add(self, topic: str, who: str, claim: str) -> None:
        with self._db() as db:
            db.execute("INSERT INTO rumours(topic, who, claim, created_at) VALUES(?,?,?,?)",
                       (topic[:300], who, claim[:200], self._clock()))

    def settle(self, claim_id: int, status: str) -> None:
        if status not in ("підтверджена", "спростована"):
            return
        with self._db() as db:
            db.execute("UPDATE rumours SET status=? WHERE id=?", (status, claim_id))

    def open(self, limit: int = 5) -> list[dict]:
        """Чутки, які ще ходять селом — вони й лягають у пакет наступного віча."""
        with self._db() as db:
            rows = db.execute("SELECT id, topic, who, claim FROM rumours WHERE status='нова' "
                              "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def standing(self, who: str) -> float:
        """Скільки важить слово людини: 1.0 — як усі, менше — спростовували.

        Підтверджена чутка НЕ підіймає вище одиниці. Правота повертає довіру, а не купує зайвий
        час слова: інакше один щасливий здогад робив би людину головною назавжди.
        """
        with self._db() as db:
            rows = db.execute("SELECT status, COUNT(*) AS n FROM rumours WHERE who=? "
                              "GROUP BY status", (who,)).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        wrong = counts.get("спростована", 0)
        right = counts.get("підтверджена", 0)
        return round(max(FLOOR, 1.0 - PENALTY * max(0, wrong - right)), 3)
