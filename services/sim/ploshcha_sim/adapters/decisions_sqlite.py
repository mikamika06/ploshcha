"""Ухвали села — стан, який переживає прогін.

Без цього «село вирішило поставити сторожа» лишається фразою в хроніці: наступне віче про це не
знає, і на сцені нічого не змінилось. Тобто рішення без наслідку — а саме брак наслідку й робив
ПЛОЩУ тонкою, скільки б подій ми не малювали.

Не RAG і не векторний пошук: таблиця й прості вибірки. Що ухвалили — визначено даними, отже
належить коду.
"""

import sqlite3
from contextlib import contextmanager
import time
from pathlib import Path

KEEP = 40


class SqliteDecisions:
    def __init__(self, path: str | Path, *, clock=time.time):
        self.path = str(path)
        self._clock = clock
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS decisions ("
                       "id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, "
                       "label TEXT NOT NULL, who TEXT NOT NULL, poi TEXT NOT NULL, "
                       "created_at REAL NOT NULL)")

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

    def add(self, topic: str, label: str, who: str, poi: str) -> None:
        with self._db() as db:
            db.execute("INSERT INTO decisions(topic, label, who, poi, created_at) "
                       "VALUES(?,?,?,?,?)", (topic[:300], label[:200], who, poi, self._clock()))

    def standing(self, limit: int = KEEP) -> list[dict]:
        """Чинні ухвали, найсвіжіші перші. Одна людина стоїть в одному місці — тримаємо ОСТАННЄ
        доручення, інакше сцена намагалась би поставити її у двох місцях одразу."""
        with self._db() as db:
            rows = db.execute("SELECT topic, label, who, poi, created_at FROM decisions "
                              "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        seen: set[str] = set()
        out: list[dict] = []
        for r in rows:
            if r["who"] in seen:
                continue
            seen.add(r["who"])
            out.append(dict(r))
        return out

    def recent(self, limit: int = 3) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT topic, label, who, poi FROM decisions "
                              "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
