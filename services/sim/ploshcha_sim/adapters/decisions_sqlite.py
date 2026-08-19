"""Ухвали села — стан, який переживає прогін.

Без цього «село вирішило поставити сторожа» лишається фразою в хроніці: наступне віче про це не
знає, і на сцені нічого не змінилось. Тобто рішення без наслідку — а саме брак наслідку й робив
ПЛОЩУ тонкою, скільки б подій ми не малювали.

Не RAG і не векторний пошук: таблиця й прості вибірки. Що ухвалили — визначено даними, отже
належить коду.
"""

import sqlite3
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

    def _db(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

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
