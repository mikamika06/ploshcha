"""Памʼять села: минулі віча, стосунки й літопис.

Без цього кожне віче починається з чистого аркуша, і немає причини приходити вдруге. Ухвали й
чутки вже живуть між прогонами — бракувало самої розмови й того, хто з ким через неї посварився.

Стосунки виводяться КОДОМ із партитури, а не питаються в моделі: хто кому піддакнув — зблизились,
хто заперечив — розійшлись. Це визначено даними, і питати про це окремо означало б платити за
відповідь, яку ми вже маємо.
"""

import sqlite3
import time
from pathlib import Path

RECALL = 2
BOND_CAP = 6


class SqliteMemory:
    def __init__(self, path: str | Path, *, clock=time.time):
        self.path = str(path)
        self._clock = clock
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS chronicles ("
                       "id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, "
                       "title TEXT NOT NULL, narration TEXT NOT NULL, mood TEXT NOT NULL, "
                       "created_at REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS bonds ("
                       "a TEXT NOT NULL, b TEXT NOT NULL, score REAL NOT NULL DEFAULT 0, "
                       "PRIMARY KEY (a, b))")

    def _db(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    # ── літопис ──────────────────────────────────────────────────────────────
    def remember(self, topic: str, title: str, narration: str, mood: str) -> None:
        with self._db() as db:
            db.execute("INSERT INTO chronicles(topic, title, narration, mood, created_at) "
                       "VALUES(?,?,?,?,?)",
                       (topic[:300], title[:160], narration[:600], mood[:40], self._clock()))

    def chronicle(self, limit: int = 20) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT topic, title, narration, mood FROM chronicles "
                              "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def recall(self, topic: str, limit: int = RECALL) -> list[dict]:
        """Спорідненi віча — за спільними словами. Не вектори: слова села короткі й конкретні,
        і зайвий шар тут дав би схожість там, де її немає."""
        words = {w.lower().strip(".,!?«»'\"") for w in topic.split() if len(w) > 4}
        best: list[tuple[int, dict]] = []
        for row in self.chronicle(limit=40):
            other = {w.lower().strip(".,!?«»'\"") for w in row["topic"].split() if len(w) > 4}
            shared = len(words & other)
            if shared:
                best.append((shared, row))
        best.sort(key=lambda x: -x[0])
        return [r for _, r in best[:limit]]

    # ── стосунки ─────────────────────────────────────────────────────────────
    def bond(self, a: str, b: str, delta: float) -> None:
        if a == b:
            return
        lo, hi = sorted((a, b))
        with self._db() as db:
            row = db.execute("SELECT score FROM bonds WHERE a=? AND b=?", (lo, hi)).fetchone()
            score = (row["score"] if row else 0.0) + delta
            score = max(-BOND_CAP, min(BOND_CAP, score))
            db.execute("INSERT OR REPLACE INTO bonds(a, b, score) VALUES(?,?,?)", (lo, hi, score))

    def bonds(self) -> dict[tuple[str, str], float]:
        with self._db() as db:
            rows = db.execute("SELECT a, b, score FROM bonds").fetchall()
        return {(r["a"], r["b"]): r["score"] for r in rows}

    def between(self, a: str, b: str) -> float:
        lo, hi = sorted((a, b))
        return self.bonds().get((lo, hi), 0.0)
