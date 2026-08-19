"""Село зберігається: повернувся — ті самі сусіди.

Породження коштує один виклик Mamay, але річ не в грошах. Якби село народжувалось щоразу заново,
ухвалам, чуткам і стосункам не було б до кого кріпитись: сьогодні сторожа поставили Іванові, а
завтра Івана вже нема. Тому склад — стан, а не побічний ефект запуску.
"""

import json
import sqlite3
from pathlib import Path

from ..domain.people import Person


class SqliteVillage:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS village ("
                       "seed INTEGER PRIMARY KEY, people TEXT NOT NULL)")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def load(self, seed: int) -> list[Person]:
        with self._db() as db:
            row = db.execute("SELECT people FROM village WHERE seed=?", (seed,)).fetchone()
        if not row:
            return []
        try:
            return [Person(**x) for x in json.loads(row[0])]
        except (ValueError, TypeError):
            # Побите збереження — не привід падати: перепороджуємо, і це видно як новий склад.
            return []

    def save(self, seed: int, people: list[Person]) -> None:
        blob = json.dumps([p.model_dump() for p in people], ensure_ascii=False)
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO village(seed, people) VALUES(?,?)", (seed, blob))

    def forget(self, seed: int) -> None:
        with self._db() as db:
            db.execute("DELETE FROM village WHERE seed=?", (seed,))
