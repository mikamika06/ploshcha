"""Село зберігається: повернувся — ті самі сусіди.

Породження коштує один виклик Mamay, але річ не в грошах. Якби село народжувалось щоразу заново,
ухвалам, чуткам і стосункам не було б до кого кріпитись: сьогодні сторожа поставили Іванові, а
завтра Івана вже нема. Тому склад — стан, а не побічний ефект запуску.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from ..domain.people import Person, refit


class SqliteVillage:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS village ("
                       "seed INTEGER PRIMARY KEY, people TEXT NOT NULL)")

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

    def load(self, seed: int) -> list[Person]:
        with self._db() as db:
            row = db.execute("SELECT people FROM village WHERE seed=?", (seed,)).fetchone()
        if not row:
            return []
        try:
            # ★ Звіряємо з малюнком НА ЧИТАННІ, а не лише на кузні.
            #
            # Село кується раз і далі береться звідси, тож імʼя, яке не пройшло б `fit_gender`
            # сьогодні, лежить у базі вічно: у живій базі власника (2026-08-29) `sheptu` звалась
            # «Яким Бувалінда», а `shynkar` — «Грицько Поговір» — два чоловічі імені на жіночих
            # фігурах із восьми. Підпис на сцені береться саме звідси (`public_cast`), тому
            # розходження видно очима.
            return refit([Person(**x) for x in json.loads(row[0])])
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
