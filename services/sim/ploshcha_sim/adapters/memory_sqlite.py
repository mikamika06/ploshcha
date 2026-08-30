"""Памʼять села: минулі віча, стосунки й літопис.

Без цього кожне віче починається з чистого аркуша, і немає причини приходити вдруге. Ухвали й
чутки вже живуть між прогонами — бракувало самої розмови й того, хто з ким через неї посварився.

Стосунки виводяться КОДОМ із партитури, а не питаються в моделі: хто кому піддакнув — зблизились,
хто заперечив — розійшлись. Це визначено даними, і питати про це окремо означало б платити за
відповідь, яку ми вже маємо.
"""

import sqlite3
from contextlib import contextmanager
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
        """Хто КОМУ, а не «хто з ким»: напрямок лишається в базі.

        ★ Доти пара сортувалась перед записом (`lo, hi = sorted((a, b))`), і саме на цьому рядку
        структура втрачалась: `bonds_from` віддає рівно трійку (мовець, адресат, дельта), тобто
        єдине місце системи, де `у_відповідь` стає фактом про людей, — а в базу лягала симетрична
        сума за всі віча. Замір на двох живих базах: 27 пар у `docs/research/eval-runs/ploshcha.db`
        (22 відʼємні, найгірша `parubok|shynkar = −6.0`, тобто вперлась у стелю) і 18 у
        `data/ploshcha/ploshcha.db`. З них видно, що парубок із шинкаркою історично сваряться, і
        не видно, хто кого вчора підтримав, — а суперечку відновлюють саме з другого.

        Схема не міняється: ключ (a, b) і доти був парою, просто впорядкованою. Рядки старих баз
        читаються як напрямок «a → b» — це вигадує напрямок там, де його не записували, зате не
        вимагає ні міграції, ні втрати накопиченого.
        """
        if a == b:
            return
        with self._db() as db:
            row = db.execute("SELECT score FROM bonds WHERE a=? AND b=?", (a, b)).fetchone()
            score = (row["score"] if row else 0.0) + delta
            score = max(-BOND_CAP, min(BOND_CAP, score))
            db.execute("INSERT OR REPLACE INTO bonds(a, b, score) VALUES(?,?,?)", (a, b, score))

    def toward(self, a: str, b: str) -> float:
        """Як `a` ставиться до `b` — саме в цей бік, і зустрічний бік може бути іншим."""
        with self._db() as db:
            row = db.execute("SELECT score FROM bonds WHERE a=? AND b=?", (a, b)).fetchone()
        return row["score"] if row else 0.0

    def directed(self) -> dict[tuple[str, str], float]:
        """Усі ребра як вони записані: ключ — (хто, кому)."""
        with self._db() as db:
            rows = db.execute("SELECT a, b, score FROM bonds").fetchall()
        return {(r["a"], r["b"]): r["score"] for r in rows}

    def bonds(self) -> dict[tuple[str, str], float]:
        """Симетричний ЗРІЗ для того, кому напрямок не потрібен, — ваги перебивки в `scatter`.

        Сварка тягне влізти поперек однаково з обох боків, тож споживач лишається з тим самим
        ключем (впорядкована пара) і тим самим числом, що й доти; напрямок живе нижче, у базі.
        """
        out: dict[tuple[str, str], float] = {}
        for (a, b), score in self.directed().items():
            key = (a, b) if a <= b else (b, a)
            out[key] = max(-BOND_CAP, min(BOND_CAP, out.get(key, 0.0) + score))
        return out

    def between(self, a: str, b: str) -> float:
        lo, hi = sorted((a, b))
        return self.bonds().get((lo, hi), 0.0)
