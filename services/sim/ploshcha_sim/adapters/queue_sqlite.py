"""Черга на SQLite: єдиний файл, який переживає краш процесу.

Чому SQLite, а не список у памʼяті: перевірити «черга переживає краш» на структурі в памʼяті
неможливо — вона гине разом із процесом. Тут стан лежить у файлі, тому тест може закрити обʼєкт,
відкрити новий і подивитись, що робота не зникла.

Три рішення, які тримають інваріанти:
  • `key` — PRIMARY KEY, тому повторна постановка того самого документа нічого не робить (і повертає
    `False`, а не падає): інгест можна перезапускати з нуля скільки завгодно;
  • видача — **аренда з міткою часу**, тому покинуте повертається (`recover_stale`), а не зникає;
  • після `max_attempts` айтем іде в `dead`, а не крутиться вічно — інакше один зламаний документ
    з'їдає весь бюджет прогону.
"""

import json
import sqlite3
from contextlib import contextmanager
import time
from pathlib import Path

from ..ports.queue import QueuePort, WorkItem

SCHEMA = """
CREATE TABLE IF NOT EXISTS work (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    leased_at REAL,
    worker TEXT,
    result TEXT,
    error TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS work_status ON work(status);
"""

MAX_ATTEMPTS = 3


class SqliteQueue(QueuePort):
    def __init__(self, path: str | Path, *, max_attempts: int = MAX_ATTEMPTS,
                 clock=time.time):
        self.path = str(path)
        self.max_attempts = max_attempts
        self._clock = clock
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        """★ Зʼєднання ЗАКРИВАЄТЬСЯ, а не лишається на совісті збирача сміття.

        `with sqlite3.connect(...) as db` не закриває зʼєднання — він лише фіксує транзакцію. Доки
        обʼєкт живий (а в потоках і в трасах винятків посилання тримаються довго), його файловий
        дескриптор теж живий. На проді це поклало ядро: 508 відкритих ручок до `ploshcha.db` і 456
        до його WAL при стелі 1024, далі `[Errno 24] Too many open files` — і як наслідок
        «OperationalError: unable to open database file», через яку цикл став на паузу.
        """
        db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        try:
            # WAL, бо читання статистики під час роботи не має блокувати воркера.
            db.execute("PRAGMA journal_mode=WAL")
            yield db
            # `with sqlite3.Connection` фіксував транзакцію сам; закриваючи зʼєднання руками, ми
            # мусимо зробити це замість нього — інакше запис тихо відкочується на закритті.
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def put(self, key: str, payload: dict) -> bool:
        with self._db() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO work(key, payload, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(payload, ensure_ascii=False), self._clock()))
            return cur.rowcount == 1

    def lease(self, worker: str, exclude_sids: tuple[str, ...] = ()) -> WorkItem | None:
        """`exclude_sids` — чиї теми зараз НЕ брати.

        Гість, що кинув три теми поспіль, отримував три віча одночасно: фронт тримає одну
        стенограму й одну локацію, тож три прогони змішувались в одному браузері («село не
        говорить, бігають» — скарга з телефона). Паралельність між РІЗНИМИ людьми лишається, бо
        заради неї все й робилось.
        """
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            if exclude_sids:
                holes = ",".join("?" * len(exclude_sids))
                row = db.execute(
                    "SELECT key, payload, attempts FROM work WHERE status='pending' "
                    f"AND COALESCE(json_extract(payload, '$.sid'), '') NOT IN ({holes}) "
                    "ORDER BY created_at, key LIMIT 1", exclude_sids).fetchone()
            else:
                row = db.execute(
                    "SELECT key, payload, attempts FROM work WHERE status='pending' "
                    "ORDER BY created_at, key LIMIT 1").fetchone()
            if row is None:
                db.execute("COMMIT")
                return None
            key, payload, attempts = row
            db.execute("UPDATE work SET status='leased', attempts=attempts+1, leased_at=?, worker=? "
                       "WHERE key=?", (self._clock(), worker, key))
            db.execute("COMMIT")
            return WorkItem(key=key, payload=json.loads(payload), attempts=attempts + 1,
                            status="leased")

    def ack(self, key: str, result: dict) -> None:
        with self._db() as db:
            db.execute("UPDATE work SET status='done', result=?, leased_at=NULL WHERE key=?",
                       (json.dumps(result, ensure_ascii=False), key))

    def fail(self, key: str, error: str) -> None:
        """Провал не остаточний, поки не вичерпані спроби — але й не вічний."""
        with self._db() as db:
            row = db.execute("SELECT attempts FROM work WHERE key=?", (key,)).fetchone()
            attempts = row[0] if row else 0
            status = "dead" if attempts >= self.max_attempts else "pending"
            db.execute("UPDATE work SET status=?, error=?, leased_at=NULL WHERE key=?",
                       (status, error[:500], key))

    def cancel_pending(self, *, key: str | None = None, sid: str | None = None) -> int:
        """★ Зняти з черги те, що ще НЕ орендоване. Повертає, скільки знято.

        Без цього «завершити» доходило рівно до того прогону, який уже веде робітник, а тема, що
        лежала В ЧЕРЗІ, спокійно дочікувалась свого. Вікно тут не теоретичне: наглядач дивиться на
        чергу раз на `SUPERVISE_EVERY_S` = 2 с і добирає робітника САМЕ за наявністю `pending`,
        тобто гість натискав «завершити» — і за дві секунди село починало голосно гомоніти про
        тему, від якої він щойно відмовився, ще й за повну ціну прогону (медіана прод-прогону
        19 093 токени на 121 записаному айтемі).

        Ціль називається `key` або `sid`, і хоч одне з двох мусить бути: `DELETE` без умови стер би
        чергу цілком, а це рівно та поламка, від якої тут стережуться. Порожній виклик тому нічого
        не робить і каже про це нулем, а не винятком, — той самий вибір, що в `requeue_dead`.

        Орендованого не чіпаємо навмисно (`status='pending'`): робітник уже в дорозі, у нього своя
        мʼяка зупинка (`Viche.hush`), і забрати айтем із-під нього означало б лишити прогін без
        термінального стану — рівно те, чого уникає `recover_stale`.

        ★ ЧОМУ `DELETE`, А НЕ СТАТУС «скасовано». Ключ теми з Дошки виводиться з її ТЕКСТУ, а `put`
        — `INSERT OR IGNORE` за ключем. Рядок, що лишився б лежати, назавжди закрив би цю саму тему
        для цього самого гостя: він написав би її вдруге, черга мовчки відповіла б `False`, і віче
        не почалось би ніколи. Скасована тема не має й що памʼятати — вона не відпрацювала, тож
        ані результату, ані причини провалу в неї немає.
        """
        if key is None and sid is None:
            return 0
        where = ["status='pending'"]
        args: list = []
        if key is not None:
            where.append("key=?")
            args.append(key)
        if sid is not None:
            where.append("COALESCE(json_extract(payload, '$.sid'), '')=?")
            args.append(sid)
        with self._db() as db:
            cur = db.execute(f"DELETE FROM work WHERE {' AND '.join(where)}", args)
            return cur.rowcount

    def requeue_dead(self, key: str | None = None) -> int:
        """Повернути мертві айтеми в чергу, скинувши лічильник спроб.

        Без цього єдиний спосіб оживити задачу, яка вмерла через уже виправлений дефект коду, — лізти
        в SQLite руками. Саме так у базі й лежав `dead: 1` від давно полагодженого `TypeError`.
        """
        with self._db() as db:
            if key is None:
                cur = db.execute("UPDATE work SET status='pending', attempts=0, error=NULL, "
                                 "leased_at=NULL WHERE status='dead'")
            else:
                cur = db.execute("UPDATE work SET status='pending', attempts=0, error=NULL, "
                                 "leased_at=NULL WHERE status='dead' AND key=?", (key,))
            return cur.rowcount

    def recover_stale(self, older_than_s: float) -> int:
        """Покинута аренда повертається в чергу. Це і є переживання краша."""
        cutoff = self._clock() - older_than_s
        with self._db() as db:
            cur = db.execute(
                "UPDATE work SET status=CASE WHEN attempts>=? THEN 'dead' ELSE 'pending' END, "
                "leased_at=NULL WHERE status='leased' AND leased_at IS NOT NULL AND leased_at < ?",
                (self.max_attempts, cutoff))
            return cur.rowcount

    def stats(self) -> dict[str, int]:
        with self._db() as db:
            rows = db.execute("SELECT status, COUNT(*) FROM work GROUP BY status").fetchall()
        return {status: count for status, count in rows}

    def results(self) -> list[WorkItem]:
        """Готове — у стабільному порядку, щоб агрегація була детермінованою."""
        with self._db() as db:
            rows = db.execute(
                "SELECT key, payload, attempts, status, result, error FROM work "
                "WHERE status IN ('done','dead') ORDER BY key").fetchall()
        return [WorkItem(key=k, payload=json.loads(p), attempts=a, status=s,
                         result=json.loads(r) if r else None, error=e)
                for k, p, a, s, r, e in rows]
