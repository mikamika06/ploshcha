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

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        # WAL, бо читання статистики під час роботи не має блокувати воркера.
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def put(self, key: str, payload: dict) -> bool:
        with self._db() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO work(key, payload, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(payload, ensure_ascii=False), self._clock()))
            return cur.rowcount == 1

    def lease(self, worker: str) -> WorkItem | None:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
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
