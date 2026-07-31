"""Пакетний прогін по черзі: узяти айтем, зробити, підтвердити — і вміти продовжити після краха.

Тут навмисно **немає** власної логіки повторів і лімітів: повтори живуть у черзі
(`attempts`/`dead`), ліміти — в `Governor`. Інакше кожен скрипт мав би свою версію обох, і
«зупинився на ліміті» означало б різне в різних місцях.

Порядок операцій критичний: **`ack` після роботи, а не до неї**. Якщо процес умирає між ними,
`recover_stale` повертає айтем у чергу — робота повториться, але не зникне. Протилежний порядок
губив би роботу тихо.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ploshcha_sim.domain.governor import Governor
from ploshcha_sim.ports.queue import QueuePort, WorkItem

STALE_S = 300.0


class BatchReport(BaseModel):
    done: int = 0
    failed: int = 0
    stopped: str | None = None
    tokens: int = 0
    usd: float = 0.0
    recovered: int = 0
    stats: dict[str, int] = Field(default_factory=dict)


def run_queue(queue: QueuePort, worker: str, handler: Callable[[WorkItem], dict],
              *, governor: Governor | None = None, stale_s: float = STALE_S,
              estimate_tokens: int = 0) -> BatchReport:
    """`handler` віддає результат як dict; його поля `tokens`/`usd` живлять governor."""
    governor = governor or Governor()
    report = BatchReport(recovered=queue.recover_stale(stale_s))

    while True:
        if (reason := governor.stop_reason(next_tokens=estimate_tokens)) is not None:
            report.stopped = reason
            break
        item = queue.lease(worker)
        if item is None:
            break
        try:
            result = handler(item)
        except Exception as exc:  # робота одного айтема не має вбивати прогін
            queue.fail(item.key, f"{type(exc).__name__}: {exc}")
            report.failed += 1
            continue
        queue.ack(item.key, result)
        report.done += 1
        tokens = int(result.get("tokens", 0) or 0)
        usd = float(result.get("usd", 0.0) or 0.0)
        governor.record(tokens=tokens, usd=usd)
        report.tokens += tokens
        report.usd += usd

    report.stats = queue.stats()
    return report


def aggregate_results(items: list[WorkItem], *, key: str = "findings") -> dict[str, Any]:
    """Зведення по корпусу: скільком айтемам що дісталось, і де саме дірки.

    `dead` перелічуються ОКРЕМО й на видноті: пакетний прогін, який тихо загубив десять документів,
    гірший за той, що впав, — бо перший виглядає успішним.
    """
    findings: list[Any] = []
    dead: list[str] = []
    empty: list[str] = []
    for item in items:
        if item.status == "dead":
            dead.append(item.key)
            continue
        payload = (item.result or {}).get(key)
        if not payload:
            empty.append(item.key)
            continue
        findings.extend(payload if isinstance(payload, list) else [payload])
    return {
        "items": len(items),
        "findings": findings,
        "dead": dead,
        "empty": empty,
        "coverage": round((len(items) - len(dead) - len(empty)) / len(items), 3) if items else 0.0,
    }
