"""Проєкція траси ядра в події контракту 1.1 — перший СПРАВЖНІЙ виробник контракту.

Доти контракт існував як опис і ручна фікстура, тобто був гіпотезою: ніхто не перевіряв, чи поля
взагалі складаються з того, що ядро віддає. Тут це перевіряється — і саме тут виявляється, чого в
трасі бракує (так з'явився `StepRecord.lane`).

Функція чиста: `(records, result) → list[dict]`. Ніякого вводу-виводу, тому її можна прогнати на
фікстурному прогоні без шлюзу й віддати результат TS-валідатору як доказ сумісності.

Два інваріанти, які тут тримаються:
  • `abstain` доїжджає окремим станом (`task.outcome`), а не як помилка;
  • `found` доїжджає тризначним (`true|false|null`), бо «нема даних» ≠ «зламався» ≠ «незастосовно».
"""

from typing import Any

from ..domain.evidence import evidence_state, found_in
from ..domain.gate import FINAL_TOOL
from ..ports.trace import StepRecord

PROTOCOL = "1.1.0"
LANES = ("lapa", "mamay", "unknown")

TOOL_STAGES = ("select", "act", "decide")


def _lane(value: str) -> str:
    return value if value in LANES else "unknown"


def _envelope(run_id: str, seq: int, ts: str, tick: int, type_: str, payload: dict) -> dict:
    return {"protocol": PROTOCOL, "runId": run_id, "seq": seq, "ts": ts, "tick": tick,
            "type": type_, "payload": payload}


def project_run(records: list[StepRecord], result: Any, *, run_id: str, ts: str,
                scene: dict | None = None, started_at: str | None = None) -> list[dict]:
    """Траса + результат → упорядкований потік подій 1.1.

    Результати інструментів беруться з `result.scratch`, бо в трасу вони не потрапляють: `StepRecord`
    фіксує РІШЕННЯ моделі, а не відповідь інструмента. Порядок викликів у трасі й у `scratch`
    однаковий, тож зіставляємо за лічильником.
    """
    events: list[dict] = []
    seq = 0

    def add(type_: str, payload: dict, tick: int = 0) -> None:
        nonlocal seq
        events.append(_envelope(run_id, seq, ts, tick, type_, payload))
        seq += 1

    if scene is not None:
        add("run.started", {"config": {"maxTicks": max(1, len(records)), "castingMode": "library"},
                            "scene": scene, "startedAt": started_at or ts})

    scratch = list(getattr(result, "scratch", []) or [])
    used = 0
    for record in records:
        tick = record.tick
        if record.agent == "notebook":
            if record.stage != "mem_read":
                continue
            payload = record.parsed or {}
            items = payload.get("items") or payload.get("notes") or []
            add("memory.recalled",
                {"items": [str(x) for x in items][:20]} | (
                    {"query": str(payload["query"])} if payload.get("query") else {}),
                tick)
            continue

        if record.agent == "verifier":
            parsed = record.parsed or {}
            add("verify.verdict", {
                "kind": parsed.get("kind", "supported"),
                "accepted": bool(parsed.get("accepted", False)),
                **({"reason": parsed["reason"]} if parsed.get("reason") else {}),
            }, tick)
            continue

        add("route.decided", {"stage": record.stage, "lane": _lane(record.lane),
                              "model": record.model,
                              **({"tier": record.ablation["tier"]}
                                 if record.ablation.get("tier") else {})}, tick)

        call = record.parsed or {}
        tool = call.get("tool")
        # `final_answer` — термінатор циклу, а не інструмент даних: подія про його «виклик» лише
        # засмічувала б потік і ламала парність tool.called/tool.result.
        if not tool or tool == FINAL_TOOL:
            continue
        args = {k: v for k, v in call.items() if k != "tool"}
        add("tool.called", {"tool": tool} | ({"args": args} if args else {}), tick)
        if used < len(scratch):
            entry = scratch[used]
            used += 1
            value = entry.get("result")
            broken = isinstance(value, dict) and "error" in value
            add("tool.result", {"tool": tool, "ok": not broken,
                                "found": entry.get("found", found_in(value)),
                                **({"error": str(value["error"])} if broken else {})}, tick)

    for note in getattr(result, "notes", []) or []:
        add("plan.revised", {"reason": str(note)}, records[-1].tick if records else 0)

    add("task.outcome", {
        "outcome": getattr(result, "outcome", "answer"),
        "evidence": getattr(result, "evidence", evidence_state(scratch)),
        **({"verdictKind": result.verdict_kind} if getattr(result, "verdict_kind", None) else {}),
        **({"incidents": list(result.incidents)} if getattr(result, "incidents", None) else {}),
    }, records[-1].tick if records else 0)
    return events
