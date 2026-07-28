import json
from typing import Literal

from pydantic import BaseModel, Field

from .retrieval import relevance_chargram

IncidentCode = Literal[
    "not_json", "no_tool_field", "unknown_tool", "bad_args",
    "dup_call", "near_dup_call",
    "tool_error", "tool_unknown",
    "empty_output", "truncated", "budget_exhausted",
]

Rung = Literal["nudge", "retighten", "widen", "escalate", "replan", "partial"]

REJECT_TO_INCIDENT: dict[str, IncidentCode] = {
    "not_json": "not_json",
    "no_tool_field": "no_tool_field",
    "unknown_tool": "unknown_tool",
    "bad_args": "bad_args",
}

PARSE_CODES: tuple[IncidentCode, ...] = ("not_json", "no_tool_field", "unknown_tool", "bad_args")

SOFT_CODES: tuple[IncidentCode, ...] = ("tool_error", "tool_unknown")

LADDERS: dict[IncidentCode, tuple[Rung, ...]] = {
    "not_json": ("retighten", "escalate", "partial"),
    "no_tool_field": ("retighten", "escalate", "partial"),
    "unknown_tool": ("retighten", "escalate", "partial"),
    "bad_args": ("retighten", "escalate", "partial"),
    "dup_call": ("nudge", "replan", "partial"),
    "near_dup_call": ("nudge", "replan", "partial"),
    "tool_unknown": ("nudge",),
    "tool_error": ("nudge",),
    "empty_output": ("retighten", "widen", "escalate", "partial"),
    "truncated": ("widen", "partial"),
    "budget_exhausted": ("partial",),
}

CAPS: dict[Rung, int] = {
    "nudge": 2,
    "retighten": 1,
    "widen": 1,
    "escalate": 1,
    "replan": 1,
    "partial": 1,
}

NEAR_DUP_THRESHOLD = 0.2
WIDEN_FACTOR = 2
WIDEN_CEILING = 2048


class StepOutcome(BaseModel):
    raw_output: str = ""
    finish_reason: str | None = None
    reject_reason: str | None = None
    duplicate: bool = False
    near_duplicate: bool = False
    tool_ok: bool | None = None
    tool_known: bool | None = None
    budget_left: bool = True


class Recovery(BaseModel):
    code: IncidentCode
    rung: Rung
    overrides: dict = Field(default_factory=dict)
    hint: str | None = None


def signature(tool: str, args: dict) -> str:
    return json.dumps({"tool": tool, **args}, sort_keys=True, ensure_ascii=False)


def arg_text(args: dict) -> str:
    return " ".join(str(v) for v in args.values())


def is_near_duplicate(tool: str, args: dict, previous: list[tuple[str, dict]],
                      threshold: float = NEAR_DUP_THRESHOLD,
                      succeeded: list[bool] | None = None) -> bool:
    sig = signature(tool, args)
    text = arg_text(args)
    for index, (prev_tool, prev_args) in enumerate(previous):
        if succeeded is not None and index < len(succeeded) and not succeeded[index]:
            continue
        if prev_tool != tool or signature(prev_tool, prev_args) == sig:
            continue
        if relevance_chargram(text, arg_text(prev_args)) >= threshold:
            return True
    return False


def classify(outcome: StepOutcome) -> IncidentCode | None:
    if not outcome.budget_left:
        return "budget_exhausted"
    if outcome.reject_reason in REJECT_TO_INCIDENT:
        if not outcome.raw_output.strip():
            return "empty_output"
        return REJECT_TO_INCIDENT[outcome.reject_reason]
    if not outcome.raw_output.strip():
        return "empty_output"
    if outcome.finish_reason == "length":
        return "truncated"
    if outcome.duplicate:
        return "dup_call"
    if outcome.near_duplicate:
        return "near_dup_call"
    if outcome.tool_ok is False:
        return "tool_error"
    if outcome.tool_known is False:
        return "tool_unknown"
    return None


def recovery_cap(max_steps: int) -> int:
    return max(1, max_steps // 2)


def policy(code: IncidentCode, attempts: dict[str, int], *,
           plan_exists: bool = False, spent: int = 0, cap_total: int | None = None,
           disabled: tuple[Rung, ...] = ()) -> Rung | None:
    ladder = LADDERS.get(code)
    if ladder is None:
        return None
    exhausted = cap_total is not None and spent >= cap_total
    for rung in ladder:
        if rung in disabled:
            continue
        if attempts.get(rung, 0) >= CAPS[rung]:
            continue
        if rung == "replan" and not plan_exists:
            continue
        if exhausted and rung != "partial":
            continue
        return rung
    return None


def overrides_for(rung: Rung, current_max_tokens: int) -> dict:
    if rung == "retighten":
        return {"tier": "strict"}
    if rung == "escalate":
        return {"tier": "strict"}
    if rung == "widen":
        return {"max_tokens": min(current_max_tokens * WIDEN_FACTOR, WIDEN_CEILING)}
    return {}


_FINISH_FIRST = "Заверши через final_answer, або зроби ІНШИЙ виклик"

HINTS: dict[IncidentCode, str] = {
    "dup_call": f"{_FINISH_FIRST}: цей самий виклик уже зроблено, його результат вище.",
    "near_dup_call": f"{_FINISH_FIRST}: подібний виклик уже зроблено, перефразування не додасть нового.",
    "tool_unknown": f"{_FINISH_FIRST}: інструмент не знає цього — повторювати те саме марно.",
    "tool_error": f"{_FINISH_FIRST}: попередній виклик дав помилку, виправ аргументи або заверши.",
    "not_json": "Віддай РІВНО один JSON-обʼєкт без пояснень до чи після.",
    "no_tool_field": "У JSON обовʼязкове поле \"tool\" з назвою інструмента.",
    "unknown_tool": "Такого інструмента немає. Обери зі списку доступних.",
    "bad_args": "Бракує обовʼязкового аргументу інструмента.",
    "empty_output": "Віддай РІВНО один JSON-обʼєкт із полем \"tool\".",
    "truncated": "Відповідь обрізало. Віддай коротший JSON.",
}


def hint_for(code: IncidentCode, detail: str | None = None) -> str:
    base = HINTS.get(code, "Попередній крок не зайшов. Зміни підхід або заверши.")
    return f"{base} ({detail})" if detail else base


def build(code: IncidentCode, rung: Rung, *, current_max_tokens: int = 256,
          detail: str | None = None) -> Recovery:
    return Recovery(
        code=code,
        rung=rung,
        overrides=overrides_for(rung, current_max_tokens),
        hint=hint_for(code, detail) if rung == "nudge" else None,
    )


def partial_answer(scratch: list[dict]) -> str | None:
    if not scratch:
        return None
    lines = []
    for entry in scratch:
        call = entry.get("call", {})
        result = entry.get("result")
        tool = call.get("tool", "?")
        lines.append(f"{tool}: {json.dumps(result, ensure_ascii=False)}")
    return "Часткова відповідь на основі здобутого:\n" + "\n".join(lines)
