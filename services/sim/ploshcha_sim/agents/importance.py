"""Крок `importance`: наскільки подія варта памʼяті."""

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..domain.retrieval import tokens
from ..ports.llm import LlmPort, LlmUsage
from ..ports.trace import StepRecord, TracePort

Strategy = Literal["heuristic", "llm"]

COUNT_MISMATCH = "count_mismatch"
SCHEMA_INVALID = "schema_invalid"
OUT_OF_RANGE = "out_of_range"
TRUNCATED = "truncated"  # провал бюджету, не формату — окремий код
BATCH_TOO_SMALL = "batch_too_small"

BASE = 3
FLOOR, CEIL = 1, 10
# Lapa-12B на пакеті з одного елемента віддає константу 1 НАВІТЬ із якорями:
# без сусідів для порівняння абсолютна оцінка не працює. Нижче цього порогу
# краще детермінована евристика, ніж вироджені дані.
MIN_LLM_BATCH = 4

# Поверхневі маркери подій, що в селі справді важать. Свідомо короткий і
# ревізований список: він є ЕВРИСТИКОЮ-нижньою-межею, а не моделлю світу.
SALIENT = {
    "весілля": 4, "похорон": 4, "помер": 4, "народи": 4, "пожеж": 4, "горить": 4,
    "злодій": 3, "вкрал": 3, "хвор": 3, "лікар": 3, "сварк": 3, "бійк": 3,
    "толок": 2, "ярмарок": 2, "гість": 2, "чужий": 2, "врожай": 2, "дощ": 1,
}
MUNDANE_PREFIXES = ("ти прийшов", "підійшов", "пішов")

RATING_SCHEMA = {
    "type": "object",
    "properties": {"ratings": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ratings"],
    "additionalProperties": False,
}

SYSTEM = (
    "Ти оцінюєш, наскільки подія важлива для мешканця українського села. "
    "1 = цілком буденне, 10 = переломна подія життя. "
    "Відповідай РІВНО одним JSON: {\"ratings\":[<число на кожну подію>]}"
)

# Без опор шкали Lapa-12B віддає константу (розкид 0) і компонент importance
# після нормалізації зникає з формули. Прапорець `anchored`
# лишає це віссю абляції, а не мовчазним дефолтом.
ANCHORS = (
    " Опори шкали: «підійшов сусід» = 1, «сусід позичив сокиру» = 3, "
    "«у селі весілля» = 7, «згоріла хата» = 9, «померла мати» = 10."
)


class ImportanceResult(BaseModel):
    ratings: list[int]
    strategy: Strategy
    schema_valid: bool = True
    reject_reason: str | None = None
    fallback: bool = False
    usage: LlmUsage = Field(default_factory=LlmUsage)
    latency_ms: int = 0


def heuristic_importance(text: str, *, addressed: bool = False) -> int:
    lowered = text.lower()
    value = BASE + (2 if addressed else 0)
    toks = tokens(text)
    for marker, weight in SALIENT.items():
        if any(t.startswith(marker) for t in toks) or marker in lowered:
            value += weight
            break
    if any(p in lowered for p in MUNDANE_PREFIXES):
        value -= 1
    return max(FLOOR, min(CEIL, value))


def heuristic_batch(observations: list[str], addressed: list[bool] | None = None) -> list[int]:
    flags = addressed or [("сказав тобі" in o) for o in observations]
    return [heuristic_importance(o, addressed=f) for o, f in zip(observations, flags)]


def build_importance_prompt(observations: list[str]) -> str:
    listed = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(observations))
    return f"Оціни важливість кожної події від 1 до 10:\n{listed}\n\nРівно {len(observations)} чисел."


def rate_importance(
    observations: list[str],
    *,
    strategy: Strategy = "heuristic",
    llm: LlmPort | None = None,
    addressed: list[bool] | None = None,
    trace: TracePort | None = None,
    run_id: str = "",
    tick: int = 0,
    agent_id: str = "",
    ablation: dict | None = None,
    anchored: bool = True,
    min_batch: int = MIN_LLM_BATCH,
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> ImportanceResult:
    """LLM-стратегія падає в евристику, але провал лишається в трасі як метрика."""
    if not observations:
        return ImportanceResult(ratings=[], strategy=strategy)
    if strategy == "heuristic" or llm is None:
        return ImportanceResult(ratings=heuristic_batch(observations, addressed), strategy="heuristic")
    if len(observations) < min_batch:
        return ImportanceResult(
            ratings=heuristic_batch(observations, addressed),
            strategy="heuristic",
            reject_reason=BATCH_TOO_SMALL,
        )

    prompt = build_importance_prompt(observations)
    res = llm.generate_structured(
        prompt,
        RATING_SCHEMA,
        system=SYSTEM + (ANCHORS if anchored else ""),
        temperature=temperature,
        max_tokens=max_tokens,
    )

    ratings: list[int] | None = None
    reason: str | None = None
    try:
        raw = json.loads(res.text)["ratings"]
        values = [int(v) for v in raw]
        if len(values) != len(observations):
            reason = COUNT_MISMATCH
        elif any(v < FLOOR or v > CEIL for v in values):
            reason = OUT_OF_RANGE
        else:
            ratings = values
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError):
        reason = TRUNCATED if res.finish_reason == "length" else SCHEMA_INVALID

    fallback = ratings is None
    if fallback:
        ratings = heuristic_batch(observations, addressed)

    if trace is not None:
        trace.emit(
            StepRecord(
                run_id=run_id,
                tick=tick,
                agent=agent_id,
                stage="importance",
                model=llm.model,
                prompt=prompt,
                raw_output=res.text,
                parsed={"ratings": ratings},
                schema_valid=not fallback,
                world_valid=not fallback,
                reject_reason=reason,
                usage=res.usage,
                latency_ms=res.latency_ms,
                finish_reason=res.finish_reason,
                ablation=ablation or {},
            )
        )

    return ImportanceResult(
        ratings=ratings,
        strategy="llm",
        schema_valid=not fallback,
        reject_reason=reason,
        fallback=fallback,
        usage=res.usage,
        latency_ms=res.latency_ms,
    )
