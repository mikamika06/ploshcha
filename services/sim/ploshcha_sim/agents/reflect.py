"""Крок `reflect`: спостереження -> питання -> висновки."""

import json

from pydantic import BaseModel, Field

from ..domain.memory import MemoryItem
from ..domain.retrieval import Retrieved, relevance_chargram
from ..domain.state import AgentState, WorldState
from ..ports.llm import LlmPort, LlmUsage
from ..ports.trace import StepRecord, TracePort
from .importance import heuristic_importance
from .recall import recall

SCHEMA_INVALID = "schema_invalid"
EVIDENCE_INVALID = "evidence_invalid"
TRUNCATED = "truncated"  # провал бюджету, не формату — окремий код
EMPTY = "empty"

# Порог накопиченої важливості. Park мав 150 при десятках подій/год;
# у нас 6 тіків на добу, тож шкала інша.
THRESHOLD = 25
QUESTIONS = 3
RECENT_WINDOW = 12
EVIDENCE_K = 8
REFLECTION_FLOOR = 5
# Різні питання тягнуть ті самі згадки, тож моделі повторюють думку майже
# дослівно. Частка викинутих = показник повторюваності рефлексії.
DEDUP_THRESHOLD = 0.85

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}

INSIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["insights"],
    "additionalProperties": False,
}

SYSTEM_Q = (
    "Ти мешканець українського села, що обмірковує останні дні. "
    "Сформулюй питання про себе й людей навколо — не про світ узагалі. "
    "Відповідай РІВНО одним JSON: {\"questions\":[…]}"
)
SYSTEM_I = (
    "Ти робиш висновки зі СВОЇХ згадок. Кожен висновок мусить опиратись на "
    "номери згадок, які тобі дали; не вигадуй номерів. "
    "Відповідай РІВНО одним JSON: {\"insights\":[{\"text\":…,\"evidence\":[номери]}]}"
)


class Insight(BaseModel):
    text: str
    evidence: list[str] = Field(default_factory=list)
    dropped_citations: int = 0


class ReflectResult(BaseModel):
    items: list[MemoryItem] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    schema_valid: bool = True
    reject_reason: str | None = None
    dropped_citations: int = 0
    duplicates_dropped: int = 0
    usage: LlmUsage = Field(default_factory=LlmUsage)
    latency_ms: int = 0
    llm_calls: int = 0


def last_reflection_tick(agent: AgentState) -> int:
    ticks = [m.tick for m in agent.memory if m.kind == "reflection"]
    return max(ticks) if ticks else -1


def accumulated_importance(agent: AgentState, since_tick: int) -> int:
    return sum(m.importance for m in agent.memory if m.tick > since_tick)


def should_reflect(agent: AgentState, threshold: int = THRESHOLD) -> bool:
    return accumulated_importance(agent, last_reflection_tick(agent)) >= threshold


def recent(agent: AgentState, window: int = RECENT_WINDOW) -> list[MemoryItem]:
    return sorted(agent.memory, key=lambda m: (m.tick, m.id))[-window:]


def build_questions_prompt(agent: AgentState, memories: list[MemoryItem]) -> str:
    listed = "\n".join(f"- {m.text}" for m in memories)
    return (
        f"Ти: {agent.persona.name}, {agent.persona.role}.\n"
        f"Останнє, що ти пережив:\n{listed}\n\n"
        f"Постав {QUESTIONS} найважливіші питання, які ти сам собі поставив би зараз."
    )


def build_insights_prompt(question: str, hits: list[Retrieved]) -> str:
    listed = "\n".join(f"{i + 1}. {h.item.text}" for i, h in enumerate(hits))
    return (
        f"Питання: {question}\n\nТвої згадки:\n{listed}\n\n"
        "Зроби 1-3 висновки. У evidence вкажи НОМЕРИ згадок, на які опираєшся."
    )


def _parse(text: str, key: str) -> list | None:
    try:
        value = json.loads(text)[key]
        return value if isinstance(value, list) else None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def reflect(
    world: WorldState,
    agent_id: str,
    llm: LlmPort,
    *,
    questions: int = QUESTIONS,
    window: int = RECENT_WINDOW,
    evidence_k: int = EVIDENCE_K,
    dedup_threshold: float = DEDUP_THRESHOLD,
    trace: TracePort | None = None,
    run_id: str = "",
    ablation: dict | None = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> ReflectResult:
    agent = world.agents[agent_id]
    window_items = recent(agent, window)
    if not window_items:
        return ReflectResult(schema_valid=False, reject_reason=EMPTY)

    usage = LlmUsage()
    latency = 0
    calls = 0
    last_reason: str | None = None

    def run(prompt: str, schema: dict, system: str, key: str, stage: str) -> tuple[list | None, str]:
        nonlocal latency, calls, last_reason
        res = llm.generate_structured(
            prompt, schema, system=system, temperature=temperature, max_tokens=max_tokens
        )
        calls += 1
        usage.prompt_tokens += res.usage.prompt_tokens
        usage.completion_tokens += res.usage.completion_tokens
        latency += res.latency_ms
        parsed = _parse(res.text, key)
        if parsed is None:
            last_reason = TRUNCATED if res.finish_reason == "length" else SCHEMA_INVALID
        if trace is not None:
            trace.emit(
                StepRecord(
                    run_id=run_id,
                    tick=world.tick,
                    agent=agent_id,
                    stage=stage,
                    model=llm.model,
                    prompt=prompt,
                    raw_output=res.text,
                    parsed={key: parsed} if parsed is not None else None,
                    schema_valid=parsed is not None,
                    world_valid=parsed is not None,
                    reject_reason=None if parsed is not None else last_reason,
                    usage=res.usage,
                    latency_ms=res.latency_ms,
                    finish_reason=res.finish_reason,
                    ablation=ablation or {},
                )
            )
        return parsed, res.text

    raw_questions, _ = run(
        build_questions_prompt(agent, window_items),
        QUESTIONS_SCHEMA,
        SYSTEM_Q,
        "questions",
        "reflect.questions",
    )
    if raw_questions is None:
        return ReflectResult(
            schema_valid=False, reject_reason=last_reason, usage=usage, latency_ms=latency,
            llm_calls=calls,
        )

    asked = [str(q) for q in raw_questions if str(q).strip()][:questions]
    insights: list[Insight] = []
    dropped = 0
    duplicates = 0

    for question in asked:
        hits = recall(
            world, agent_id, query=question, k=evidence_k,
            trace=trace, run_id=run_id, ablation=ablation,
        )
        if not hits:
            continue
        raw_insights, _ = run(
            build_insights_prompt(question, hits),
            INSIGHTS_SCHEMA,
            SYSTEM_I,
            "insights",
            "reflect.insights",
        )
        if raw_insights is None:
            continue
        for entry in raw_insights:
            if not isinstance(entry, dict) or not str(entry.get("text", "")).strip():
                continue
            text = str(entry["text"]).strip()
            if any(relevance_chargram(text, kept.text) >= dedup_threshold for kept in insights):
                duplicates += 1
                continue
            ids: list[str] = []
            bad = 0
            for n in entry.get("evidence") or []:
                # номер поза діапазоном = вигадане посилання; це метрика, не помилка коду
                if isinstance(n, int) and 1 <= n <= len(hits):
                    ids.append(hits[n - 1].item.id)
                else:
                    bad += 1
            dropped += bad
            insights.append(Insight(text=text, evidence=ids, dropped_citations=bad))

    items = [
        MemoryItem(
            id=f"{agent_id}:t{world.tick}:r{i}",
            tick=world.tick,
            kind="reflection",
            text=ins.text,
            importance=max(REFLECTION_FLOOR, heuristic_importance(ins.text)),
            provenance=f"agent:{agent_id}",
            evidence=ins.evidence,
        )
        for i, ins in enumerate(insights)
    ]

    return ReflectResult(
        items=items,
        questions=asked,
        insights=insights,
        schema_valid=bool(items),
        reject_reason=(EVIDENCE_INVALID if dropped and items else (None if items else last_reason)),
        dropped_citations=dropped,
        duplicates_dropped=duplicates,
        usage=usage,
        latency_ms=latency,
        llm_calls=calls,
    )
