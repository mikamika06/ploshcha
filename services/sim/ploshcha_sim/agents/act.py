"""Крок `act`: модель обирає одну дію."""

from pydantic import BaseModel, Field, ValidationError

from ..domain.action import Action, Wait, parse_action, wire_action_json_schema
from ..domain.memory import MemoryItem
from ..domain.reducer import validate_action
from ..domain.state import WorldState
from ..ports.llm import LlmPort, LlmUsage
from ..ports.trace import StepRecord, TracePort

SYSTEM = (
    "Ти — мешканець традиційного українського села. Говори і думай природною українською "
    "мовою у сільському регістрі. Відповідай РІВНО одним обʼєктом JSON — дією. "
    "Жодного тексту поза JSON."
)

# Форми дій у промпті обовʼязкові, коли constrained decoding недоступний
# (хостований Lapathoniia мовчки ігнорує guided_json).
ACTION_FORMS = """Формат відповіді — рівно один JSON, одна з форм:
{"type":"move_to","poi":"<id локації>"}
{"type":"speak","to":["<id людини>"],"text":"<що кажеш>"}
{"type":"use_object","poi":"<id локації>"}
{"type":"post_to_board","topic":"<тема>"}
{"type":"reflect"}
{"type":"wait","reason":"<чому>"}"""

SCHEMA_INVALID = "schema_invalid"
# обрізання по max_tokens — це провал БЮДЖЕТУ, не формату; окремий код, щоб не
# псувати метрику schema-validity
TRUNCATED = "truncated"


class ActResult(BaseModel):
    action: Action
    schema_valid: bool
    world_valid: bool
    reject_reason: str | None = None
    raw_output: str = ""
    fallback: bool = False
    usage: LlmUsage = Field(default_factory=LlmUsage)
    latency_ms: int = 0


def build_act_prompt(
    world: WorldState,
    agent_id: str,
    observations: list[str] | None = None,
    memories: list[MemoryItem] | None = None,
) -> str:
    agent = world.agents[agent_id]
    here = [a for a in world.agents_at(agent.location) if a != agent_id]
    # id ТА імʼя разом — інакше модель звертається до імені, а світ знає лише id
    here_labeled = ", ".join(f"{a} ({world.agents[a].persona.name})" for a in here)
    pois = ", ".join(f"{p.id} ({p.name})" for p in world.pois.values())
    usable = ", ".join(p.id for p in world.pois.values() if p.usable)

    lines = [
        f"Ти: {agent.persona.name}, {agent.persona.role}. {agent.persona.bio}".strip(),
        f"Пора доби: {world.time_of_day}. Ти зараз тут: {agent.location}.",
        f"Локації села: {pois}.",
        f"З чим можна взаємодіяти (use_object): {usable or '—'}.",
        f"Поруч із тобою: {here_labeled or 'нікого'}.",
    ]
    if world.board:
        lines.append("На Дошці: " + "; ".join(world.board))
    if observations:
        lines.append("Ти щойно помітив: " + "; ".join(observations))
    if memories:
        lines.append("Пригадуєш: " + "; ".join(m.text for m in memories))
    lines.append(
        "Обери ОДНУ дію. У полях poi і to пиши САМЕ id (не імʼя). "
        "Не вигадуй локацій і людей, яких немає в списках. "
        "Якщо доречної дії немає — обери wait."
    )
    lines.append(ACTION_FORMS)
    return "\n".join(lines)


def act(
    world: WorldState,
    agent_id: str,
    llm: LlmPort,
    *,
    observations: list[str] | None = None,
    memories: list[MemoryItem] | None = None,
    trace: TracePort | None = None,
    run_id: str = "",
    ablation: dict | None = None,
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> ActResult:
    prompt = build_act_prompt(world, agent_id, observations, memories)
    res = llm.generate_structured(
        prompt,
        wire_action_json_schema(),
        system=SYSTEM,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    parsed: Action | None = None
    schema_valid = True
    reason: str | None = None
    try:
        parsed = parse_action_text(res.text)
    except (ValidationError, ValueError):
        schema_valid = False
        reason = TRUNCATED if res.finish_reason == "length" else SCHEMA_INVALID

    world_valid = False
    if parsed is not None:
        reason = validate_action(world, agent_id, parsed)
        world_valid = reason is None

    fallback = not (schema_valid and world_valid)
    action: Action = parsed if not fallback else Wait(reason=reason)  # type: ignore[assignment]

    if trace is not None:
        trace.emit(
            StepRecord(
                run_id=run_id,
                tick=world.tick,
                agent=agent_id,
                stage="act",
                model=llm.model,
                prompt=prompt,
                raw_output=res.text,
                parsed=parsed.model_dump() if parsed else None,
                schema_valid=schema_valid,
                world_valid=world_valid,
                reject_reason=reason,
                usage=res.usage,
                latency_ms=res.latency_ms,
                finish_reason=res.finish_reason,
                ablation=ablation or {},
            )
        )

    return ActResult(
        action=action,
        schema_valid=schema_valid,
        world_valid=world_valid,
        reject_reason=reason,
        raw_output=res.text,
        fallback=fallback,
        usage=res.usage,
        latency_ms=res.latency_ms,
    )


def parse_action_text(text: str) -> Action:
    import json

    return parse_action(json.loads(text))
