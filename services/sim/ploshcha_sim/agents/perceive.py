"""Крок `perceive`: світ -> спостереження + памʼять. Без LLM."""

from ..domain.memory import MemoryItem
from ..domain.state import WorldState

# Ті самі мітки `id (Імʼя)`, що в промпті act — інакше модель звертається до
# імені, а світ знає лише id (реальний баг).
def label(world: WorldState, agent_id: str) -> str:
    a = world.agents.get(agent_id)
    return f"{agent_id} ({a.persona.name})" if a else agent_id


def perceive(
    world: WorldState,
    agent_id: str,
    previous: WorldState | None = None,
) -> list[str]:
    """Що агент помітив ЦЬОГО тіку. Порядок стабільний: місце, люди, мова, дошка."""
    me = world.agents[agent_id]
    here_now = [a for a in world.agents_at(me.location) if a != agent_id]
    out: list[str] = []

    prev_me = previous.agents.get(agent_id) if previous else None
    if prev_me is not None and prev_me.location != me.location:
        poi = world.pois.get(me.location)
        out.append(f"Ти прийшов у {me.location} ({poi.name if poi else me.location}).")

    if previous is not None:
        here_before = {a for a in previous.agents_at(me.location) if a != agent_id}
        for a in here_now:
            if a not in here_before:
                out.append(f"{label(world, a)} підійшов.")
        for a in sorted(here_before - set(here_now)):
            out.append(f"{label(world, a)} пішов.")

    for u in world.utterances_at(me.location, world.tick):
        if u.speaker == agent_id:
            continue
        if agent_id in u.to:
            out.append(f"{label(world, u.speaker)} сказав тобі: «{u.text}»")
        elif not u.to:
            out.append(f"{label(world, u.speaker)} сказав усім: «{u.text}»")
        else:
            out.append(f"Ти почув, як {label(world, u.speaker)} каже: «{u.text}»")

    before_board = set(previous.board) if previous else set()
    for topic in world.board:
        if topic not in before_board:
            out.append(f"На Дошці зʼявилось: «{topic}»")

    return out


def to_memories(
    observations: list[str],
    importances: list[int],
    tick: int,
    agent_id: str,
    *,
    seq_start: int = 0,
) -> list[MemoryItem]:
    if len(observations) != len(importances):
        raise ValueError("кількість оцінок не збігається з кількістю спостережень")
    return [
        MemoryItem(
            id=f"{agent_id}:t{tick}:{seq_start + i}",
            tick=tick,
            kind="observation",
            text=text,
            importance=imp,
        )
        for i, (text, imp) in enumerate(zip(observations, importances))
    ]
