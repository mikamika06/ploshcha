from typing import Literal

from pydantic import BaseModel

Shape = Literal["scalar", "aggregate", "collection"]
SideEffect = Literal["read", "write"]
Trust = Literal["trusted", "untrusted"]

ITERATION_CEILING = 2
NOTE_PREFIX = "skills:collection"


class SkillSpec(BaseModel):
    model_config = {"frozen": True}

    name: str
    capability: str
    shape: Shape = "scalar"
    side_effect: SideEffect = "read"
    trust: Trust = "trusted"
    cost_hint: int = 1
    max_items: int = 0


def collection_skills(specs: list[SkillSpec]) -> list[SkillSpec]:
    return [s for s in specs if s.shape == "collection"]


def aggregate_skills(specs: list[SkillSpec]) -> list[SkillSpec]:
    return [s for s in specs if s.shape == "aggregate"]


def iteration_load(specs: list[SkillSpec]) -> int:
    return max((s.max_items for s in collection_skills(specs)), default=0)


def needs_fanout(specs: list[SkillSpec], *, ceiling: int = ITERATION_CEILING) -> bool:
    """Заміряно (K7c, 72 прогони з 72): плоский цикл робить не більше `ceiling` викликів даних.

    Тому колекція, більша за стелю, плоским циклом не проходиться — це не прогноз, а те, що вже
    сталося. Константа виміряна на гетерогенному routing'у через шлюз; на іншій сітці — перезаміряти.
    """
    return iteration_load(specs) > ceiling


def shape_notes(specs: list[SkillSpec], *, ceiling: int = ITERATION_CEILING) -> list[str]:
    """Гучно, але НЕ корективно: причина потрапляє у звіт, рішення лишається за застосунком."""
    return [f"{NOTE_PREFIX}={s.name}×{s.max_items}"
            for s in collection_skills(specs) if s.max_items > ceiling]


def write_skills(specs: list[SkillSpec]) -> list[SkillSpec]:
    return [s for s in specs if s.side_effect == "write"]


def untrusted_skills(specs: list[SkillSpec]) -> list[SkillSpec]:
    return [s for s in specs if s.trust == "untrusted"]


def scope(specs: list[SkillSpec], capabilities: tuple[str, ...]) -> list[SkillSpec]:
    return [s for s in specs if s.capability in capabilities]


def total_cost_hint(specs: list[SkillSpec]) -> int:
    return sum(s.cost_hint for s in specs)
