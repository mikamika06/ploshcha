from typing import Literal

from pydantic import BaseModel

Shape = Literal["scalar", "aggregate", "collection"]
SideEffect = Literal["read", "write"]
Trust = Literal["trusted", "untrusted"]

# Стеля обходу колекції — властивість КОНФІГУРАЦІЇ, не системи (K7e виправив K7c).
# Без відстеження залишку: 2 виклики, у 104 прогонах з 104 (K7c + K7d, два незалежні набори).
# З відстеженням (`coverage=True`): спостережений ПОВНИЙ обхід до 8 елементів (8/8 викликів),
# але не надійно — 5 повних обходів із 9 задач; на 20 елементах обходу не спостерігалось жодного.
ITERATION_CEILING = 2
ITERATION_CEILING_COVERAGE = 8
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


def ceiling_for(*, coverage: bool = False) -> int:
    return ITERATION_CEILING_COVERAGE if coverage else ITERATION_CEILING


def needs_fanout(specs: list[SkillSpec], *, ceiling: int | None = None,
                 coverage: bool = False) -> bool:
    """Чи колекція більша за те, що ця конфігурація реально обходить.

    Стеля залежить від конфігурації, і це виправлення власного висновку K7c: там я записав 2 як
    властивість системи, а це властивість циклу БЕЗ відстеження залишку. З `coverage=True` той самий
    цикл робив 8 викликів із 8 (K7e). Обидві цифри заміряні на гетерогенному routing'у через шлюз;
    на іншій сітці — перезаміряти, а не успадковувати.
    """
    limit = ceiling_for(coverage=coverage) if ceiling is None else ceiling
    return iteration_load(specs) > limit


def shape_notes(specs: list[SkillSpec], *, ceiling: int | None = None,
                coverage: bool = False) -> list[str]:
    """Гучно, але НЕ корективно: причина потрапляє у звіт, рішення лишається за застосунком."""
    limit = ceiling_for(coverage=coverage) if ceiling is None else ceiling
    return [f"{NOTE_PREFIX}={s.name}×{s.max_items}"
            for s in collection_skills(specs) if s.max_items > limit]


def write_skills(specs: list[SkillSpec]) -> list[SkillSpec]:
    return [s for s in specs if s.side_effect == "write"]


def untrusted_skills(specs: list[SkillSpec]) -> list[SkillSpec]:
    return [s for s in specs if s.trust == "untrusted"]


def scope(specs: list[SkillSpec], capabilities: tuple[str, ...]) -> list[SkillSpec]:
    return [s for s in specs if s.capability in capabilities]


def total_cost_hint(specs: list[SkillSpec]) -> int:
    return sum(s.cost_hint for s in specs)
