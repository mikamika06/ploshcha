"""Породження села: один виклик Mamay на всіх, і жодного разу більше.

Село народжується РАЗ на сід і зберігається. Повернувся — ті самі люди; нове село це свідома дія.
Інакше кожен запуск давав би інших сусідів, і памʼять села, ухвали й стосунки не мали б до кого
кріпитись.
"""

import json

from ..domain.people import (
    Person,
    describe,
    people_schema,
    repair_people,
    roll_traits,
    village_roles,
)

FORGE_TOKENS = 1800

FORGE_SYSTEM = """Ти — Мамай. Тобі дають перелік сільських ролей і норов кожного, вже визначений.
Зроби з цього ЖИВИХ людей: імʼя (як гукали б у селі), одне речення про себе й примовку — коротку
фразу, яку ця людина повторює.

Норов НЕ переписуй і не переказуй: він уже є. Твоє — щоб з нього вийшла людина, яку впізнаєш з
першого слова. Імена різні між собою, без повторів і без «пан/пані»."""


def forge_village(router, effort, *, seed: int, roles: list[str], lenses: dict[str, str],
                  size: int, system: str = FORGE_SYSTEM, budget=None) -> list[Person]:
    """Норов кидає код, людину робить модель, ремонт знову код."""
    picked = village_roles(seed, roles, size)
    traits = {r: roll_traits(seed, r) for r in picked}
    prompt = ("СЕЛО, яке треба заселити:\n"
              + "\n".join(describe(r, traits[r], lenses.get(r, r)) for r in picked)
              + "\n\nЗроби людину з кожного.")

    llm = router.route("decide")
    cfg = effort.effort("decide")
    res = llm.generate_structured(prompt, people_schema(picked), system=system,
                                  temperature=cfg.temperature, max_tokens=FORGE_TOKENS, seed=seed)
    if budget is not None:
        budget.spend(res.usage.total, router.lane("decide"), res.usage.prompt_tokens,
                     stage="decide")
    people = repair_people(_safe(res.text), picked, traits)

    # Кого модель загубила — лишається з роллю замість імені. Гучно й без вигадок: краще людина без
    # історії, ніж село, яке мовчки поменшало.
    have = {p.role for p in people}
    for role in picked:
        if role not in have:
            people.append(Person(role=role, name=role, traits=traits[role]))
    return sorted(people, key=lambda person: picked.index(person.role))


def _safe(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None
