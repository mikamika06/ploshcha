"""Розподіл задачі й бюджету між суб-агентами — чистий домен.

Розподіл ДЕТЕРМІНОВАНИЙ і не звертається до моделі. Причина не в економії: K7g/K7h/ABSTAIN тричі
показали, що крок, повністю визначений даними, віддавати моделі — програш. Розбити «поясни ці чотири
слова» на чотири підзадачі — рівно такий крок.

`budget.child(n)` ділить стелю, а не копіює її: інакше N дітей витратять N×бюджет, і «масштаб»
означав би просто «дорожче».
"""

import re

from pydantic import BaseModel

from .task import Budget

MIN_CHILD_STEPS = 2
QUOTED = re.compile(r"[«\"']([^«»\"']{2,40})[»\"']")


class Split(BaseModel):
    """Підзадачі + бюджет на кожну. `depth` рахується від кореня, щоб `max_depth` був перевірюваним."""

    tasks: list[str]
    child_budget: Budget
    depth: int = 1

    @property
    def width(self) -> int:
        return len(self.tasks)


def child_budget(budget: Budget, parts: int) -> Budget:
    """Стеля ділиться на кількість дітей, але не нижче за поріг, на якому крок ще можливий."""
    parts = max(1, parts)
    return Budget(
        max_steps=max(MIN_CHILD_STEPS, budget.max_steps // parts),
        max_tokens=max(1, (budget.max_tokens - budget.tokens_used) // parts),
    )


def split_by_items(task: str, items: list[str], template: str) -> list[str]:
    """Одна підзадача на елемент. `template` тримає формулювання поза доменом."""
    return [template.format(item=item, task=task) for item in items]


def quoted_items(task: str) -> list[str]:
    """Елементи, названі в лапках, — найдешевший детермінований спосіб побачити фан-аут у тексті.

    Свідомо НЕ просимо модель «розбити задачу»: розбиття тут визначене текстом, а не судженням.
    """
    return list(dict.fromkeys(QUOTED.findall(task)))


def plan_split(task: str, budget: Budget, *, template: str, depth: int = 1,
               max_depth: int = 2, min_width: int = 2) -> Split | None:
    """`None` означає «фан-аут не виправданий» — і це нормальний, найчастіший випадок.

    Гейт по глибині стоїть ДО розбору тексту: інакше на дні рекурсії ми б платили за розбір, який
    однаково нікуди не піде.
    """
    if depth > max_depth:
        return None
    items = quoted_items(task)
    if len(items) < min_width:
        return None
    return Split(tasks=split_by_items(task, items, template),
                 child_budget=child_budget(budget, len(items)), depth=depth)
