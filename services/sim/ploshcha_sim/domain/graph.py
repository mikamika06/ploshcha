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
    """Групи елементів + бюджет на групу. `depth` рахується від кореня, щоб `max_depth` перевірявся.

    Групи, а не плоский список підзадач: якщо елементів більше за дозволену ширину, вони ріжуться на
    частини, і кожна частина ділиться далі на наступному рівні. Так N необмежене, а ширина вузла —
    ні. Раніше перевищення ширини скочувало все в ОДИН цикл, тобто рівно в той режим, який при
    ширині 16 здається на 12.7 кроках із 64.
    """

    groups: list[list[str]]
    tasks: list[str]
    child_budget: Budget
    depth: int = 1

    @property
    def width(self) -> int:
        return len(self.groups)

    @property
    def leaves(self) -> int:
        return sum(len(g) for g in self.groups)


def child_budget(budget: Budget, parts: int) -> Budget:
    """Стеля ділиться на кількість дітей, але не нижче за поріг, на якому крок ще можливий."""
    parts = max(1, parts)
    return Budget(
        max_steps=max(MIN_CHILD_STEPS, budget.max_steps // parts),
        max_tokens=max(1, (budget.max_tokens - budget.tokens_used) // parts),
    )


def chunk(items: list[str], size: int) -> list[list[str]]:
    """Рівні частини, не «перші size і решта»: інакше остання дитина несла б увесь хвіст."""
    size = max(1, size)
    parts = -(-len(items) // size)
    if parts <= 1:
        return [list(items)]
    base, extra = divmod(len(items), parts)
    out, at = [], 0
    for i in range(parts):
        take = base + (1 if i < extra else 0)
        out.append(items[at:at + take])
        at += take
    return out


def split_by_items(task: str, items: list[str], template: str) -> list[str]:
    """Одна підзадача на елемент. `template` тримає формулювання поза доменом."""
    return [template.format(item=item, task=task) for item in items]


def group_task(task: str, group: list[str], *, one: str, many: str) -> str:
    """Група з одного елемента — це листок; група з кількох — задача, яку ще можна різати."""
    if len(group) == 1:
        return one.format(item=group[0], task=task)
    listed = ", ".join(f"«{item}»" for item in group)
    return many.format(items=listed, task=task)


def quoted_items(task: str) -> list[str]:
    """Елементи, названі в лапках, — найдешевший детермінований спосіб побачити фан-аут у тексті.

    Свідомо НЕ просимо модель «розбити задачу»: розбиття тут визначене текстом, а не судженням.
    """
    return list(dict.fromkeys(QUOTED.findall(task)))


def plan_split(task: str, budget: Budget, *, template: str, depth: int = 1,
               max_depth: int = 2, min_width: int = 2, max_width: int = 6,
               many_template: str | None = None) -> Split | None:
    """`None` означає «фан-аут не виправданий» — нормальний і найчастіший випадок.

    Гейт по глибині стоїть ДО розбору тексту: інакше на дні рекурсії ми платили б за розбір, який
    однаково нікуди не піде.
    """
    if depth > max_depth:
        return None
    items = quoted_items(task)
    if len(items) < min_width:
        return None
    # Вміщається у ширину — одна дитина на елемент (листки). Не вміщається — ріжемо на частини, і
    # кожну ріже наступний рівень. Без цієї розвилки «одна група з усіх елементів» рекурсувала б на
    # тій самій задачі й давала вкладене зведення замість роботи.
    # `max_width` обмежує КІЛЬКІСТЬ ДІТЕЙ, а не розмір групи. Різати по розміру означало б, що при
    # 100 елементах вузол має 17 дітей — тобто ширина знову необмежена. Тому розмір групи
    # обчислюється як ⌈N / max_width⌉, і глибина дерева зростає як log(N).
    groups = ([[item] for item in items] if len(items) <= max_width
              else chunk(items, -(-len(items) // max_width)))
    if many_template is None:
        tasks = [template.format(item=g[0], task=task) if len(g) == 1
                 else template.format(item=", ".join(g), task=task) for g in groups]
    else:
        tasks = [group_task(task, g, one=template, many=many_template) for g in groups]
    return Split(groups=groups, tasks=tasks,
                 child_budget=child_budget(budget, len(groups)), depth=depth)
