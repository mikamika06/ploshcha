MAX_SHOWN = 12
PENDING_LABEL = "Залишилось здобути"
DONE_LABEL = "Усі елементи переліку здобуті"


def collection_items(value) -> list[str]:
    """Перелік із результату колекційного скіла.

    Угода евристична й свідомо вузька: беремо **найдовше** поле-список зі рядкових елементів.
    Реєстр віддає `{"записи": [...]}`, літописи — `{"абзаци": [...]}`; агрегатні набори віддають
    список СЛОВНИКІВ, і тоді покриття не потрібне — тому елементи-нерядки відкидаємо.
    """
    if not isinstance(value, dict):
        return []
    best: list[str] = []
    for field in value.values():
        if not isinstance(field, list) or not field:
            continue
        if not all(isinstance(x, str) for x in field):
            continue
        if len(field) > len(best):
            best = list(field)
    return best


def mark_fetched(pending: list[str], args: dict) -> list[str]:
    """Прибрати з залишку те, що щойно запитали — за будь-яким аргументом виклику."""
    used = {str(v) for v in args.values()}
    return [item for item in pending if item not in used]


def render_pending(pending: list[str], limit: int = MAX_SHOWN) -> str | None:
    if not pending:
        return None
    shown = pending[:limit]
    tail = "" if len(pending) <= limit else f" … ще {len(pending) - limit}"
    return f"{PENDING_LABEL} ({len(pending)}): {', '.join(shown)}{tail}"


def targets_pending(args: dict, pending: list[str]) -> bool:
    """Чи цей виклик — наступний елемент оголошеної колекції.

    Дефект 19: `is_near_duplicate` вважає `запис(зп-1893-02)` після `запис(зп-1893-01)` майже тим
    самим викликом — бо аргументи різняться на кілька символів. Тобто законний обхід колекції
    класифікувався як збій, і драбина відштовхувала модель від нього (0.000, 1 виклик на прогін).
    Відповідь дає стан покриття: якщо аргумент указує на елемент, що ЩЕ в залишку, це ітерація.
    """
    if not pending:
        return False
    return bool({str(v) for v in args.values()} & set(pending))


# ЗАМІРЯНО (K7e@16, негатив): засівати залишок із ТЕКСТУ задачі (слова в лапках) — не працює.
# При ширині 16 покриття зі списку інструмента не мало чого відстежувати, бо елементи названі в
# задачі; спроба засіяти їх звідти дала 16/48 слів проти 21/48 без неї, кроків 17.3 проти 12.7 і
# +47% токенів. Тобто нагадування про залишок з'їдає бюджет, не додаючи обходу. Плюс це зачепило б
# набори `chain`/`docs`, де в задачах теж є лапки. Рання термінація на великій ширині лікується
# розгалуженням (K6), а не бухгалтерією в одному циклі.
