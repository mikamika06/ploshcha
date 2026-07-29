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
