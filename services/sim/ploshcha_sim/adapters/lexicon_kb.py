"""Довідник рідкісної української лексики: інструмент віддає СЛОВНИКОВУ СТАТТЮ.

Це клас «довідка» в чистому вигляді — те, чого модель не пам'ятає. На відміну від `reference_kb`
(загальновідомі факти, де довідка нічого не додає), тут лексика така, що без статті відповідь можна
лише вгадати.

Дані заморожені в `evalkit/data/lexis-uk.json` і здобуті з зовнішнього джерела дослівно
(uk.wiktionary, CC BY-SA 4.0): еталон не писався з голови.
"""

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "evalkit" / "data" / "lexis-uk.json"
NOT_FOUND = "слова немає в довіднику"
STEM = 4


ABSENT_STRATUM = "absent"


@lru_cache(maxsize=1)
def _entries() -> dict[str, dict]:
    """Страт `absent` НЕ потрапляє в довідник: на цих словах міряється чесна відмова."""
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    return {e["слово"]: e for e in payload["статті"] if e["страт"] != ABSENT_STRATUM}


@lru_cache(maxsize=1)
def _held_out() -> dict[str, dict]:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    return {e["слово"]: e for e in payload["статті"] if e["страт"] == ABSENT_STRATUM}


def _resolve(query: str) -> str | None:
    """Форма в запиті може бути відмінена. Збіг за стемом, але з вибором НАЙБЛИЖЧОГО слова.

    Без цього «ватри» тягне «ватралка» так само добре, як «ватра», і довідник тихо віддає не те.
    """
    q = query.strip().casefold()
    words = _entries()
    if q in words:
        return q
    same = [w for w in words if w[:STEM] == q[:STEM]]
    if not same:
        return None

    def _shared(word: str) -> int:
        return sum(1 for _ in zip(word, q, strict=False))

    return min(same, key=lambda w: (-_shared(w), abs(len(w) - len(q))))


def article(query: str) -> str | None:
    word = _resolve(query)
    if word is None:
        return None
    entry = _entries()[word]
    senses = "; ".join(entry["значення"])
    parts = [f"{word} — {senses}."]
    if entry["приклади"]:
        parts.append(f"Приклад слововжитку: «{entry['приклади'][0]}»")
    if entry["джерело"]:
        parts.append(f"Джерело: {entry['джерело'][0]}")
    return " ".join(parts)


def words(stratum: str | None = None) -> list[str]:
    table = _held_out() if stratum == ABSENT_STRATUM else _entries()
    return sorted(w for w, e in table.items() if stratum is None or e["страт"] == stratum)
