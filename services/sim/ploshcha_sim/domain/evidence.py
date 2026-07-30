"""Стан доказів: чи здобув цикл щось, чи інструменти чесно сказали «нема».

Це домен, а не адаптер: на цьому стані стоять три різні рішення системи — чи казати моделі, що
даних немає (`orchestrator`), чи вважати відмову валідним завершенням (`outcome_of`), і чи вимагати
від відповіді доказ (`verify`). Доки стан жив у payload під ключем «відомо», кожен шар угадував
домовленість окремо, і саме тому чесна відмова читалась як збій.
"""

from typing import Literal

FOUND_KEYS = ("відомо", "known", "found")

Outcome = Literal["answer", "abstain", "failure"]


def found_in(value: object) -> bool | None:
    """`None` = інструмент не шукає (обчислення, зведення), тож питання незастосовне."""
    if not isinstance(value, dict):
        return None
    for key in FOUND_KEYS:
        if isinstance(flag := value.get(key), bool):
            return flag
    return None


def evidence_state(scratch: list[dict]) -> bool | None:
    """True — щось знайдено; False — шукали й НЕ знайшли; None — пошуку не було."""
    seen = False
    for entry in scratch:
        flag = entry.get("found")
        if flag is None:
            flag = found_in(entry.get("result"))
        if flag is True:
            return True
        if flag is False:
            seen = True
    return False if seen else None


def outcome_of(answer: str | None, *, degraded: bool, partial: bool,
               evidence: bool | None) -> Outcome:
    """Відмова — це не збій: вона окремий тип завершення, і рахувати її треба окремо.

    Правило детерміноване й без моделі: якщо відповідь непорожня, а всі здобуті докази кажуть
    «не знайдено», то єдина чесна поведінка — визнати незнання, і завершення саме такого типу.
    """
    if answer is None or not answer.strip():
        return "failure"
    if degraded or partial:
        return "failure"
    return "abstain" if evidence is False else "answer"
