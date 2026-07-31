"""Стан доказів: чи здобув цикл щось, чи інструменти чесно сказали «нема».

Це домен, а не адаптер: на цьому стані стоять три різні рішення системи — чи казати моделі, що
даних немає (`orchestrator`), чи вважати відмову валідним завершенням (`outcome_of`), і чи вимагати
від відповіді доказ (`verify`). Доки стан жив у payload під ключем «відомо», кожен шар угадував
домовленість окремо, і саме тому чесна відмова читалась як збій.
"""

from typing import Literal

FOUND_KEYS = ("відомо", "known", "found")

Outcome = Literal["answer", "abstain", "failure"]

# Позначка в `notes`, а не в `incidents`: складання відмови кодом — дія системи, не збій.
ABSENT_COMPOSED = "absent_answer_composed"


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


def failed_queries(scratch: list[dict]) -> list[str]:
    """Що саме питали й не знайшли — потрібне, щоб відмова називала предмет, а не була загальною."""
    out = []
    for entry in scratch:
        flag = entry.get("found")
        if flag is None:
            flag = found_in(entry.get("result"))
        if flag is not False:
            continue
        call = entry.get("call", {})
        args = [str(v) for k, v in call.items() if k != "tool"]
        if args:
            out.append(args[0])
    return list(dict.fromkeys(out))


def absent_answer(scratch: list[dict]) -> str | None:
    """Відмову складає КОД, коли всі докази кажуть «не знайдено».

    K7g показав: щойно крок стає детермінованим, віддавати його моделі — програш (зведення в коді
    дало 1.000 проти 0.556 в оркестрації). Чесна відмова саме такий крок: її зміст повністю
    визначений станом доказів, а модель на ній нестабільна (`pass^k` = 0.000 при t=0.7).

    Це рішення СИСТЕМИ, не спроможність моделі, і в доках воно так і підписане.
    """
    if evidence_state(scratch) is not False:
        return None
    asked = failed_queries(scratch)
    if not asked:
        return "У джерелі немає потрібних даних, тому тлумачення не даю."
    listed = ", ".join(f"«{q}»" for q in asked)
    return (f"У довіднику немає статті за запитом {listed}. "
            f"Тому значення не подаю — вигадувати його не буду.")


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
