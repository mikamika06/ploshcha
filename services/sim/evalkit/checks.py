import json
import re

DUMP_MIN_LEN = 24
JSON_SIGNS = ('{"', '":')


def _answer(result) -> str:
    return result.answer or ""


def _dumped(result, min_len: int = DUMP_MIN_LEN) -> bool:
    """Відповідь, яка дослівно містить payload інструмента, — це дамп, а не відповідь.

    Драбина відновлення на рунзі `partial` вивалює сирі результати в текст відповіді, і тоді
    будь-який змістовий чек проходить тривіально: потрібне значення лежить усередині дампа.
    """
    answer = _answer(result)
    for entry in result.scratch:
        payload = json.dumps(entry.get("result", {}), ensure_ascii=False)
        if len(payload) >= min_len and payload in answer:
            return True
    return False


MIN_PREFIX = 4
WORD = re.compile(r"[а-яїієґА-ЯЇІЄҐ'\u2019]+")


def _stem_hit(answer: str, values: list[str], min_prefix: int) -> bool:
    """Збіг ПОВНИХ слів за спільним префіксом, а не підрядком обрізаного стема.

    Підрядковий чек вісім разів давав хибні провали на відмінюванні: «лайка» не міститься в «лайку»,
    «діти» — в «дітей», «бажати» — в «бажання». Лематизація не ліки (`pymorphy3`: «різці» → «різка»),
    тому міряємо спільний префікс двох повних слів.

    `len(want) - 1` дозволяє РІВНО одну літеру розходження в короткому ключі («діти» ↔ «дітей»,
    спільне лише «діт»). Більше послаблення почало б приймати чужі слова: «вівця» проти «жінка» і
    «засуджувати» проти «лаяти» мусять лишатись провалами, і лишаються.

    Токен КОРОТШИЙ за поріг не рахується: інакше службове «до» задовольняло б ключ «доглядати», і
    чуже значення проходило б — гейт V6 зловив це на чотирьох айтемах одразу.
    """
    tokens = [w.casefold() for w in WORD.findall(answer)]
    for value in values:
        want = value.casefold()
        need = min(min_prefix, len(want) - 1)
        if need < 3:
            continue
        for token in tokens:
            if len(token) < need:
                continue
            shared = 0
            for a, b in zip(want, token, strict=False):
                if a != b:
                    break
                shared += 1
            if shared >= need:
                return True
    return False


def _tools(result) -> list[str]:
    return [x["call"]["tool"] for x in result.scratch]


def check(spec: dict, result) -> bool:
    kind = spec["kind"]
    if kind == "answer_contains":
        return spec["value"].casefold() in _answer(result).casefold()
    if kind == "answer_not_contains":
        return spec["value"].casefold() not in _answer(result).casefold()
    if kind == "answer_stem_any":
        return _stem_hit(_answer(result), spec["values"], spec.get("min_prefix", MIN_PREFIX))
    if kind == "answer_contains_any":
        text = _answer(result).casefold()
        return any(v.casefold() in text for v in spec["values"])
    if kind == "answer_contains_all":
        text = _answer(result).casefold()
        return all(v.casefold() in text for v in spec["values"])
    if kind == "used_tool":
        return spec["tool"] in _tools(result)
    if kind == "used_tool_any":
        return any(t in _tools(result) for t in spec["tools"])
    if kind == "no_data_tool":
        return len(_tools(result)) == 0
    if kind == "abstain":
        return len(_tools(result)) == 0 and bool(_answer(result).strip())
    if kind == "outcome_is":
        return getattr(result, "outcome", "answer") == spec["value"]
    if kind == "not_rejected":
        return bool(getattr(result, "accepted", False))
    if kind == "verdict_kind_in":
        return getattr(result, "verdict_kind", None) in set(spec["kinds"])
    if kind == "multi_hop":
        return len(set(_tools(result))) >= spec.get("n", 2)
    if kind == "accepted":
        return result.accepted
    if kind == "answered":
        return bool(_answer(result).strip()) and not result.degraded
    if kind == "no_incident":
        code = spec.get("code")
        incidents = list(getattr(result, "incidents", []))
        return code not in incidents if code else not incidents
    if kind == "steps_between":
        lo, hi = spec.get("lo", 0), spec.get("hi", 10**6)
        return lo <= result.steps <= hi
    if kind == "tool_calls_at_most":
        return len(_tools(result)) <= spec["n"]
    if kind == "tool_calls_at_least":
        return len(_tools(result)) >= spec["n"]
    if kind == "not_degraded":
        return not result.degraded
    if kind == "not_partial":
        return not getattr(result, "partial", False)
    if kind == "answer_not_dumped":
        return not _dumped(result, spec.get("min_len", DUMP_MIN_LEN))
    if kind == "answer_no_json":
        return not any(sign in _answer(result) for sign in JSON_SIGNS)
    raise ValueError(f"unknown check kind: {kind}")


# `not_rejected` і `verdict_kind_in` — гігієна СВІДОМО: це думка верифікатора, а не результат задачі.
# Змішавши їх з результатом, ми знову міряли б суддю замість моделі (K9: страта чесних відмов 0/8).
HYGIENE_KINDS = frozenset({
    "no_incident", "tool_calls_at_most", "tool_calls_at_least", "not_partial", "steps_between",
    "abstain", "no_data_tool", "not_rejected", "verdict_kind_in",
})


def is_hygiene(spec: dict) -> bool:
    return spec["kind"] in HYGIENE_KINDS


def run_checks(specs: list[dict], result) -> dict[str, bool]:
    return {str(s): check(s, result) for s in specs}


def split_checks(specs: list[dict], result) -> tuple[dict[str, bool], dict[str, bool]]:
    """Успіх задачі рахується ЛИШЕ за результатом; гігієна шляху — окремо."""
    outcome, hygiene = {}, {}
    for spec in specs:
        (hygiene if is_hygiene(spec) else outcome)[str(spec)] = check(spec, result)
    return outcome, hygiene


def outcome_tier(result) -> str:
    """повна відповідь / часткова з доказом / нічого — це три різні речі."""
    if result.answer is None or not str(result.answer).strip():
        return "empty"
    if getattr(result, "partial", False):
        return "partial" if result.scratch else "empty"
    return "full"
