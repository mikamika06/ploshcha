from pydantic import BaseModel

from ..domain.arith import ArithError, evaluate

from .registry_kb import DISTRACTOR, FIELDS, MISSING, RECORDS, VILLAGES, ids_for
from .tools_fake import FinalAnswerArgs, Tool, _final_answer, _lemma_set

VILLAGE_MATCH_MIN = 0.5


class СписокArgs(BaseModel):
    село: str


class ЗаписArgs(BaseModel):
    ідентифікатор: str


class ОбчисленняArgs(BaseModel):
    вираз: str


def _match_village(query: str) -> str | None:
    q = query.strip().casefold()
    for name in VILLAGES:
        low = name.casefold()
        if low == q or low in q or q in low:
            return name
    ql = _lemma_set(query)
    if not ql:
        return None
    best, score = None, 0.0
    for name in VILLAGES:
        nl = _lemma_set(name)
        if not nl:
            continue
        overlap = len(ql & nl) / len(nl)
        if overlap > score:
            best, score = name, overlap
    return best if score >= VILLAGE_MATCH_MIN else None


def _список_записів(a: СписокArgs) -> dict:
    село = _match_village(a.село)
    if село is None:
        return {"відомо": False, "записи": []}
    return {"відомо": True, "село": село, "записи": ids_for(село)}


def _запис(a: ЗаписArgs) -> dict:
    rid = a.ідентифікатор.strip()
    rec = RECORDS.get(rid)
    if rec is None:
        return {"відомо": False, "ідентифікатор": rid}
    return {"відомо": True, "ідентифікатор": rid, **{f: rec[f] for f in FIELDS}}


def _обчислити(a: ОбчисленняArgs) -> dict:
    return {"результат": evaluate(a.вираз)}


def _записи_села(a: СписокArgs) -> dict:
    """Агрегат замість колекції для обходу — перевірка гіпотези H2 (K7c §12)."""
    село = _match_village(a.село)
    if село is None:
        return {"відомо": False, "записи": []}
    return {"відомо": True, "село": село,
            "записи": [{"ідентифікатор": rid, **{f: rec[f] for f in FIELDS}}
                       for rid, rec in RECORDS.items()
                       if rec["село"] == село and rid != DISTRACTOR]}


ALLOWED_CHARS = "0123456789+-*/(). "


def _обчислити_teach(a: ОбчисленняArgs) -> dict:
    """Помилка мусить УЧИТИ контракту, інакше модель повторює той самий хибний виклик.

    Заміряно (K7f): усі точні повтори були на `обчислити` — модель передавала список
    ідентифікаторів, отримувала «заборонені символи» й повторювала те саме тричі, поки драбина
    не здавалась. Текст помилки не казав ні що дозволено, ні як виправити.
    """
    try:
        return {"результат": evaluate(a.вираз)}
    except ArithError as exc:
        raise ValueError(
            f"{exc} — наприклад «62+41+88». Спершу дістань числа з потрібних записів, "
            f"потім передай їх як арифметичний вираз") from exc


def _запис_teach(a: ЗаписArgs) -> dict:
    rid = a.ідентифікатор.strip()
    rec = RECORDS.get(rid)
    if rec is None:
        return {"відомо": False, "ідентифікатор": rid,
                "підказка": "візьми ідентифікатор дослівно зі списку записів"}
    return {"відомо": True, "ідентифікатор": rid, **{f: rec[f] for f in FIELDS}}


REGISTRY_TOOLS = [
    Tool("список_записів", "Перелічити ідентифікатори записів реєстру для села.",
         СписокArgs, _список_записів),
    Tool("запис", "Дістати поля одного запису реєстру за ідентифікатором.", ЗаписArgs, _запис),
    Tool("обчислити", "Обчислити арифметичний вираз.", ОбчисленняArgs, _обчислити),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]

AGG_TOOLS = [
    Tool("записи_села", "Дістати ВСІ записи реєстру для села одним викликом.",
         СписокArgs, _записи_села),
    Tool("обчислити", "Обчислити арифметичний вираз.", ОбчисленняArgs, _обчислити),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]

class ПідсумокArgs(BaseModel):
    числа: list[float]


def _підсумувати(a: ПідсумокArgs) -> dict:
    """Скіл під те, що агент РЕАЛЬНО робить: підсумувати числа, здобуті з записів.

    Заміряно (K7f/K7g): усі точні повтори були на `обчислити` — модель мусила зліпити рядок-вираз і
    натомість передавала список. Той самий урок, що й з агрегатом: форма інструмента визначає успіх.
    """
    return {"сума": sum(a.числа), "кількість": len(a.числа)}


class ЗвестиArgs(BaseModel):
    ідентифікатори: list[str]


def _звести(a: ЗвестиArgs) -> dict:
    """Зведення рахує КОД, а не модель.

    Заміряно (K7g): щойно оркестрація перестала бути вузьким місцем, провали переїхали в
    арифметику над зібраним — «всього=8, із сумою=6» замість 7, максимум 73 замість 88.
    Логічний кінець правила «форма скіла визначає успіх»: модель маршрутизує, код обчислює.
    """
    known = [(rid, RECORDS[rid]) for rid in a.ідентифікатори if rid in RECORDS]
    with_sum = [(rid, rec) for rid, rec in known if rec["сума"] != MISSING]
    if not with_sum:
        return {"знайдено": len(known), "із_сумою": 0, "сума": 0, "невідомі":
                [r for r in a.ідентифікатори if r not in RECORDS]}
    top = max(with_sum, key=lambda pair: pair[1]["сума"])
    return {
        "знайдено": len(known),
        "із_сумою": len(with_sum),
        "без_суми": [{"ідентифікатор": rid, "майстер": rec["майстер"]}
                     for rid, rec in known if rec["сума"] == MISSING],
        "сума": sum(rec["сума"] for _, rec in with_sum),
        "максимум": top[1]["сума"],
        "максимум_запис": top[0],
        "максимум_майстер": top[1]["майстер"],
        "ремесла": sorted({rec["ремесло"] for _, rec in known}),
        "невідомі": [r for r in a.ідентифікатори if r not in RECORDS],
    }


REGISTRY_REDUCE_TOOLS = [
    Tool("список_записів", "Перелічити ідентифікатори записів реєстру для села.",
         СписокArgs, _список_записів),
    Tool("звести", "Звести перелік записів: сума, кількість, максимум, ремесла, дірки.",
         ЗвестиArgs, _звести),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]

REGISTRY_SUM_TOOLS = [
    Tool("список_записів", "Перелічити ідентифікатори записів реєстру для села.",
         СписокArgs, _список_записів),
    Tool("запис", "Дістати поля одного запису реєстру за ідентифікатором.", ЗаписArgs, _запис),
    Tool("підсумувати", "Скласти числа й повернути суму та кількість.", ПідсумокArgs, _підсумувати),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]

REGISTRY_TEACH_TOOLS = [
    Tool("список_записів", "Перелічити ідентифікатори записів реєстру для села.",
         СписокArgs, _список_записів),
    Tool("запис", "Дістати поля одного запису реєстру за ідентифікатором.", ЗаписArgs, _запис_teach),
    Tool("обчислити", "Обчислити арифметичний вираз із чисел.", ОбчисленняArgs, _обчислити_teach),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]

__all__ = ["REGISTRY_TOOLS", "REGISTRY_TEACH_TOOLS", "REGISTRY_SUM_TOOLS",
           "REGISTRY_REDUCE_TOOLS", "AGG_TOOLS", "MISSING"]
