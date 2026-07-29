from pydantic import BaseModel

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
    if not set(a.вираз) <= set("0123456789+-*/(). "):
        raise ValueError("заборонені символи")
    return {"результат": eval(a.вираз, {"__builtins__": {}}, {})}


def _записи_села(a: СписокArgs) -> dict:
    """Агрегат замість колекції для обходу — перевірка гіпотези H2 (K7c §12)."""
    село = _match_village(a.село)
    if село is None:
        return {"відомо": False, "записи": []}
    return {"відомо": True, "село": село,
            "записи": [{"ідентифікатор": rid, **{f: rec[f] for f in FIELDS}}
                       for rid, rec in RECORDS.items()
                       if rec["село"] == село and rid != DISTRACTOR]}


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

__all__ = ["REGISTRY_TOOLS", "AGG_TOOLS", "MISSING"]
