from pydantic import BaseModel

from .tools_fake import FinalAnswerArgs, Tool, _final_answer
from .ua_norm import euphony, feminitive, fix_calques, spelling_2019, vocative


class ТекстArgs(BaseModel):
    текст: str


class СловоArgs(BaseModel):
    слово: str


class КличнийArgs(BaseModel):
    слово: str
    рід: str | None = None


class МилозвучністьArgs(BaseModel):
    наступне_слово: str


def _виправити_кальку(a: ТекстArgs) -> dict:
    found, fixed = fix_calques(a.текст)
    return {"знайдено": found, "виправлено": fixed, "чисто": not found}


def _кличний(a: КличнийArgs) -> dict:
    return {"кличний": vocative(a.слово, a.рід)}


def _милозвучність(a: МилозвучністьArgs) -> dict:
    return {"прийменник": euphony(a.наступне_слово)}


def _фемінітив(a: СловоArgs) -> dict:
    form = feminitive(a.слово)
    return {"фемінітив": form, "відомо": form is not None}


def _правопис(a: СловоArgs) -> dict:
    variants = spelling_2019(a.слово)
    return {"варіанти": variants, "відомо": bool(variants)}


UA_NORM_TOOLS = [
    Tool("виправити_кальку", "Знайти й виправити кальковані конструкції в тексті.",
         ТекстArgs, _виправити_кальку),
    Tool("кличний", "Поставити імʼя або звертання в кличний відмінок.", КличнийArgs, _кличний),
    Tool("милозвучність", "Обрати «у» або «в» перед наступним словом.",
         МилозвучністьArgs, _милозвучність),
    Tool("фемінітив", "Утворити фемінітив за правописом 2019.", СловоArgs, _фемінітив),
    Tool("правопис", "Варіанти слова за правописом 2019.", СловоArgs, _правопис),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]
