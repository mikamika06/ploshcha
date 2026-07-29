from pydantic import BaseModel

from .tools_fake import EVENT_YEARS, FACTS, FinalAnswerArgs, Tool, _final_answer, _match


class ПеревіркаДатиArgs(BaseModel):
    рік: int
    подія: str


class ПошукФактуArgs(BaseModel):
    сутність: str


class ОбчисленняArgs(BaseModel):
    вираз: str


def _перевірити_дату(a: ПеревіркаДатиArgs) -> dict:
    рік = _match(a.подія, EVENT_YEARS)
    return {"збігається": рік == a.рік, "рік_довідника": рік, "відомо": рік is not None}


def _знайти_факт(a: ПошукФактуArgs) -> dict:
    факт = _match(a.сутність, FACTS)
    return {"факт": факт, "відомо": факт is not None}


def _обчислити(a: ОбчисленняArgs) -> dict:
    if not set(a.вираз) <= set("0123456789+-*/(). "):
        raise ValueError("заборонені символи")
    return {"результат": eval(a.вираз, {"__builtins__": {}}, {})}


UA_TOOLS = [
    Tool("перевірити_дату", "Перевірити, чи рік відповідає історичній події.",
         ПеревіркаДатиArgs, _перевірити_дату),
    Tool("знайти_факт", "Знайти факт про сутність.", ПошукФактуArgs, _знайти_факт),
    Tool("обчислити", "Обчислити арифметичний вираз.", ОбчисленняArgs, _обчислити),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]
