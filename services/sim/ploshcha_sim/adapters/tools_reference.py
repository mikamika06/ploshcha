from pydantic import BaseModel

from ..domain.arith import evaluate
from .reference_kb import NOT_FOUND, article
from .tools_fake import FinalAnswerArgs, Tool, _final_answer


class ДовідкаArgs(BaseModel):
    запит: str


class ОбчисленняArgs(BaseModel):
    вираз: str


def _довідка(a: ДовідкаArgs) -> dict:
    """Віддає СТАТТЮ, не вердикт: дата лежить у суцільному тексті, окремого поля «рік» немає."""
    text = article(a.запит)
    if text is None:
        return {"відомо": False, "запит": a.запит, "стаття": NOT_FOUND}
    return {"відомо": True, "запит": a.запит, "стаття": text}


def _обчислити(a: ОбчисленняArgs) -> dict:
    return {"результат": evaluate(a.вираз)}


REFERENCE_TOOLS = [
    Tool("довідка", "Дістати довідкову статтю про подію або особу.", ДовідкаArgs, _довідка),
    Tool("обчислити", "Обчислити арифметичний вираз.", ОбчисленняArgs, _обчислити),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]
