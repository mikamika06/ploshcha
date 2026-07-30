from pydantic import BaseModel

from .lexicon_kb import NOT_FOUND, article
from .tools_fake import FinalAnswerArgs, Tool, _final_answer


class СловникArgs(BaseModel):
    слово: str


def _словник(a: СловникArgs) -> dict:
    text = article(a.слово)
    if text is None:
        return {"відомо": False, "слово": a.слово, "стаття": NOT_FOUND}
    return {"відомо": True, "слово": a.слово, "стаття": text}


LEXIS_TOOLS = [
    Tool("словник", "Дістати словникову статтю про рідкісне українське слово.", СловникArgs, _словник),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]
