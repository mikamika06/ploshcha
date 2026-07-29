from pydantic import BaseModel

from .docs_kb import DOCUMENTS, PARAGRAPHS, ids_for
from .tools_fake import FinalAnswerArgs, Tool, _final_answer, _lemma_set

DOC_MATCH_MIN = 0.5


class ДокументArgs(BaseModel):
    документ: str


class АбзацArgs(BaseModel):
    ідентифікатор: str


def _parts(name: str) -> tuple[str, str]:
    stem, _, number = name.rpartition("-")
    return (stem or name), number


def _match_document(query: str) -> str | None:
    """Стебло — за лемами (вісь U6: «у літописі-3»), НОМЕР — точно.

    Лематизація сама по собі тут небезпечна: «літопис-9» має те саме стебло, що «літопис-1», і
    перекриття лем ≥ порогу віддавало б чужий документ ТИХО. Впевнена підміна гірша за «не знаю».
    """
    q = query.strip().casefold()
    for name in DOCUMENTS:
        if name.casefold() in q or q == name.casefold():
            return name
    numbers = {tok for tok in q.replace("-", " ").split() if tok.isdigit()}
    if len(numbers) != 1:
        return None
    ql = _lemma_set(query)
    for name in DOCUMENTS:
        stem, number = _parts(name)
        if number not in numbers:
            continue
        if _lemma_set(stem) & ql:
            return name
    return None


def _список_абзаців(a: ДокументArgs) -> dict:
    doc = _match_document(a.документ)
    if doc is None:
        return {"відомо": False, "абзаци": []}
    return {"відомо": True, "документ": doc, "абзаци": ids_for(doc)}


def _абзац(a: АбзацArgs) -> dict:
    pid = a.ідентифікатор.strip()
    para = PARAGRAPHS.get(pid)
    if para is None:
        return {"відомо": False, "ідентифікатор": pid}
    return {"відомо": True, "ідентифікатор": pid, **para}


def _абзаци_документа(a: ДокументArgs) -> dict:
    doc = _match_document(a.документ)
    if doc is None:
        return {"відомо": False, "абзаци": []}
    return {"відомо": True, "документ": doc,
            "абзаци": [{"ідентифікатор": pid, **PARAGRAPHS[pid]} for pid in ids_for(doc)]}


DOCS_TOOLS = [
    Tool("список_абзаців", "Перелічити ідентифікатори абзаців документа.",
         ДокументArgs, _список_абзаців),
    Tool("абзац", "Дістати текст одного абзацу за ідентифікатором.", АбзацArgs, _абзац),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]

DOCS_AGG_TOOLS = [
    Tool("абзаци_документа", "Дістати ВСІ абзаци документа одним викликом.",
         ДокументArgs, _абзаци_документа),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]
