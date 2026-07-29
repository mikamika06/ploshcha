import re

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


def _роки_документа(a: ДокументArgs) -> dict:
    """Цифри-роки з кожного абзацу дістає КОД (regex), а не модель.

    Розділяє два діагнози: якщо з цією підказкою бал стрибає — провал був у увазі до довгого тексту;
    якщо не стрибає — це розуміння (заміряно: 4 з 5 провалів були на слові «рік» у абзацах БЕЗ дати).
    """
    doc = _match_document(a.документ)
    if doc is None:
        return {"відомо": False, "абзаци": []}
    out = []
    for pid in ids_for(doc):
        text = PARAGRAPHS[pid]["текст"]
        years = re.findall(r"\b(1[89]\d\d)\b", text)
        out.append({"ідентифікатор": pid, "текст": text,
                    "роки_цифрами": years, "є_рік": bool(years)})
    return {"відомо": True, "документ": doc, "абзаци": out,
            "з_роком": sum(1 for x in out if x["є_рік"]),
            "без_року": [x["ідентифікатор"] for x in out if not x["є_рік"]]}


DOCS_YEARS_TOOLS = [
    Tool("роки_документа",
         "Дістати всі абзаци документа разом із роками, знайденими цифрами, та підсумком.",
         ДокументArgs, _роки_документа),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]

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
