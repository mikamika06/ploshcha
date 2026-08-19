import json
from typing import Literal

from pydantic import BaseModel

from ..ports.router import EffortPolicy, ModelRouter
from ..ports.trace import StepRecord, TracePort

VerdictKind = Literal["supported", "abstain", "unsupported", "contradicted", "parse_fail",
                      "no_evidence"]

ACCEPTING: tuple[VerdictKind, ...] = ("supported", "abstain", "no_evidence")

# Заміряно (K9): із доказами заземлений суддя дає 32/32 згоди з правдою, без доказів — 20/32 проти
# 21/32 у базового, тобто лише переставляє помилки з «хибно прийняв» на «хибно відкинув». Без доказів
# заземлювати НІ НА ЧОМУ, тому режим `auto` не витрачає токенів і не вдає, що має думку.
NO_EVIDENCE_REASON = "немає доказів для заземлення — вердикт не виносився"

# Режим `contrast` пише чотири поля замість двох, і на бюджеті судді вивід обрізало по `max_tokens` —
# усі 77 клітинок пішли в `parse_fail`. Структура була правильна (модель давала `contradicted`),
# помирала лише довжина. Тому в цього режиму власна підлога бюджету.
CONTRAST_MIN_TOKENS = 320

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"accepted": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["accepted", "reason"],
    "additionalProperties": False,
}

# `contrast`: ті самі види, але схема ЗМУШУЄ спершу виписати два твердження й лише тоді обрати вид.
# Інструкція «спершу порівняй зміст» не спрацювала (згода 69/77 → 68/77, помилки лишились), а в цьому
# проєкті структура вже не раз била інструкцію — це перевірка того самого важеля на суддівському кроці.
CONTRAST_SCHEMA = {
    "type": "object",
    "properties": {
        "доказ_каже": {"type": "string", "description": "що стверджує доказ, одним рядком"},
        "відповідь_каже": {"type": "string", "description": "що стверджує відповідь, одним рядком"},
        "той_самий_референт": {"type": "boolean"},
        "kind": {"type": "string",
                 "enum": ["supported", "abstain", "unsupported", "contradicted"]},
        "reason": {"type": "string"},
    },
    "required": ["доказ_каже", "відповідь_каже", "той_самий_референт", "kind"],
    "additionalProperties": False,
}

GROUNDED_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string",
                 "enum": ["supported", "abstain", "unsupported", "contradicted"]},
        "reason": {"type": "string"},
    },
    "required": ["kind", "reason"],
    "additionalProperties": False,
}

SYSTEM = (
    "Ти перевіряльник. Оціни, чи відповідь правильна Й підтверджена доказами (результатами "
    "інструментів). Якщо докази підтверджують відповідь — accepted=true. Відкидай (false) лише за "
    "фактичну помилку чи суперечність доказам, не за брак довіри до самих інструментів. "
    "Відповідай РІВНО одним JSON: {\"accepted\":<bool>,\"reason\":\"<чому>\"}"
)

GROUNDED_HEAD = (
    "Ти перевіряльник. Твоє завдання — не оцінити тон чи повноту, а встановити, чи стоїть за "
    "відповіддю доказ. Обери РІВНО один вид:\n"
    "• supported — твердження відповіді ЗБІГАЄТЬСЯ ЗІ ЗМІСТОМ доказу. Сам факт, що інструмент щось "
    "повернув, підтвердженням НЕ є: спершу прочитай, що саме сказано в доказі, і лише тоді "
    "порівнюй;\n"
    "• abstain — доказів немає або інструмент повідомив, що даних немає, І відповідь чесно це "
    "визнає, не вигадуючи змісту. Це ПРАВИЛЬНА поведінка, а не провал;\n"
    "• unsupported — відповідь стверджує зміст, під який доказу немає (вигадка);\n"
    "• contradicted — доказ Є, але відповідь каже ІНШЕ, ніж у ньому. Це найчастіша пастка: "
    "інструмент знайшов статтю, а відповідь переказує щось своє — тоді це contradicted, не "
    "supported.\n"
    "Довжина, впевненість тону й обсяг пояснень НЕ впливають на вид: коротке чесне «даних немає» — "
    "це abstain, а не unsupported."
)

CONTENT_CHECK = ("Перш ніж обрати вид, назви собі одним рядком, ЩО САМЕ стверджує доказ, і ЩО САМЕ "
                 "стверджує відповідь. Якщо ці два твердження про різні речі — це contradicted.")

GROUNDING_RULES: dict[str, str] = {
    "required": ("У цьому прогоні модель мала інструменти даних, тому твердження без доказу — "
                 "unsupported."),
    "optional": ("У цьому прогоні інструментів даних НЕ було, тому відповідь із власних знань "
                 "припустима: supported, якщо вона правильна, contradicted — якщо хибна."),
}

ABSENT_NOTE = ("Увага: усі виклики інструментів повідомили, що даних немає. Отже правильна "
               "відповідь тут — визнання незнання (abstain).")

TAIL = "Відповідай РІВНО одним JSON: {\"kind\":\"<вид>\",\"reason\":\"<чому>\"}"

CONTRAST_TAIL = (
    "Заповни поля ПО ПОРЯДКУ: спершу `доказ_каже` — що стверджує доказ; далі `відповідь_каже` — що "
    "стверджує відповідь; далі `той_самий_референт` — чи це про ОДНЕ Й ТЕ САМЕ; і лише тоді `kind`. "
    "Якщо `той_самий_референт` = false, то `kind` мусить бути `contradicted`."
)


def _loads_lenient(text: str) -> dict:
    """Шлюз не примушує схему, тому модель уміє обірвати JSON на півслові.

    Спостережено дослівно: `{"доказ_каже": …, "той_самий_референт": false, "kind": "contradicted"`
    і далі порожні рядки без закриття. Зміст правильний, гине лише синтаксис — усі 77 клітинок
    режиму `contrast` пішли в `parse_fail` саме через це. Тому дописуємо дужки, а не втрачаємо вердикт.
    """
    text = text.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("у відповіді немає обʼєкта")
    body = text[start:]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    trimmed = body.rstrip().rstrip(",")
    for _ in range(3):
        trimmed = trimmed.rstrip().rstrip(",")
        try:
            return json.loads(trimmed + "}" * (trimmed.count("{") - trimmed.count("}")))
        except json.JSONDecodeError:
            cut = max(trimmed.rfind(","), trimmed.rfind("\n"))
            if cut <= 0:
                break
            trimmed = trimmed[:cut]
    raise ValueError("не вдалося полагодити JSON вердикту")


class Verdict(BaseModel):
    accepted: bool
    kind: VerdictKind = "supported"
    reason: str = ""
    tokens: int = 0
    prompt_tokens: int = 0


def _grounded_system(grounding: str, absent: bool, *, contrast: bool = False) -> str:
    parts = [GROUNDED_HEAD, GROUNDING_RULES.get(grounding, GROUNDING_RULES["optional"])]
    if absent:
        parts.append(ABSENT_NOTE)
    elif not contrast:
        parts.append(CONTENT_CHECK)
    parts.append(CONTRAST_TAIL if contrast else TAIL)
    return " ".join(parts)


def verify(task, answer, router: ModelRouter, effort: EffortPolicy, *,
           evidence: list | None = None, seed: int = 0,
           trace: TracePort | None = None, run_id: str = "",
           mode: str = "basic", grounding: str = "optional", absent: bool = False) -> Verdict:
    if mode == "auto" and grounding != "required":
        return Verdict(accepted=True, kind="no_evidence", reason=NO_EVIDENCE_REASON, tokens=0)
    llm = router.route("judge")
    cfg = effort.effort("judge")
    ev = json.dumps(evidence, ensure_ascii=False) if evidence else "—"
    prompt = f"Задача: {task}\nВідповідь: {answer}\nДокази (результати інструментів): {ev}\n\nОдин JSON."
    grounded = mode != "basic"
    contrast = mode == "contrast"
    system = _grounded_system(grounding, absent, contrast=contrast) if grounded else SYSTEM
    schema = (CONTRAST_SCHEMA if contrast else GROUNDED_SCHEMA) if grounded else VERDICT_SCHEMA
    budget = max(cfg.max_tokens, CONTRAST_MIN_TOKENS) if contrast else cfg.max_tokens
    res = llm.generate_structured(prompt, schema, system=system,
                                  temperature=cfg.temperature, max_tokens=budget, seed=seed)
    try:
        payload = _loads_lenient(res.text)
        if grounded:
            kind: VerdictKind = payload["kind"]
            verdict = Verdict(accepted=kind in ACCEPTING, kind=kind,
                              reason=str(payload.get("reason", "")), tokens=res.usage.total,
                              prompt_tokens=res.usage.prompt_tokens)
        else:
            accepted = bool(payload["accepted"])
            verdict = Verdict(accepted=accepted, kind="supported" if accepted else "unsupported",
                              reason=str(payload.get("reason", "")), tokens=res.usage.total,
                              prompt_tokens=res.usage.prompt_tokens)
        ok = True
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        verdict = Verdict(accepted=False, kind="parse_fail", reason="verify_parse_fail",
                          tokens=res.usage.total, prompt_tokens=res.usage.prompt_tokens)
        ok = False
    if trace is not None:
        trace.emit(StepRecord(
            run_id=run_id, tick=0, agent="verifier", stage="judge", model=llm.model,
            lane=router.lane("judge"),
            prompt=prompt, raw_output=res.text, parsed=verdict.model_dump(),
            schema_valid=ok, world_valid=ok, reject_reason=None if ok else "verify_parse_fail",
            usage=res.usage, latency_ms=res.latency_ms, finish_reason=res.finish_reason, seed=seed,
        ))
    return verdict
