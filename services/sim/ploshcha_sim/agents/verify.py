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

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"accepted": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["accepted", "reason"],
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
    "• supported — твердження відповіді підтверджене доказами;\n"
    "• abstain — доказів немає або інструмент повідомив, що даних немає, І відповідь чесно це "
    "визнає, не вигадуючи змісту. Це ПРАВИЛЬНА поведінка, а не провал;\n"
    "• unsupported — відповідь стверджує зміст, під який доказу немає (вигадка);\n"
    "• contradicted — відповідь суперечить доказам.\n"
    "Довжина, впевненість тону й обсяг пояснень НЕ впливають на вид: коротке чесне «даних немає» — "
    "це abstain, а не unsupported."
)

GROUNDING_RULES: dict[str, str] = {
    "required": ("У цьому прогоні модель мала інструменти даних, тому твердження без доказу — "
                 "unsupported."),
    "optional": ("У цьому прогоні інструментів даних НЕ було, тому відповідь із власних знань "
                 "припустима: supported, якщо вона правильна, contradicted — якщо хибна."),
}

ABSENT_NOTE = ("Увага: усі виклики інструментів повідомили, що даних немає. Отже правильна "
               "відповідь тут — визнання незнання (abstain).")

TAIL = "Відповідай РІВНО одним JSON: {\"kind\":\"<вид>\",\"reason\":\"<чому>\"}"


class Verdict(BaseModel):
    accepted: bool
    kind: VerdictKind = "supported"
    reason: str = ""
    tokens: int = 0


def _grounded_system(grounding: str, absent: bool) -> str:
    parts = [GROUNDED_HEAD, GROUNDING_RULES.get(grounding, GROUNDING_RULES["optional"])]
    if absent:
        parts.append(ABSENT_NOTE)
    parts.append(TAIL)
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
    system = _grounded_system(grounding, absent) if grounded else SYSTEM
    schema = GROUNDED_SCHEMA if grounded else VERDICT_SCHEMA
    res = llm.generate_structured(prompt, schema, system=system,
                                  temperature=cfg.temperature, max_tokens=cfg.max_tokens, seed=seed)
    try:
        payload = json.loads(res.text)
        if grounded:
            kind: VerdictKind = payload["kind"]
            verdict = Verdict(accepted=kind in ACCEPTING, kind=kind,
                              reason=str(payload.get("reason", "")), tokens=res.usage.total)
        else:
            accepted = bool(payload["accepted"])
            verdict = Verdict(accepted=accepted, kind="supported" if accepted else "unsupported",
                              reason=str(payload.get("reason", "")), tokens=res.usage.total)
        ok = True
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        verdict = Verdict(accepted=False, kind="parse_fail", reason="verify_parse_fail",
                          tokens=res.usage.total)
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
